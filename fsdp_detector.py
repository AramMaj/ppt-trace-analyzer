"""FSDP2/TP/optimizer phase detection from the CPU event tree.

Transforms the flat ``LogicalOperation`` tree into per-layer phases
consumed by the bottleneck detector, timeline, and annotator.

Phase flow for one FSDP2 layer (ProfilerStep#X wrapper omitted):
  pre_forward (layers.X)
    all_gather (layers.X)           — mesh_fsdp NCCL all-gather (unshard)
    all_gather_copy_out (layers.X)  — split_with_sizes_copy into flat param
    RegisterPostBackwardFunction
  fwd_compute                        — aten::linear, scaled_dot_product_attention, T5LayerNorm
  post_forward (layers.X)

  pre_backward (layers.X)
    all_gather_copy_out (layers.X)  — re-gather after reshard_after_forward=True
  bwd_compute                        — autograd::engine::evaluate_function (backward stream)
  post_backward_reduce (layers.X)   — NCCL reduce-scatter (grad sync)
  post_backward_reshard (layers.X)  — free shard

Key validation traces (both live in traces/):
  8B TP trace (rank0_trace_8b_tp.json):  3 ProfilerSteps, 34 layers, tid=24529
    (fwd) + tid=24759 (bwd stream).  34 FSDP units.
  async TP trace (async-tensor-parall/): single ProfilerStep#9, 8 units, TP on mesh_tp.

Uses **conditional thread filtering**: when backward runs on a different
CUDA stream thread (tid=24759) than forward (tid=24529), forward compute
scans only the profiler thread and backward compute filters it out.
"""

from typing import Dict, List, Optional, Tuple, Set, Iterator, Any

from trace_parser import LogicalOperation, TraceParserHelper


FSDP_PREFIXES = ('FSDP::', 'FullyShardedDataParallel::')

TP_PG_DESC = 'mesh_tp'
FSDP_PG_DESC = 'mesh_fsdp'

NCCL_COLLECTIVE_NAMES = {
    'allgather': 'tp_all_gather',
    'all_gather': 'tp_all_gather',
    'reducescatter': 'tp_reduce_scatter',
    'reduce_scatter': 'tp_reduce_scatter',
    'allreduce': 'tp_all_reduce',
    'all_reduce': 'tp_all_reduce',
}


def _classify_nccl_kernel(ev: dict) -> str:
    """Classify a GPU NCCL kernel as ``tp_all_gather`` / ``tp_reduce_scatter`` / ``tp_all_reduce``.
    Uses ``_coll_name`` first, then name substring match, then ``'tp_other'``.
    """
    coll = ev.get('_coll_name', '')
    if not coll:
        name = ev.get('name', '')
        if 'AllGather' in name:
            return 'tp_all_gather'
        if 'ReduceScatter' in name:
            return 'tp_reduce_scatter'
        if 'AllReduce' in name:
            return 'tp_all_reduce'
        return 'tp_other'
    key = coll.lower().replace('-', '_')
    # Try exact match first, then substring
    if key in NCCL_COLLECTIVE_NAMES:
        return NCCL_COLLECTIVE_NAMES[key]
    for coll_key, phase in NCCL_COLLECTIVE_NAMES.items():
        if coll_key in key:
            return phase
    return 'tp_other'


def _has_fsdp_prefix(name: str) -> bool:
    """Match ``FSDP::`` or ``FullyShardedDataParallel::`` — used to skip
    FSDP-internal ops (all_gather, copy_out, pre_forward) when collecting
    fwd_compute / bwd_compute nodes.
    """
    return name.startswith(FSDP_PREFIXES)


def _fsdp_phase_name(phase: str, layer: str = '') -> List[str]:
    """Return both ``FSDP::`` and ``FullyShardedDataParallel::`` variants of a phase name
    (e.g. ``['FSDP::pre_forward (layers.0)', 'FullyShardedDataParallel::pre_forward (layers.0)']``).
    """
    names = []
    for p in FSDP_PREFIXES:
        name = f'{p}{phase}'
        if layer:
            names.append(f'{name} ({layer})')
        else:
            names.append(name)
    return names


class FSDP:
    """Container for all detected phases: per-layer FSDPUnit list, optimizer
    nodes, TP collectives, and computed aggregates.  Properties compute on the
    fly so they stay consistent if nodes are added post-construction.
    """
    def __init__(self):
        self.units: List["FSDPUnit"] = []
        self.optimizer_step: List[LogicalOperation] = []
        self.optimizer_zero_grad: List[LogicalOperation] = []
        self.tp_all_gather: List[dict] = []
        self.tp_reduce_scatter: List[dict] = []
        self.tp_all_reduce: List[dict] = []

    @property
    def tp_all_gather_gpu(self) -> float:
        """Total GPU wall time of TP all-gather kernels (µs)."""
        return sum(k.get('dur', 0) for k in self.tp_all_gather)

    @property
    def tp_reduce_scatter_gpu(self) -> float:
        """Total GPU wall time of TP reduce-scatter kernels (µs)."""
        return sum(k.get('dur', 0) for k in self.tp_reduce_scatter)

    @property
    def tp_all_reduce_gpu(self) -> float:
        """Total GPU wall time of TP all-reduce kernels (µs)."""
        return sum(k.get('dur', 0) for k in self.tp_all_reduce)

    @property
    def tp_total_gpu(self) -> float:
        """Total GPU wall time of all TP collectives (µs)."""
        return self.tp_all_gather_gpu + self.tp_reduce_scatter_gpu + self.tp_all_reduce_gpu

    @property
    def optimizer_gpu(self) -> float:
        """Total GPU duration attributed to Optimizer.step CPU nodes (µs)."""
        return sum(n.gpu_duration for n in self.optimizer_step)

    @property
    def optimizer_cpu(self) -> float:
        """Total CPU duration of Optimizer.step nodes (µs)."""
        return sum(n.cpu_duration for n in self.optimizer_step)

    @property
    def optimizer_wall(self) -> float:
        """Wall-clock span covering all optimizer-step nodes (µs)."""
        if not self.optimizer_step:
            return 0.0
        start = min(n.start_time for n in self.optimizer_step)
        end = max(n.end_time for n in self.optimizer_step)
        return end - start

    @property
    def optimizer_step_gpu_kernels(self) -> List[dict]:
        """Collect all GPU kernel events under all Optimizer.step nodes (recursive)."""
        kernels = []
        for n in self.optimizer_step:
            kernels.extend(n.direct_gpu_kernels if isinstance(n.direct_gpu_kernels, list) else [])
            for ch in _iter_logical(n):
                if ch.direct_gpu_kernels:
                    kernels.extend(ch.direct_gpu_kernels if isinstance(ch.direct_gpu_kernels, list) else [])
        return kernels

    @property
    def step_wall(self) -> float:
        """Wall-clock span from earliest phase to latest (including optimizer)."""
        if not self.units:
            return 0.0
        all_units = []
        for u in self.units:
            all_units.extend(u.all_gather_fwd)
            all_units.extend(u.fwd_compute)
            all_units.extend(u.all_gather_bwd)
            all_units.extend(u.bwd_compute)
            all_units.extend(u.reduce_scatter)
        all_units.extend(self.optimizer_step)
        all_units.extend(self.optimizer_zero_grad)
        if not all_units:
            return 0.0
        start = min(n.start_time for n in all_units)
        end = max(n.end_time for n in all_units)
        return end - start


class FSDPUnit:
    """One FSDP2 layer — stores CPU nodes for each of the five phases plus
    computed GPU kernel lists.  The core abstraction consumed by bottleneck
    detection, timeline, and annotation.

    ``fwd_compute_span`` / ``bwd_compute_span`` are the exact wall-clock
    boundaries of the compute phase (e.g. ``all_gather_copy_out.end`` →
    ``next_all_gather_copy_out.start``), stored separately from the node
    lists so that wall time is always a clean continuous interval even
    when the collected nodes are sparse or include anomalous kernels.
    """
    def __init__(self, layer_name: str):
        self.layer_name: str = layer_name
        self.all_gather_fwd: List[LogicalOperation] = []
        self.all_gather_bwd: List[LogicalOperation] = []
        self.reduce_scatter: List[LogicalOperation] = []
        self.fwd_compute: List[LogicalOperation] = []
        self.bwd_compute: List[LogicalOperation] = []
        self.fwd_compute_span: Optional[Tuple[float, float]] = None
        self.bwd_compute_span: Optional[Tuple[float, float]] = None
        self.all_gather_bwd_end: Optional[float] = None
        self.ag_bwd_supplement_us: float = 0.0

    @property
    def all_gather_fwd_gpu_kernels(self) -> List[dict]:
        """GPU kernel events under all ``all_gather_fwd`` CPU nodes."""
        kernels = []
        for n in self.all_gather_fwd:
            kernels.extend(n.direct_gpu_kernels if isinstance(n.direct_gpu_kernels, list) else [])
            for ch in _iter_logical(n):
                if ch.direct_gpu_kernels:
                    kernels.extend(ch.direct_gpu_kernels if isinstance(ch.direct_gpu_kernels, list) else [])
        return kernels

    @property
    def all_gather_bwd_gpu_kernels(self) -> List[dict]:
        """GPU kernel events under all ``all_gather_bwd`` CPU nodes."""
        kernels = []
        for n in self.all_gather_bwd:
            kernels.extend(n.direct_gpu_kernels if isinstance(n.direct_gpu_kernels, list) else [])
            for ch in _iter_logical(n):
                if ch.direct_gpu_kernels:
                    kernels.extend(ch.direct_gpu_kernels if isinstance(ch.direct_gpu_kernels, list) else [])
        return kernels

    @property
    def reduce_scatter_gpu_kernels(self) -> List[dict]:
        """GPU kernel events under all ``reduce_scatter`` CPU nodes."""
        kernels = []
        for n in self.reduce_scatter:
            kernels.extend(n.direct_gpu_kernels if isinstance(n.direct_gpu_kernels, list) else [])
            for ch in _iter_logical(n):
                if ch.direct_gpu_kernels:
                    kernels.extend(ch.direct_gpu_kernels if isinstance(ch.direct_gpu_kernels, list) else [])
        return kernels


def _iter_logical(node: LogicalOperation) -> Iterator[LogicalOperation]:
    """DFS pre-order traversal of a subtree — yields the node then its children recursively."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _extract_layer(name: str) -> str:
    """Extract the layer name from a parenthetical suffix (handles nested parens).
    ``'FSDP::pre_forward (layers.7)'`` → ``'layers.7'``.
    """
    if '(' not in name or not name.endswith(')'):
        return name
    start = name.index('(')
    depth = 0
    for i in range(start, len(name)):
        if name[i] == '(':
            depth += 1
        elif name[i] == ')':
            depth -= 1
            if depth == 0:
                return name[start + 1:i]
    return name


def _pick_latest_with_allgather(nodes: List[LogicalOperation]) -> LogicalOperation:
    """Pick the latest-starting node that has an ``FSDP::all_gather`` child.
    Stale ``pre_forward`` nodes from earlier steps may still be in the tree;
    this heuristic selects the one for the active step.
    """
    picks = [n for n in nodes if any('all_gather' in ch.name and _has_fsdp_prefix(ch.name) for ch in n.children)]
    return max(picks, key=lambda n: n.start_time) if picks else max(nodes, key=lambda n: n.start_time)


def _match_gpu_by_interval(gpu_events: List[dict], start: float, end: float) -> List[dict]:
    """GPU events fully contained in [start, end].  Debugging helper, not used in the main pipeline."""
    out = []
    ts = start
    for ev in gpu_events:
        ev_start = ev.get('ts', 0)
        ev_dur = ev.get('dur', 0)
        if ev_start >= ts and (ev_start + ev_dur) <= end:
            out.append(ev)
    return out


class StandardFSDPDetector:
    """Orchestrates seven sub-detectors into ``extract_fsdp_phases(roots)``.
    Each sub-detector is independently verifiable; the class manages
    cross-cutting concerns like conditional thread filtering.
    """
    def __init__(self, gpu_events: Optional[List[dict]] = None):
        self.gpu_events = gpu_events or []
        self.all_nodes = []

    def _detect_all_gather_fwd(self, roots: List[LogicalOperation], fsdp_units: FSDP):
        """Function searches ``FSDP::pre_forward (layer.X)`` event which consists of 
        ``FSDP::all_gather`` (NCCL all-gather that collects parameters) and 
        `all_gather_copy_out`` (Copies result into flat parameters)"""
        self.all_nodes = list(TraceParserHelper.iter_nodes(roots))
        for unit in fsdp_units.units:
            pre_fwd_list = []
            for name in _fsdp_phase_name('pre_forward', unit.layer_name):
                pre_fwd_list.extend(n for n in self.all_nodes if n.name == name)
            if not pre_fwd_list:
                continue
            pre_fwd = _pick_latest_with_allgather(pre_fwd_list)

            for ch in pre_fwd.children:
                if _has_fsdp_prefix(ch.name) and 'all_gather' in ch.name:
                    if ch not in unit.all_gather_fwd:
                        unit.all_gather_fwd.append(ch)

    def _detect_all_gather_bwd(self, roots: List[LogicalOperation], fsdp_units: FSDP):
        """Function searches ``FSDP::pre_backward (layer.X)`` event which consists of 
        ``FSDP::all_gather`` and  `all_gather_copy_out`` (mirrors forward) """
        for unit in fsdp_units.units:
            pre_bwd_list = []
            for name in _fsdp_phase_name('pre_backward', unit.layer_name):
                pre_bwd_list.extend(n for n in self.all_nodes if n.name == name)
            if not pre_bwd_list:
                continue
            pre_bwd = _pick_latest_with_allgather(pre_bwd_list)

            for ch in pre_bwd.children:
                if _has_fsdp_prefix(ch.name) and 'all_gather' in ch.name:
                    if ch not in unit.all_gather_bwd:
                        unit.all_gather_bwd.append(ch)

    @staticmethod
    def _has_other_tids(all_nodes, profiler_tid):
        """Check if any tree node lives on a thread other than *profiler_tid*.
        The 8B TP trace uses tid=24529 for the main profiler thread and
        tid=24759 for the backward CUDA stream — this heuristic detects that split.
        """
        if profiler_tid is None:
            return False
        return any(
            n.raw_event and n.raw_event.get('tid') != profiler_tid
            for n in all_nodes if n.raw_event
        )

    def _detect_fwd_cmp(self, roots: List[LogicalOperation], fsdp_units: FSDP, profiler_tid: int = None):
        """Function collects all compute operations (attn/MLP/norm) starting from its 
        layer's ``all_gather_copy_out.end_time`` and the **next** layer's 
        ``all_gather_copy_out.start`
        
        Using the next layer's copy_out start as the boundary (instead of
        ``pre_forward.start``) is correct for FSDP2 pipelining: the NCCL all-gather
        inside the next layer's ``pre_forward`` runs concurrently with this layer's
        compute.  The copy_out only starts after the all-gather finishes and the
        data has been split into contiguous buffers — that is the true end of
        this layer's compute window.

        Skips FSDP-internal ops (all_gather, copy_out), ProfilerStep markers,
        and Optimizer events.  When backward runs on a separate thread
        (``has_other_tids``, as with tid=24759 in the 8B trace), only the
        profiler thread (tid=24529) is matched — the backward-stream events
        are reserved for ``_detect_bwd_cmp``.
        """
        all_nodes = sorted(self.all_nodes, key=lambda n: n.start_time)
        units = fsdp_units.units
        has_other_tids = self._has_other_tids(all_nodes, profiler_tid)

        all_pre_fwd = {}
        for n in all_nodes:
            if _has_fsdp_prefix(n.name) and 'pre_forward' in n.name:
                all_pre_fwd[n.name] = n  # last ProfilerStep wins

        for i, unit in enumerate(units):
            copy_out_end = None
            for n in unit.all_gather_fwd:
                if 'all_gather_copy_out' in n.name:
                    copy_out_end = n.end_time

            if copy_out_end is None:
                continue

            if i + 1 < len(units):
                next_unit = units[i + 1]
                next_copy_out_start = None
                for ag_node in next_unit.all_gather_fwd:
                    if 'all_gather_copy_out' in ag_node.name:
                        if next_copy_out_start is None or ag_node.start_time < next_copy_out_start:
                            next_copy_out_start = ag_node.start_time
                if next_copy_out_start is not None:
                    end_time = next_copy_out_start
                else:
                    next_node = None
                    for name in _fsdp_phase_name('pre_forward', units[i + 1].layer_name):
                        next_node = all_pre_fwd.get(name)
                        if next_node:
                            break
                    end_time = next_node.start_time if next_node else copy_out_end
            else:
                post_fwd_list = []
                for name in _fsdp_phase_name('post_forward', unit.layer_name):
                    post_fwd_list.extend(n for n in all_nodes if n.name == name)
                if post_fwd_list:
                    post_fwd = max(post_fwd_list, key=lambda n: n.start_time)
                    end_time = post_fwd.end_time
                else:
                    end_time = copy_out_end

            unit.fwd_compute_span = (copy_out_end, end_time)

            for n in all_nodes:
                if has_other_tids and profiler_tid is not None:
                    n_tid = n.raw_event.get('tid') if n.raw_event else None
                    if n_tid != profiler_tid:
                        continue
                if n.start_time < end_time and copy_out_end < n.end_time <= end_time:
                    if _has_fsdp_prefix(n.name) or n.name.startswith('ProfilerStep') or n.name.startswith('Optimizer'):
                        continue
                    if n.name in ('backward_pass', 'forward_pass', 'root_pre_forward',
                                  'root_post_backward_callback', 'inputs_to_device',
                                  'cast_forward_inputs'):
                        continue
                    unit.fwd_compute.append(n)

    def _detect_bwd_cmp(self, roots: List[LogicalOperation], fsdp_units: FSDP, profiler_tid: int = None):
        """Function collects all ``autograd::engine::evaluate_function`` compute 
        operations between its layer's ``all_gather_copy_out.end_time`` 
        (from ``pre_backward``) and ``reduce_scatter.start_time``.

        The ``all_gather_copy_out`` end is more precise than ``pre_backward.end``
        — the latter may include trailing bookkeeping children after the copy_out.
        No thread filtering here: the time window alone is tight enough that
        nodes from the wrong thread won't match, and filtering on thread would
        miss steps whose backward runs on the same thread as forward.
        Falls back to ``post_backward_reduce`` name matching if the RS node
        was missed, then to ``pre_backward.start_time`` of the next layer for
        traces (e.g. gpu-server steps #4/#6/#8) that skip reduce-scatter.
        """
        all_nodes = sorted(self.all_nodes, key=lambda n: n.start_time)

        # All pre_backward nodes across every layer, sorted by start_time =
        # backward execution order (last forward layer's bwd completes first).
        all_pre_bwd_global = []
        for unit in fsdp_units.units:
            for name in _fsdp_phase_name('pre_backward', unit.layer_name):
                all_pre_bwd_global.extend(n for n in all_nodes if n.name == name)
        all_pre_bwd_global = sorted(all_pre_bwd_global, key=lambda n: n.start_time)

        # Step end from ProfilerStep roots — last-resort boundary for the final
        # backward layer when no RS or root_post_backward_callback exists.
        step_end = None
        for r in roots:
            if r.name.startswith('ProfilerStep#'):
                if step_end is None or r.end_time > step_end:
                    step_end = r.end_time

        # root_post_backward_callback fires after all layers' backward completes
        # — a more precise alternative to step_end for the final layer boundary.
        root_post_bwd = None
        root_post_bwd_names = _fsdp_phase_name('root_post_backward_callback', '')
        for n in all_nodes:
            if n.name in root_post_bwd_names:
                root_post_bwd = n
                break

        for unit in fsdp_units.units:
            all_pre_bwd = []
            for name in _fsdp_phase_name('pre_backward', unit.layer_name):
                all_pre_bwd.extend(n for n in all_nodes if n.name == name)

            rs_start = None
            if unit.reduce_scatter:
                rs_node = unit.reduce_scatter[0]
                rs_start = rs_node.start_time
                pre_bwd = None
                for n in all_pre_bwd:
                    if n.end_time <= rs_start:
                        if pre_bwd is None or (rs_start - n.end_time) < (rs_start - pre_bwd.end_time):
                            pre_bwd = n
            else:
                pre_bwd = all_pre_bwd[0] if all_pre_bwd else None
                for name in _fsdp_phase_name('post_backward_reduce', unit.layer_name):
                    rs_list = [n for n in all_nodes if n.name == name]
                    if rs_list:
                        rs_start = rs_list[0].start_time
                        break

            # Fallback for traces whose backward thread runs no reduce-scatter
            # (gpu-server steps #4/#6/#8): use the *next* pre_backward.start_time
            # as the end boundary instead.  Backward execution walks layers in
            # reverse order, so each layer's compute window ends when the next
            # pre_backward fires.
            if rs_start is None and all_pre_bwd_global:
                for pi, pb in enumerate(all_pre_bwd_global):
                    if _extract_layer(pb.name) == unit.layer_name:
                        pre_bwd = pb
                        if pi + 1 < len(all_pre_bwd_global):
                            rs_start = all_pre_bwd_global[pi + 1].start_time
                        elif root_post_bwd is not None:
                            rs_start = root_post_bwd.start_time
                        elif step_end is not None:
                            rs_start = step_end
                        break

            copy_out_end_bwd = None
            for ag_node in unit.all_gather_bwd:
                if 'all_gather_copy_out' in ag_node.name:
                    if copy_out_end_bwd is None or ag_node.end_time > copy_out_end_bwd:
                        copy_out_end_bwd = ag_node.end_time
            all_gather_end = copy_out_end_bwd if copy_out_end_bwd is not None else (pre_bwd.end_time if pre_bwd else None)
            unit.all_gather_bwd_end = all_gather_end

            if all_gather_end is None or rs_start is None:
                continue

            unit.bwd_compute_span = (all_gather_end, rs_start)

            for n in all_nodes:
                if n.start_time < rs_start and all_gather_end < n.end_time <= rs_start:
                    if _has_fsdp_prefix(n.name) or n.name.startswith('ProfilerStep') or n.name.startswith('Optimizer'):
                        continue
                    if n.name in ('backward_pass', 'forward_pass', 'root_pre_forward',
                                  'root_post_backward_callback', 'inputs_to_device',
                                  'cast_forward_inputs'):
                        continue
                    unit.bwd_compute.append(n)

    def _detect_tp_gpu(self, fsdp: FSDP):
        """Classify GPU events with ``_pg_desc == 'mesh_tp'`` into TP collectives.
        FSDP and TP collectives are distinguished here so per-layer metrics can
        report ``tp_comm_ratio`` separately from ``fsdp_comm_ratio`` — critical
        for diagnosing async TP overlap efficiency.
        """
        if not self.gpu_events:
            return

        for ev in self.gpu_events:
            pg = ev.get('_pg_desc', '')
            if pg != TP_PG_DESC:
                continue
            cat = _classify_nccl_kernel(ev)
            if cat == 'tp_all_gather':
                fsdp.tp_all_gather.append(ev)
            elif cat == 'tp_reduce_scatter':
                fsdp.tp_reduce_scatter.append(ev)
            elif cat == 'tp_all_reduce':
                fsdp.tp_all_reduce.append(ev)

    def _detect_optimizer_step(self, roots: List[LogicalOperation], fsdp_units: FSDP):
        """Collect ``Optimizer.step#N`` / ``Optimizer.zero_grad#N`` CPU nodes.
        In the 8B TP trace, fused ADAMW kernels execute ~727ms after the CPU
        ProfilerStep#3 ends — these nodes are collected but their GPU time may
        fall outside the active step's time margin (handled in ``sanitize_optimizer``).
        """
        for n in self.all_nodes:
            if n.name.startswith('Optimizer.step#'):
                if n not in fsdp_units.optimizer_step:
                    fsdp_units.optimizer_step.append(n)
            if n.name.startswith('Optimizer.zero_grad#'):
                if n not in fsdp_units.optimizer_zero_grad:
                    fsdp_units.optimizer_zero_grad.append(n)

    def _detect_reduce_scatter(self, roots: List[LogicalOperation], fsdp_units: FSDP):
        """FSDP2 gradient sync: latest ``FSDP::post_bkward_reduce (layer.X)`` node
        per unit.  This is the NCCL reduce-scatter on ``mesh_fsdp`` that averages
        gradients across the FSDP shard group.  In the 8B trace, these nodes appear
        on the backward stream thread (tid=24759) interleaved with autograd.

        Falls back to NCCL GPU kernel matching when no CPU node is found for a
        layer: searches for reduce-scatter NCCL kernels whose ``_pg_desc`` or
        ``_coll_name`` indicates FSDP (not TP), then wraps them in a synthetic
        ``LogicalOperation`` anchored to the last layer's backward compute window.
        """
        for unit in fsdp_units.units:
            rs_list = []
            for name in _fsdp_phase_name('post_backward_reduce', unit.layer_name):
                rs_list.extend(n for n in self.all_nodes if n.name == name)
            if rs_list:
                rs_node = max(rs_list, key=lambda n: n.start_time)
                if rs_node not in unit.reduce_scatter:
                    unit.reduce_scatter.append(rs_node)
                continue

            # Fallback: find NCCL ReduceScatter GPU kernels for this layer
            bwd_span = unit.bwd_compute_span
            if bwd_span is None:
                continue
            for ev in self.gpu_events:
                pg = ev.get('_pg_desc', '')
                coll = ev.get('_coll_name', '')
                if pg == FSDP_PG_DESC and 'reduce_scatter' in coll.lower():
                    ev_start = ev.get('ts', 0)
                    ev_end = ev_start + ev.get('dur', 0)
                    if ev_start >= bwd_span[0] and ev_end <= bwd_span[1] + 5000:
                        import copy
                        fake = copy.deepcopy(unit.bwd_compute[0]) if unit.bwd_compute else None
                        if fake is not None:
                            fake.name = f'FSDP::post_backward_reduce ({unit.layer_name})'
                            fake.start_time = ev_start
                            fake.end_time = ev_end
                            fake.cpu_duration = 0.0
                            fake.gpu_duration = ev.get('dur', 0)
                            fake.direct_gpu_kernels = [ev]
                            fake.direct_gpu_duration = ev.get('dur', 0)
                            fake.children = []
                            fake.external_ids = set()
                            if fake not in unit.reduce_scatter:
                                unit.reduce_scatter.append(fake)
                                break

    def _profiler_tid(self, roots: List[LogicalOperation]):
        """Thread ID of the first ProfilerStep# event — used for conditional
        thread filtering.  Returns ``24529`` for the 8B TP trace (matching the
        profiler's main thread).  Falls back to ``None`` if no ProfilerStep
        is found (single-step traces without the marker).
        """
        for root in roots:
            raw = root.raw_event or {}
            if raw.get("name", "").startswith("ProfilerStep#") and raw.get("ph") == "X":
                return raw.get("tid")
        return None

    def extract_fsdp_phases(self, roots: List[LogicalOperation]) -> FSDP:
        """Run all seven sub-detectors in dependency order and return the populated ``FSDP`` container.

        Order matters:
        1. ``_detect_all_gather_fwd`` — identifies layers (pre_forward → layer name)
        2. ``_detect_fwd_cmp`` — needs all_gather_fwd populated for copy_out_end
        3. ``_detect_all_gather_bwd`` — needs layer list from step 1
        4. ``_detect_reduce_scatter`` — needed before bwd_cmp (provides rs_start)
        5. ``_detect_bwd_cmp`` — needs pre_backward + reduce_scatter boundaries
        6. ``_detect_optimizer_step`` — independent, collects Optimizer.step/zero_grad
        7. ``_detect_tp_gpu`` — independent, classifies mesh_tp GPU kernels from gpu_events

        This single call transforms the raw CPU tree into structured phases consumed
        by the bottleneck detector, report, timeline, and annotator.
        """
        fsdp = FSDP()
        all_nodes = list(TraceParserHelper.iter_nodes(roots))

        pre_fwd_nodes = [n for n in all_nodes
                         if _has_fsdp_prefix(n.name) and 'pre_forward' in n.name and '(' in n.name]
        pre_fwd_nodes.sort(key=lambda n: n.start_time)

        seen = set()
        for n in pre_fwd_nodes:
            layer = _extract_layer(n.name)
            if layer not in seen:
                seen.add(layer)
                fsdp.units.append(FSDPUnit(layer))

        profiler_tid = self._profiler_tid(roots)

        self._detect_all_gather_fwd(roots, fsdp)
        self._detect_fwd_cmp(roots, fsdp, profiler_tid)
        self._detect_all_gather_bwd(roots, fsdp)
        self._detect_reduce_scatter(roots, fsdp)
        self._detect_bwd_cmp(roots, fsdp, profiler_tid)
        self._detect_optimizer_step(roots, fsdp)
        self._detect_tp_gpu(fsdp)

        return fsdp