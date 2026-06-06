"""
Main FSDP Recognition Logic

=> Pre Forward (layer X) in der CPU finden => Alle GPU Kernel in FSDPUnit.all_gather einfügen

FSDP Trace phase pattern (ProfilerStep#X wraps everything):
  Forward:
    pre_forward (layers.X)
      all_gather (layers.X)      -- NCCL all-gather (CPU + GPU kernels)
        aten::empty, aten::split_with_sizes, fsdp::all_gather_copy_in, c10d::_allgather_base_
        NCCL GPU kernels: ncclDevKernel_AllGather_RING_LL ...
      all_gather_copy_out (layers.X)  -- copy-out after all-gather
        fsdp::split_with_sizes_copy
      RegisterPostBackwardFunction
    ... forward compute ops (aten::linear, scaled_dot_product_attention, rms_norm ...)
    post_forward (layers.X)

  Backward (reverse order, layers.31 .. layers.0):
    pre_backward (layers.X)
      all_gather_copy_out (layers.X)  -- re-all-gather for backward
      backward_prefetch for layers.X-1
    ... autograd::engine::evaluate_function:* (backward compute)
    post_backward_accumulate (layers.X)
    post_backward_rs_wait (layers.X)
    post_backward_reduce (layers.X)  -- reduce-scatter
      fsdp::chunk_cat, c10d::_reduce_scatter_base_
      NCCL GPU kernels: ncclDevKernel_ReduceScatter_RING_LL ...
    post_backward_reshard (layers.X)
"""

from typing import Dict, List, Optional, Tuple, Set, Iterator, Any
from enum import Enum

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
    return name.startswith(FSDP_PREFIXES)


def _fsdp_phase_name(phase: str, layer: str = '') -> List[str]:
    names = []
    for p in FSDP_PREFIXES:
        name = f'{p}{phase}'
        if layer:
            names.append(f'{name} ({layer})')
        else:
            names.append(name)
    return names


class FSDP:
    def __init__(self):
        self.units: List["FSDPUnit"] = []
        self.optimizer_step: List[LogicalOperation] = []
        self.optimizer_zero_grad: List[LogicalOperation] = []
        self.tp_all_gather: List[dict] = []
        self.tp_reduce_scatter: List[dict] = []
        self.tp_all_reduce: List[dict] = []

    @property
    def tp_all_gather_gpu(self) -> float:
        return sum(k.get('dur', 0) for k in self.tp_all_gather)

    @property
    def tp_reduce_scatter_gpu(self) -> float:
        return sum(k.get('dur', 0) for k in self.tp_reduce_scatter)

    @property
    def tp_all_reduce_gpu(self) -> float:
        return sum(k.get('dur', 0) for k in self.tp_all_reduce)

    @property
    def tp_total_gpu(self) -> float:
        return self.tp_all_gather_gpu + self.tp_reduce_scatter_gpu + self.tp_all_reduce_gpu

    @property
    def optimizer_gpu(self) -> float:
        return sum(n.gpu_duration for n in self.optimizer_step)

    @property
    def optimizer_cpu(self) -> float:
        return sum(n.cpu_duration for n in self.optimizer_step)

    @property
    def optimizer_wall(self) -> float:
        if not self.optimizer_step:
            return 0.0
        start = min(n.start_time for n in self.optimizer_step)
        end = max(n.end_time for n in self.optimizer_step)
        return end - start

    @property
    def optimizer_step_gpu_kernels(self) -> List[dict]:
        kernels = []
        for n in self.optimizer_step:
            kernels.extend(n.direct_gpu_kernels if isinstance(n.direct_gpu_kernels, list) else [])
            for ch in _iter_logical(n):
                if ch.direct_gpu_kernels:
                    kernels.extend(ch.direct_gpu_kernels if isinstance(ch.direct_gpu_kernels, list) else [])
        return kernels

    @property
    def step_wall(self) -> float:
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
    def __init__(self, layer_name: str):
        self.layer_name: str = layer_name
        self.all_gather_fwd: List[LogicalOperation] = []
        self.all_gather_bwd: List[LogicalOperation] = []
        self.reduce_scatter: List[LogicalOperation] = []
        self.fwd_compute: List[LogicalOperation] = []
        self.bwd_compute: List[LogicalOperation] = []

    @property
    def all_gather_fwd_gpu_kernels(self) -> List[dict]:
        kernels = []
        for n in self.all_gather_fwd:
            kernels.extend(n.direct_gpu_kernels if isinstance(n.direct_gpu_kernels, list) else [])
            for ch in _iter_logical(n):
                if ch.direct_gpu_kernels:
                    kernels.extend(ch.direct_gpu_kernels if isinstance(ch.direct_gpu_kernels, list) else [])
        return kernels

    @property
    def all_gather_bwd_gpu_kernels(self) -> List[dict]:
        kernels = []
        for n in self.all_gather_bwd:
            kernels.extend(n.direct_gpu_kernels if isinstance(n.direct_gpu_kernels, list) else [])
            for ch in _iter_logical(n):
                if ch.direct_gpu_kernels:
                    kernels.extend(ch.direct_gpu_kernels if isinstance(ch.direct_gpu_kernels, list) else [])
        return kernels

    @property
    def reduce_scatter_gpu_kernels(self) -> List[dict]:
        kernels = []
        for n in self.reduce_scatter:
            kernels.extend(n.direct_gpu_kernels if isinstance(n.direct_gpu_kernels, list) else [])
            for ch in _iter_logical(n):
                if ch.direct_gpu_kernels:
                    kernels.extend(ch.direct_gpu_kernels if isinstance(ch.direct_gpu_kernels, list) else [])
        return kernels


def _iter_logical(node: LogicalOperation) -> Iterator[LogicalOperation]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _extract_layer(name: str) -> str:
    """Extract layer name from parens, handling nested parentheses."""
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
    picks = [n for n in nodes if any('all_gather' in ch.name and _has_fsdp_prefix(ch.name) for ch in n.children)]
    return max(picks, key=lambda n: n.start_time) if picks else max(nodes, key=lambda n: n.start_time)


def _match_gpu_by_interval(gpu_events: List[dict], start: float, end: float) -> List[dict]:
    out = []
    ts = start
    for ev in gpu_events:
        ev_start = ev.get('ts', 0)
        ev_dur = ev.get('dur', 0)
        if ev_start >= ts and (ev_start + ev_dur) <= end:
            out.append(ev)
    return out


class StandardFSDPDetector:
    def __init__(self, gpu_events: Optional[List[dict]] = None):
        self.gpu_events = gpu_events or []

    def _detect_all_gather_fwd(self, roots: List[LogicalOperation], fsdp_units: FSDP):
        all_nodes = list(TraceParserHelper.iter_nodes(roots))
        for unit in fsdp_units.units:
            pre_fwd_list = []
            for name in _fsdp_phase_name('pre_forward', unit.layer_name):
                pre_fwd_list.extend(n for n in all_nodes if n.name == name)
            if not pre_fwd_list:
                continue
            pre_fwd = _pick_latest_with_allgather(pre_fwd_list)

            for ch in pre_fwd.children:
                if _has_fsdp_prefix(ch.name) and 'all_gather' in ch.name:
                    if ch not in unit.all_gather_fwd:
                        unit.all_gather_fwd.append(ch)

    def _detect_all_gather_bwd(self, roots: List[LogicalOperation], fsdp_units: FSDP):
        all_nodes = list(TraceParserHelper.iter_nodes(roots))
        for unit in fsdp_units.units:
            pre_bwd_list = []
            for name in _fsdp_phase_name('pre_backward', unit.layer_name):
                pre_bwd_list.extend(n for n in all_nodes if n.name == name)
            if not pre_bwd_list:
                continue
            pre_bwd = _pick_latest_with_allgather(pre_bwd_list)

            for ch in pre_bwd.children:
                if _has_fsdp_prefix(ch.name) and 'all_gather_copy_out' in ch.name:
                    if ch not in unit.all_gather_bwd:
                        unit.all_gather_bwd.append(ch)

    def _detect_fwd_cmp(self, roots: List[LogicalOperation], fsdp_units: FSDP, profiler_tid: int = None):
        all_nodes = sorted(TraceParserHelper.iter_nodes(roots), key=lambda n: n.start_time)
        units = fsdp_units.units

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
                    all_pre_bwd = sorted(
                        [n for n in all_nodes if _has_fsdp_prefix(n.name) and 'pre_backward' in n.name],
                        key=lambda n: n.start_time
                    )
                    end_time = post_fwd.end_time
                    for n in all_pre_bwd:
                        if n.start_time >= copy_out_end:
                            end_time = n.start_time
                            break
                else:
                    end_time = copy_out_end

            for n in all_nodes:
                if profiler_tid is not None:
                    n_tid = n.raw_event.get('tid') if n.raw_event else None
                    if n_tid != profiler_tid:
                        continue
                if n.start_time >= copy_out_end and n.end_time <= end_time:
                    if _has_fsdp_prefix(n.name) or n.name.startswith('ProfilerStep') or n.name.startswith('Optimizer'):
                        continue
                    unit.fwd_compute.append(n)

    def _detect_bwd_cmp(self, roots: List[LogicalOperation], fsdp_units: FSDP, profiler_tid: int = None):
        all_nodes = sorted(TraceParserHelper.iter_nodes(roots), key=lambda n: n.start_time)

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

            all_gather_end = pre_bwd.end_time if pre_bwd else None

            if all_gather_end is None or rs_start is None:
                continue

            for n in all_nodes:
                if profiler_tid is not None:
                    n_tid = n.raw_event.get('tid') if n.raw_event else None
                    if n_tid == profiler_tid:
                        continue
                if n.start_time >= all_gather_end and n.end_time <= rs_start:
                    if _has_fsdp_prefix(n.name) or n.name.startswith('ProfilerStep') or n.name.startswith('Optimizer'):
                        continue
                    unit.bwd_compute.append(n)

    def _detect_tp_gpu(self, fsdp: FSDP):
        """Classify GPU NCCL kernels by process group into TP vs FSDP categories."""
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
        all_nodes = list(TraceParserHelper.iter_nodes(roots))
        for n in all_nodes:
            if n.name.startswith('Optimizer.step#'):
                if n not in fsdp_units.optimizer_step:
                    fsdp_units.optimizer_step.append(n)
            if n.name.startswith('Optimizer.zero_grad#'):
                if n not in fsdp_units.optimizer_zero_grad:
                    fsdp_units.optimizer_zero_grad.append(n)

    def _detect_reduce_scatter(self, roots: List[LogicalOperation], fsdp_units: FSDP):
        all_nodes = list(TraceParserHelper.iter_nodes(roots))
        for unit in fsdp_units.units:
            rs_list = []
            for name in _fsdp_phase_name('post_backward_reduce', unit.layer_name):
                rs_list.extend(n for n in all_nodes if n.name == name)
            if rs_list:
                rs_node = max(rs_list, key=lambda n: n.start_time)
                if rs_node not in unit.reduce_scatter:
                    unit.reduce_scatter.append(rs_node)

    def _profiler_tid(self, roots: List[LogicalOperation]):
        for root in roots:
            raw = root.raw_event or {}
            if raw.get("name", "").startswith("ProfilerStep#") and raw.get("ph") == "X":
                return raw.get("tid")
        return None

    def extract_fsdp_phases(self, roots: List[LogicalOperation]) -> FSDP:
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
