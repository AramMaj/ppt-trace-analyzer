"""
Per-layer FSDP2/TP metrics, bottleneck classification, and text report generation.

Six sections:
1. Low-level helpers (GPU kernel collection, NCCL vs copy vs compute classification)
2. Overlap/pipeline decomposition (sweep-line across layer wall spans)
3. Memory collection from ``[memory]`` trace events
4. ModelConfig dataclass — stores model hyperparameters and computes estimated FLOPs
5. Metrics class — per-FSDP-shard-group computed values (GPU time, comm ratio, overlap,
   compute-to-comm ratio, overlap efficiency, MFU/HFU, tokens/sec/GPU, etc.)
6. Bottlenecks class — threshold-based detection of 16+ bottleneck types (compute-bound,
   comm-bound, copy-heavy all-gather, async TP overlap asymmetry, host-bound,
    exposed all-gather, HBM bandwidth, sync TP on critical path, NVLink saturation, etc.)
7. Report class — aggregation, formatting, JSON markers
"""

import json
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from trace_parser import LogicalOperation, TraceParserHelper
from fsdp_detector import FSDP, FSDPUnit, FSDP_PREFIXES, FSDP_PG_DESCS, _extract_layer

# NB: Own DFS in _collect_gpu_kernels rather than reusing _iter_logical from fsdp_detector.


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _collect_gpu_kernels(nodes):
    """DFS-collect all ``direct_gpu_kernels`` (CUDA kernels, NCCL collectives,
    memcpy ops) from a list of LogicalOperation nodes and their subtrees.
    Uses an explicit stack to avoid Python recursion limits on deep FSDP trees.
    Only populated after ``attribute_gpu_kernel_with_logical_operation``.
    """
    kernels = []
    stack = list(nodes)
    while stack:
        n = stack.pop()
        if n.direct_gpu_kernels:
            kernels.extend(n.direct_gpu_kernels)
        stack.extend(n.children)
    return kernels


def _collect_nccl_kernel_time(nodes):
    """DFS-sum of AllGather NCCL GPU kernel durations from *nodes* and their subtrees.

    Only AllGather kernels (not ReduceScatter, AllReduce, etc.) are counted
    because this function is used to capture backward-prefetch all-gather
    kernels that live structurally inside backward compute but are functionally
    all-gather backward work.  Including other NCCL types would inflate the
    all-gather backward metric with ReduceScatter time.
    """
    total = 0.0
    stack = list(nodes)
    while stack:
        n = stack.pop()
        for gpu in (n.direct_gpu_kernels or []):
            name = gpu.get('name', '').lower()
            if 'nccl' in name and ('allgather' in name or 'all_gather' in name):
                total += gpu.get('dur', 0)
        stack.extend(n.children)
    return total


def _classify_kernel(name: str) -> str:
    """Substring-based classification: ``nccl`` / ``copy`` / ``compute``.
    NCCL patterns checked first to avoid false-positive "copy" classification on NCCL names.
    """
    lower = name.lower()
    if any(p in lower for p in ('nccl', 'allgather', 'all_gather', 'allreduce',
                                 'all_reduce', 'reducescatter', 'reduce_scatter')):
        return 'nccl'
    if any(p in lower for p in ('copy', 'memcpy', 'memset')):
        return 'copy'
    return 'compute'


def _is_fsdp_name(name: str) -> bool:
    """Check whether a node name matches one of the known FSDP name prefixes."""
    return name.startswith(FSDP_PREFIXES)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _phase_gpu_time(nodes: List[LogicalOperation],
                    seen: Optional[Set[Tuple[float, float]]] = None) -> float:
    """Unique GPU duration across all nodes' subtrees.

    Collects direct GPU-kernel events from the entire subtree of each node
    (via DFS), deduplicates by ``(timestamp, duration)``, and returns the
    sum.  This avoids over-counting when ancestor-descendant pairs appear
    in the same node list (as happens with time-window-based phase attribution
    under ``torch.compile``, where a single ``CompiledFunctionBackward`` node
    wraps many child events that all fall inside the same phase window).

    When *seen* is provided, it is used as the dedup set instead of a local
    one.  This enables global deduplication across multiple calls so that the
    same GPU kernel appearing in different layers' phase nodes (e.g. via the
    ``record_param_comms`` CPU wrapper that overlaps two adjacent layers) is
    counted exactly once.  The return value reflects only *new* kernels added
    by this call (*seen* is updated in place).

    When subtrees are disjoint (the common case) the result is identical to
    ``sum(n.gpu_duration for n in nodes)`` but never over-counts.
    """
    local: Set[Tuple[float, float]] = set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        for gpu in (n.direct_gpu_kernels or []):
            dur = gpu.get('dur', 0)
            if dur > 0:
                local.add((gpu.get('ts', 0), dur))
        stack.extend(n.children)
    if seen is not None:
        new = local - seen
        seen.update(local)
        return sum(dur for _, dur in new)
    return sum(dur for _, dur in local)


def _phase_gpu_time_breakdown(nodes: List[LogicalOperation],
                               seen: Optional[Set[Tuple[float, float]]] = None,
                               allowed_pg: Optional[Set[str]] = None,
                              ) -> Tuple[float, float, float]:
    """Split GPU time from nodes' subtrees into ReduceScatter, AllReduce, and other.

    Same tree walk and dedup logic as :func:`_phase_gpu_time`, but returns
    ``(reduce_scatter_us, all_reduce_us, other_us)`` classified by kernel name.

    *allowed_pg* restricts collection to GPU kernels whose ``_pg_desc`` is in
    the given set (useful for excluding TP kernels from FSDP phase breakdowns).
    """
    local: Set[Tuple[float, float]] = set()
    local_names: Dict[Tuple[float, float], str] = {}
    stack = list(nodes)
    while stack:
        n = stack.pop()
        for gpu in (n.direct_gpu_kernels or []):
            if allowed_pg is not None and gpu.get('_pg_desc', '') not in allowed_pg:
                continue
            dur = gpu.get('dur', 0)
            if dur > 0:
                key = (gpu.get('ts', 0), dur)
                local.add(key)
                if key not in local_names:
                    local_names[key] = gpu.get('name', '')
        stack.extend(n.children)
    if seen is not None:
        new = local - seen
        seen.update(local)
    else:
        new = local

    rs = ar = other = 0.0
    for ts, dur in new:
        name = local_names.get((ts, dur), '')
        if 'ReduceScatter' in name:
            rs += dur
        elif 'AllReduce' in name:
            ar += dur
        else:
            other += dur
    return rs, ar, other


def _phase_gpu_time_direct(nodes: List[LogicalOperation]) -> float:
    """Sum of **direct** (non-overlapping) GPU duration across nodes.

    Unlike :func:`_phase_gpu_time`, this uses ``direct_gpu_duration`` which
    excludes GPU time from descendant nodes.  When the same node list contains
    both parent and child CPU operations (as happens with time-window-based
    phase attribution), ``direct_gpu_duration`` counts each GPU kernel exactly
    once and avoids double-counting container wrappers.
    """
    return sum(n.direct_gpu_duration for n in nodes)


def _phase_cpu_time(nodes: List[LogicalOperation]) -> float:
    """Sum of inclusive CPU dispatch duration (µs).  Same caveat: not a union."""
    return sum(n.cpu_duration for n in nodes)


def _phase_wall_time(nodes: List[LogicalOperation], fallback_span: Optional[Tuple[float, float]] = None) -> float:
    """Union wall-clock span — earliest ``start_time`` to latest ``end_time``
    across *nodes*.  When *nodes* is empty, *fallback_span* ``(start, end)``
    provides a best-effort substitute so callers (the annotator, bottleneck
    report) always get a span, even for degenerate layers.

    Gaps between phases (e.g. pipeline bubbles between reduce-scatter and
    the next all-gather) are included.
    """
    if not nodes:
        if fallback_span is not None:
            return fallback_span[1] - fallback_span[0]
        return 0.0
    start = min(n.start_time for n in nodes)
    end = max(n.end_time for n in nodes)
    return end - start


def _phase_wall_span(nodes):
    """``(start, end)`` tuple version of ``_phase_wall_time``.  Used by the
    annotator for Chrome Trace flow-event boundaries between consecutive
    FSDP phases (e.g. all-gather→fwd compute).
    Returns ``None`` for empty input.
    """
    if not nodes:
        return None
    start = min(n.start_time for n in nodes)
    end = max(n.end_time for n in nodes)
    return (start, end)


# ---------------------------------------------------------------------------
# Overlap / pipeline decomposition
# ---------------------------------------------------------------------------

def _compute_overlap_metrics(units: List['FSDPUnit']) -> dict:
    """Sweep-line decomposition of the FSDP2 pipeline step into serial / overlap
    / idle time.  Each layer's wall interval (all-gather→compute→RS) is one
    span; counter=0 → pipeline bubble, =1 → serial (only one shard group active),
    ≥2 → overlap (FSDP2 pipelining working).  This is the single most important
    efficiency metric for pipelined schedules (FSDP2 + TP).
    """
    start_end_pairs = []
    for unit in units:
        all_events = (unit.all_gather_fwd + unit.fwd_compute + unit.all_gather_bwd
                      + unit.bwd_compute + unit.reduce_scatter)
        if all_events:
            start = min(n.start_time for n in all_events)
            end = max(n.end_time for n in all_events)
            start_end_pairs.append((start, end))

    if not start_end_pairs:
        return {'overlap_time': 0.0, 'serial_time': 0.0, 'idle_time': 0.0,
                'step_wall': 0.0, 'overlap_ratio': 0.0, 'serial_ratio': 1.0,
                'idle_ratio': 0.0}

    events = []
    for start, end in start_end_pairs:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()

    active = 0
    prev_ts = events[0][0]
    serial_time = 0.0
    overlap_time = 0.0
    idle_time = 0.0

    for ts, delta in events:
        dur = ts - prev_ts
        if dur > 0:
            if active == 0:
                idle_time += dur
            elif active == 1:
                serial_time += dur
            else:
                overlap_time += dur
        active += delta
        prev_ts = ts

    total = serial_time + overlap_time + idle_time
    step_wall = max(e for _, e in start_end_pairs) - min(s for s, _ in start_end_pairs)

    return {
        'overlap_time': overlap_time,
        'serial_time': serial_time,
        'idle_time': idle_time,
        'step_wall': step_wall,
        'overlap_ratio': overlap_time / (serial_time + overlap_time) if (serial_time + overlap_time) > 0 else 0.0,
        'serial_ratio': serial_time / total if total > 0 else 1.0,
        'idle_ratio': idle_time / total if total > 0 else 0.0,
    }


def _merge_intervals(intervals):
    """Merge a list of ``(start, end)`` tuples into non-overlapping spans and
    return the total covered wall-clock time.
    """
    if not intervals:
        return 0.0
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _collect_kernel_intervals(phase_kernel_lists):
    """Extract ``(start, end)`` timestamps from one or more GPU-kernel lists
    (each element is a ``dict`` with ``'ts'`` and ``'dur'`` keys).
    """
    intervals = []
    for kernels in phase_kernel_lists:
        for k in kernels:
            s = k.get('ts', 0)
            intervals.append((s, s + k.get('dur', 0)))
    return intervals


def _compute_gpu_active_time(unit: 'FSDPUnit') -> float:
    """Total wall-clock time where at least one GPU kernel (attributed to this
    unit) was executing.  Merges overlapping intervals across streams so that
    parallel kernels on different CUDA streams are NOT double-counted.

    Returns ``0.0`` when the unit has no attributed GPU kernels.
    """
    intervals = _collect_kernel_intervals([
        unit.all_gather_fwd_gpu_kernels,
        unit.all_gather_bwd_gpu_kernels,
        unit.reduce_scatter_gpu_kernels,
        _collect_gpu_kernels(unit.fwd_compute),
        _collect_gpu_kernels(unit.bwd_compute),
    ])
    return _merge_intervals(intervals)


def _get_layer_gpu_span(unit: 'FSDPUnit'):
    """Collect all GPU kernel timestamps across every phase for an FSDP unit,
    returning ``(gpu_start, gpu_end)`` as the union span of all GPU kernels.

    Returns ``None`` if the unit has no attributed GPU kernels (no kernel
    data in the trace, or attribution failed).
    """
    all_kernels = []
    for phase_kernels in [
        unit.all_gather_fwd_gpu_kernels,
        unit.all_gather_bwd_gpu_kernels,
        unit.reduce_scatter_gpu_kernels,
        _collect_gpu_kernels(unit.fwd_compute),
        _collect_gpu_kernels(unit.bwd_compute),
    ]:
        all_kernels.extend(phase_kernels)
    if not all_kernels:
        return None
    min_start = min(k.get('ts', 0) for k in all_kernels)
    max_end = max(k.get('ts', 0) + k.get('dur', 0) for k in all_kernels)
    return (min_start, max_end)


def _compute_cross_layer_overlap(units):
    """GPU-execution overlap ratio between consecutive FSDP layers.

    For each adjacent pair ``(N, N+1)``:
        overlap = max(0, min(end_N, end_N1) - max(start_N, start_N1))
        pair_ratio = overlap / max(span_N, span_N1)

    Returns ``(avg_ratio, per_layer_overlap)`` where *per_layer_overlap[i]*
    is layer *i*'s GPU-span overlap with layer *i-1* (``0.0`` for layer 0).
    """
    spans = [_get_layer_gpu_span(unit) for unit in units]
    pair_ratios = []
    for i in range(len(spans) - 1):
        if spans[i] is None or spans[i + 1] is None:
            pair_ratios.append(0.0)
            continue
        s1, e1 = spans[i]
        s2, e2 = spans[i + 1]
        overlap = max(0.0, min(e1, e2) - max(s1, s2))
        max_span = max(e1 - s1, e2 - s2)
        pair_ratios.append(overlap / max_span if max_span > 0 else 0.0)
    per_layer = [0.0]
    for i in range(1, len(spans)):
        if spans[i - 1] is None or spans[i] is None:
            per_layer.append(0.0)
            continue
        s_prev, e_prev = spans[i - 1]
        s_cur, e_cur = spans[i]
        overlap = max(0.0, min(e_prev, e_cur) - max(s_prev, s_cur))
        cur_span = e_cur - s_cur
        per_layer.append(overlap / cur_span if cur_span > 0 else 0.0)
    avg = sum(pair_ratios) / len(pair_ratios) if pair_ratios else 0.0
    return avg, per_layer


def _collect_memory(unit: 'FSDPUnit', metrics: 'Metrics'):
    """Track running peak memory from ``memory_delta`` events.

    Walks all phase nodes in chronological order, applies each ``memory_delta``
    to a running counter, and records the maximum.  Only meaningful when
    ``profile_memory=True`` in ``torch.profiler`` — guarded by ``memory_has_data``.
    """
    all_nodes = sorted(
        unit.all_gather_fwd + unit.fwd_compute + unit.all_gather_bwd
        + unit.bwd_compute + unit.reduce_scatter,
        key=lambda n: n.start_time,
    )
    total_alloc = 0
    total_free = 0
    running = 0
    peak = 0
    for n in all_nodes:
        delta = n.memory_delta
        if delta > 0:
            total_alloc += delta
            running += delta
        elif delta < 0:
            total_free += -delta
            running += delta
        if running > peak:
            peak = running
    if total_alloc > 0 or total_free > 0:
        metrics.memory_has_data = True
        metrics.memory_allocated = total_alloc
        metrics.memory_freed = total_free
        metrics.memory_peak = peak


# ---------------------------------------------------------------------------
# Model configuration — FLOP accounting for MFU/HFU/tokens-per-second
# ---------------------------------------------------------------------------

class ModelConfig:
    """Model hyperparameters for FLOP-count estimation and throughput metrics.

    All dimensions can be left at zero / None — in that case MFU, HFU, and
    tokens/sec are omitted from the report (``compute_throughput_metrics``
    returns zeros).

    ``num_gpus`` affects per-GPU throughput scaling.  ``activation_checkpointing``
    is a fraction (0..1) of layers using checkpointing, or ``None`` if unknown.
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        num_layers: int = 0,
        num_heads: int = 0,
        seq_len: int = 0,
        vocab_size: int = 0,
        batch_size: int = 0,
        num_gpus: int = 1,
        activation_checkpointing: Optional[float] = None,
        precision_bits: int = 16,
        gpu_peak_tflops: float = 989.0,   # H100-SXM bf16
        gpu_hbm_bw_gbps: float = 3350.0,   # H100-SXM HBM bandwidth GB/s
        intermediate_dim: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.activation_checkpointing = activation_checkpointing
        self.precision_bits = precision_bits
        self.gpu_peak_tflops = gpu_peak_tflops
        self.gpu_hbm_bw_gbps = gpu_hbm_bw_gbps
        self.intermediate_dim = intermediate_dim or 4 * hidden_dim
        self.num_kv_heads = num_kv_heads or num_heads

    @property
    def is_configured(self) -> bool:
        return self.hidden_dim > 0 and self.num_layers > 0 and self.batch_size > 0

    def estimate_flops_per_step(self) -> float:
        """Estimated FLOPs for one training step (forward + backward).

        Based on the standard transformer FLOP formula from Kaplan et al.:
        forward ≈ 2 * num_layers * (2 * hidden_dim * seq_len * batch_size * (
            4 * hidden_dim  [attention QKV proj + output]
            + 3 * intermediate_dim * hidden_dim * 2  [MLP gate+up+down]
        ))

        Backward is ~2× forward.  Total = 3 × forward for no checkpointing,
        or (2 + activation_checkpointing) × forward when checkpointing is used.

        Returns 0.0 when not configured.
        """
        if not self.is_configured:
            return 0.0
        h = self.hidden_dim
        s = self.seq_len
        b = self.batch_size
        n = self.num_layers
        ffn = self.intermediate_dim

        # Per-token, per-layer FLOPs
        attn_proj = 4 * h * h           # QKV projection + output projection
        attn_scores = 4 * s * h         # QK^T + PV (attention score computation)
        mlp_flops = 3 * 2 * h * ffn     # gate_proj + up_proj + down_proj (×2 for FWD)
        per_layer_flops = attn_proj + attn_scores + mlp_flops

        # Embedding + LM head (if vocab_size > 0)
        embed_flops = 2 * h * self.vocab_size if self.vocab_size > 0 else 0

        forward_flops = b * s * (n * per_layer_flops + embed_flops)
        backward_mult = 3.0 if self.activation_checkpointing is None else 2.0 + self.activation_checkpointing
        return forward_flops * backward_mult

    def estimate_tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len if self.batch_size and self.seq_len else 0

    @property
    def gpu_peak_flops(self) -> float:
        return self.gpu_peak_tflops * 1e12

    @property
    def gpu_hbm_bw_bytes_per_sec(self) -> float:
        return self.gpu_hbm_bw_gbps * 1e9


def _compute_ag_per_layer(root_nodes, time_range=None):
    """Walk all CPU nodes and compute per-layer FSDP all-gather GPU time.

    Uses ``_pg_desc`` to distinguish FSDP all-gather kernels (``mesh_dp_shard``
    or ``mesh_fsdp``) from TP redistribute all-gathers, then classifies each
    kernel as forward or backward by walking its CPU ancestor chain.

    *time_range* ``(start, end)`` filters GPU kernels to those within the
    given window (inclusive).  This is necessary for multi-step traces where
    ``root_nodes`` contain GPU kernel attributions from all ProfilerSteps
    while the downstream analysis only considers the last step.

    Returns ``{layer_name: (ag_fwd_us, ag_bwd_us)}`` — one entry per layer
    that has FSDP all-gather kernels.  Layers with no FSDP all-gather (e.g.
    the last layer when backward_prefetch is absent) get ``(0, 0)``.
    """
    all_nodes = list(TraceParserHelper.iter_nodes(root_nodes))

    parent_map = {}
    stack = list(root_nodes)
    while stack:
        n = stack.pop()
        for ch in n.children:
            parent_map[id(ch)] = n
            stack.append(ch)

    from collections import defaultdict
    totals = defaultdict(lambda: [0.0, 0.0])  # layer -> [fwd, bwd]
    seen = set()

    for node in all_nodes:
        for gpu in (node.direct_gpu_kernels or []):
            if gpu.get('_pg_desc', '') not in FSDP_PG_DESCS:
                continue
            if time_range is not None:
                ts = gpu.get('ts', 0)
                if ts < time_range[0] or ts > time_range[1]:
                    continue
            name = gpu.get('name', '').lower()
            if 'nccl' not in name or not ('allgather' in name or 'all_gather' in name):
                continue
            key = (gpu.get('ts', 0), gpu.get('dur', 0))
            if key in seen:
                continue
            seen.add(key)
            dur = gpu.get('dur', 0)

            layer = None
            phase = None
            curr = node
            while curr is not None:
                cn = curr.name
                if cn.startswith(FSDP_PREFIXES):
                    if 'pre_forward' in cn:
                        layer = _extract_layer(cn) if '(' in cn else None
                        phase = 'fwd'
                        break
                    if 'backward_prefetch' in cn:
                        idx = cn.find(' for ')
                        layer = cn[idx + 5:] if idx >= 0 else None
                        phase = 'bwd'
                        break
                    if 'pre_backward' in cn:
                        layer = _extract_layer(cn) if '(' in cn else None
                        phase = 'bwd'
                        break
                curr = parent_map.get(id(curr))

            if layer and phase:
                if phase == 'fwd':
                    totals[layer][0] += dur
                else:
                    totals[layer][1] += dur

    return dict(totals)


class Metrics:
    """Per-layer computed metrics — all derived quantities for a single FSDP unit.

    One ``Metrics`` per layer: 34 for the 8B TP trace, 8 for async TP.
    The constructor pre-computes everything from the raw ``FSDPUnit`` and global
    aggregates so ``Bottlenecks.detect()`` has all numbers without re-traversal.

    ``total_gpu`` covers FSDP NCCL collectives + aten compute + optimizer (even split).
    ``tp_total_gpu`` is additive — the full picture for FSDP+TP models is
    ``total_gpu + tp_total_gpu``.
    """

    def __init__(self, unit: FSDPUnit, global_optimizer_gpu: float = 0.0,
                 global_optimizer_cpu: float = 0.0,
                 num_units: int = 1, global_tp_ag_gpu: float = 0.0,
                 global_tp_rs_gpu: float = 0.0,
                 global_tp_ar_gpu: float = 0.0,
                 tp_kernels: Optional[List[dict]] = None,
                 ag_per_layer: Optional[Dict[str, Tuple[float, float]]] = None,
                 rs_global_seen: Optional[Set[Tuple[float, float]]] = None):
        self.layer_name = unit.layer_name
        ln = unit.layer_name

        # --- Phase raw times — these are the building blocks for everything ---
        if ag_per_layer is not None and ln in ag_per_layer:
            self.ag_fwd_gpu = ag_per_layer[ln][0]
            self.ag_bwd_gpu = ag_per_layer[ln][1] + unit.ag_bwd_supplement_us
        else:
            self.ag_fwd_gpu = _phase_gpu_time(unit.all_gather_fwd)
            self.ag_bwd_gpu = (_phase_gpu_time(unit.all_gather_bwd)
                               + _phase_gpu_time(unit.all_gather_bwd_nccl)
                               + unit.ag_bwd_supplement_us
                               + _collect_nccl_kernel_time(unit.bwd_compute))
        self.ag_fwd_cpu = _phase_cpu_time(unit.all_gather_fwd)
        self.ag_fwd_wall = _phase_wall_time(unit.all_gather_fwd)
        self.ag_bwd_cpu = (_phase_cpu_time(unit.all_gather_bwd)
                           + _phase_cpu_time(unit.all_gather_bwd_nccl))
        self.ag_bwd_wall = _phase_wall_time(
            unit.all_gather_bwd + unit.all_gather_bwd_nccl)

        self.fwd_cmp_gpu = _phase_gpu_time_direct(unit.fwd_compute)
        self.fwd_cmp_cpu = _phase_cpu_time(unit.fwd_compute)
        self.fwd_cmp_wall = _phase_wall_time(unit.fwd_compute, unit.fwd_compute_span)

        self.bwd_cmp_gpu = _phase_gpu_time_direct(unit.bwd_compute)
        self.ag_bwd_cpu = (_phase_cpu_time(unit.all_gather_bwd)
                           + _phase_cpu_time(unit.all_gather_bwd_nccl))
        self.ag_bwd_wall = _phase_wall_time(
            unit.all_gather_bwd + unit.all_gather_bwd_nccl
        )

        self.bwd_cmp_gpu = _phase_gpu_time_direct(unit.bwd_compute)
        self.bwd_cmp_cpu = _phase_cpu_time(unit.bwd_compute)
        self.bwd_cmp_wall = _phase_wall_time(unit.bwd_compute, unit.bwd_compute_span)

        rs_gpu, ar_in_rs_gpu, rs_overhead_gpu = _phase_gpu_time_breakdown(
            unit.reduce_scatter, rs_global_seen,
            allowed_pg=FSDP_PG_DESCS)
        self.rs_gpu = rs_gpu
        self.ar_in_rs_gpu = ar_in_rs_gpu
        self.rs_overhead_gpu = rs_overhead_gpu
        self.rs_cpu = _phase_cpu_time(unit.reduce_scatter)
        self.rs_wall = _phase_wall_time(unit.reduce_scatter)

        # --- Optimizer (evenly split across layers) ---
        self.optimizer_gpu = global_optimizer_gpu / num_units if num_units > 0 else 0.0
        self.optimizer_cpu = global_optimizer_cpu / num_units if num_units > 0 else 0.0

        # --- Totals (all FSDP phases) ---
        self.total_gpu = (self.ag_fwd_gpu + self.fwd_cmp_gpu + self.ag_bwd_gpu
                          + self.bwd_cmp_gpu + self.rs_gpu + self.ar_in_rs_gpu
                          + self.optimizer_gpu)
        self.total_cpu = (self.ag_fwd_cpu + self.fwd_cmp_cpu + self.ag_bwd_cpu
                          + self.bwd_cmp_cpu + self.rs_cpu + self.optimizer_cpu)

        # --- Communication vs compute ratio ---
        comm_gpu = self.ag_fwd_gpu + self.ag_bwd_gpu + self.rs_gpu + self.ar_in_rs_gpu  # FSDP NCCL
        comp_gpu = self.fwd_cmp_gpu + self.bwd_cmp_gpu  # attn/MLP/norm
        self.comp_ratio = comp_gpu / self.total_gpu if self.total_gpu > 0 else 0.0

        self.optimizer_ratio = self.optimizer_gpu / self.total_gpu if self.total_gpu > 0 else 0.0

        # --- Per-phase exposed ratio ---
        # How much of each collective's wall span is occupied by its GPU work.
        #   GPU time / CPU wall span → 1.0 = fully exposed (no overlap),
        #   << 1.0 = well hidden (GPU execution overlaps with other work).
        # Values > 1.0 are capped at 1.0 because async NCCL and ac2g copy-out
        # streams routinely produce GPU_time > CPU_wall (NCCL kernel dispatched
        # in 0.4ms CPU, runs 24ms on GPU).  Capping keeps avg_exposed_ratio
        # in [0, 1].
        self.ag_fwd_exposed_ratio = min(1.0, (
            self.ag_fwd_gpu / self.ag_fwd_wall if self.ag_fwd_wall > 0 else 0.0
        ))
        self.rs_exposed_ratio = min(1.0, (
            (self.rs_gpu + self.ar_in_rs_gpu) / self.rs_wall if self.rs_wall > 0 else 0.0
        ))
        self.ag_bwd_exposed_ratio = min(1.0, (
            self.ag_bwd_gpu / self.ag_bwd_wall if self.ag_bwd_wall > 0 else 0.0
        ))

        # --- Per-phase wall spans (sum of forward + backward, not union,
        # so the pipeline gap between forward and backward is excluded) ---
        fwd_events = unit.all_gather_fwd + unit.fwd_compute
        bwd_events = unit.all_gather_bwd + unit.bwd_compute + unit.reduce_scatter
        fwd_span = _phase_wall_time(fwd_events) if fwd_events else 0.0
        bwd_span = _phase_wall_time(bwd_events) if bwd_events else 0.0
        self.layer_span = fwd_span + bwd_span

        # --- Per-phase GPU active time + overall busy ---
        fwd_intervals = _collect_kernel_intervals([
            unit.all_gather_fwd_gpu_kernels,
            _collect_gpu_kernels(unit.fwd_compute),
        ])
        bwd_intervals = _collect_kernel_intervals([
            unit.all_gather_bwd_gpu_kernels,
            unit.reduce_scatter_gpu_kernels,
            _collect_gpu_kernels(unit.bwd_compute),
        ])
        fwd_active = _merge_intervals(fwd_intervals)
        bwd_active = _merge_intervals(bwd_intervals)
        gpu_active_us = fwd_active + bwd_active
        # Use max(cpu_span, gpu_kernel_span) as the denominator so that
        # async-issued kernels whose timestamps fall outside the CPU event
        # window don't push ratio > 1.0.
        fwd_gpu_span = (max(e for _, e in fwd_intervals) - min(s for s, _ in fwd_intervals)) if fwd_intervals else 0.0
        bwd_gpu_span = (max(e for _, e in bwd_intervals) - min(s for s, _ in bwd_intervals)) if bwd_intervals else 0.0
        self.fwd_busy = min(1.0, fwd_active / max(fwd_span, fwd_gpu_span)) if max(fwd_span, fwd_gpu_span) > 0 else 0.0
        self.bwd_busy = min(1.0, bwd_active / max(bwd_span, bwd_gpu_span)) if max(bwd_span, bwd_gpu_span) > 0 else 0.0
        self.gpu_busy = min(1.0, gpu_active_us / max(self.layer_span, fwd_gpu_span + bwd_gpu_span)) if max(self.layer_span, fwd_gpu_span + bwd_gpu_span) > 0 else 0.0

        # --- TP metrics (evenly split across layers) ---
        self.tp_ag_gpu = global_tp_ag_gpu / num_units if num_units > 0 else 0.0
        self.tp_rs_gpu = global_tp_rs_gpu / num_units if num_units > 0 else 0.0
        self.tp_ar_gpu = global_tp_ar_gpu / num_units if num_units > 0 else 0.0
        self.tp_total_gpu = self.tp_ag_gpu + self.tp_rs_gpu + self.tp_ar_gpu
        # TP contention inflation (backfilled by Report._compute_aggregated)
        self.tp_contention_inflation = 1.0
        self.tp_effective_gpu_us = self.tp_total_gpu

        # --- Kernel counts per phase (small-kernel detection) ---
        self.ag_fwd_count = len(unit.all_gather_fwd_gpu_kernels)
        self.ag_bwd_count = len(unit.all_gather_bwd_gpu_kernels)
        self.rs_count = len(unit.reduce_scatter_gpu_kernels)

        # --- Per-layer async TP overlap ---
        self.fwd_comp_comm_overlap = 0.0
        self.bwd_comp_comm_overlap = 0.0
        self.pipeline_overlap_ratio = 0.0
        if tp_kernels and unit.fwd_compute and unit.bwd_compute:
            # Use the union of CPU event window and GPU kernel window to
            # determine which TP kernels overlap with compute phase execution.
            # Pure CPU timestamps are unreliable with torch.compile (very short
            # CPU dispatch windows don't capture async GPU kernel execution).
            fwd_kernels = _collect_gpu_kernels(unit.fwd_compute)
            bwd_kernels = _collect_gpu_kernels(unit.bwd_compute)
            fwd_cpu_start = unit.fwd_compute[0].start_time
            fwd_cpu_end = unit.fwd_compute[-1].end_time
            bwd_cpu_start = unit.bwd_compute[0].start_time
            bwd_cpu_end = unit.bwd_compute[-1].end_time
            if fwd_kernels:
                fwd_start = min(fwd_cpu_start,
                                min(k.get('ts', 0) for k in fwd_kernels))
                fwd_end = max(fwd_cpu_end,
                              max(k.get('ts', 0) + k.get('dur', 0) for k in fwd_kernels))
            else:
                fwd_start, fwd_end = fwd_cpu_start, fwd_cpu_end
            if bwd_kernels:
                bwd_start = min(bwd_cpu_start,
                                min(k.get('ts', 0) for k in bwd_kernels))
                bwd_end = max(bwd_cpu_end,
                              max(k.get('ts', 0) + k.get('dur', 0) for k in bwd_kernels))
            else:
                bwd_start, bwd_end = bwd_cpu_start, bwd_cpu_end
            fwd_wall = fwd_end - fwd_start
            bwd_wall = bwd_end - bwd_start
            fwd_tp = sum(k.get('dur', 0) for k in tp_kernels
                         if k.get('ts', 0) >= fwd_start
                         and k.get('ts', 0) + k.get('dur', 0) <= fwd_end)
            bwd_tp = sum(k.get('dur', 0) for k in tp_kernels
                         if k.get('ts', 0) >= bwd_start
                         and k.get('ts', 0) + k.get('dur', 0) <= bwd_end)
            self.fwd_comp_comm_overlap = fwd_tp / fwd_wall if fwd_wall > 0 else 0.0
            self.bwd_comp_comm_overlap = bwd_tp / bwd_wall if bwd_wall > 0 else 0.0

        # --- Memory ---
        self.memory_peak = 0
        self.memory_allocated = 0
        self.memory_freed = 0
        self.memory_has_data = False
        _collect_memory(unit, self)

        # --- Overlap (backfilled by Report._compute_aggregated) ---
        self.overlap_ratio = 0.0
        self.serial_ratio = 1.0
        self.idle_ratio = 0.0
        self.step_wall = 0.0

        # --- Communication ratio including TP ---
        total_gpu = self.total_gpu + self.tp_total_gpu
        fsdp_comm = self.ag_fwd_gpu + self.ag_bwd_gpu + self.rs_gpu + self.ar_in_rs_gpu
        self.fsdp_comm_ratio = fsdp_comm / total_gpu if total_gpu > 0 else 0.0
        self.tp_comm_ratio = self.tp_total_gpu / total_gpu if total_gpu > 0 else 0.0
        self.comm_ratio = (fsdp_comm + self.tp_total_gpu) / total_gpu if total_gpu > 0 else 0.0

        # --- Compute-to-communicate ratio ---
        # Arithmetic intensity proxy: compute GPU time ÷ all comm GPU time.
        # Higher = more compute-heavy; <1.0 = comm-dominated.
        all_comm = fsdp_comm + self.tp_total_gpu
        comp = self.fwd_cmp_gpu + self.bwd_cmp_gpu
        self.compute_to_comm_ratio = comp / all_comm if all_comm > 0 else float('inf')

        # --- Average exposed ratio ---
        # Average of the three per-phase exposed ratios (ag_fwd, rs, ag_bwd).
        # Each exposed ratio = gpu_time / wall_time for that collective:
        #   high (near 1.0) = collective's GPU fills its entire wall span = exposed
        #   low  (near 0.0) = collective squeezed into a small GPU window = hidden
        self.avg_exposed_ratio = 0.0
        ratios = [self.ag_fwd_exposed_ratio, self.rs_exposed_ratio,
                  self.ag_bwd_exposed_ratio]
        valid = [r for r in ratios if r > 0]
        if valid:
            self.avg_exposed_ratio = sum(valid) / len(valid)

        # --- Bottleneck sub-metrics ---
        self._compute_kernel_metrics(unit)
        self._compute_copy_vs_nccl_metrics(unit)
        self._compute_wall_metrics(unit)
        self._compute_gpu_kernel_classification(unit)

    def compute_throughput_metrics(self, model_config: ModelConfig):
        """Populate MFU, HFU, tokens/sec/GPU from model config.

        Call this after ``__init__`` and after ``overlap_ratio / step_wall``
        have been backfilled by ``Report._compute_aggregated``.  Metrics are
        stored as attributes on ``self`` for downstream use.
        """
        self.tokens_per_second_per_gpu = 0.0
        self.mfu = 0.0
        self.hfu = 0.0
        self.estimated_flops_per_step = 0.0

        if not model_config.is_configured:
            return

        step_wall_us = self.step_wall if self.step_wall > 0 else self.layer_span
        if step_wall_us <= 0:
            return

        step_wall_s = step_wall_us / 1_000_000
        tokens_per_step = model_config.estimate_tokens_per_step()
        num_gpus = model_config.num_gpus

        self.tokens_per_second_per_gpu = tokens_per_step / step_wall_s / num_gpus

        flops = model_config.estimate_flops_per_step()
        self.estimated_flops_per_step = flops
        observed_flops = flops / step_wall_s
        self.mfu = observed_flops / model_config.gpu_peak_flops / num_gpus

        if model_config.activation_checkpointing is not None:
            # HFU accounts for recompute: with activation checkpointing,
            # backward re-computes the forward activations.  The multiplier is
            # (2 + c) / 3 where c is the checkpointing fraction.
            checkpoint_mult = (2.0 + model_config.activation_checkpointing) / 3.0
            self.hfu = self.mfu / checkpoint_mult if checkpoint_mult > 0 else 0.0
        else:
            self.hfu = self.mfu

    # ------------------------------------------------------------------
    # Sub-metric computation (called from ``__init__``)
    # ------------------------------------------------------------------

    def _compute_kernel_metrics(self, unit):
        """Count GPU kernels across all phases for this layer.
        The 8B trace's ``all_gather_copy_out`` produces 300+ split_with_sizes_copy
        kernels averaging <4µs — this is the main small-kernel hotspot.  The
        detector fires when count ≥ 100 and average <5µs.
        """
        all_kernels = _collect_gpu_kernels(
            unit.all_gather_fwd + unit.fwd_compute
        ) + _collect_gpu_kernels(
            unit.all_gather_bwd + unit.bwd_compute + unit.reduce_scatter
        )
        self.kernel_count = len(all_kernels)
        total_dur = sum(k.get('dur', 0) for k in all_kernels)
        self.avg_kernel_dur_us = total_dur / self.kernel_count if self.kernel_count > 0 else 0.0

    def _compute_copy_vs_nccl_metrics(self, unit):
        """Decompose all-gather GPU time into NCCL collective vs data-movement.
        Each GPU kernel under all-gather nodes is classified by name:
          - ``ncclKernel_*``, ``ncclDevice*`` → NCCL (wire time)
          - ``split_with_sizes_copy``, ``aten::copy_`` → copy (host-device memcpy)
        When copy exceeds 50% of the sum, the ``copy-heavy`` bottleneck fires.
        This is common for small FSDP buffer sizes where memcpy overhead is
        comparable to NCCL latency.
        """
        nccl_dur = 0.0
        copy_dur = 0.0
        for ag_node in unit.all_gather_fwd + unit.all_gather_bwd:
            for kernel in _collect_gpu_kernels([ag_node]):
                cat = _classify_kernel(kernel.get('name', ''))
                dur = kernel.get('dur', 0)
                if cat == 'nccl':
                    nccl_dur += dur
                elif cat == 'copy':
                    copy_dur += dur
        self.nccl_comm_gpu_us = nccl_dur
        self.copy_data_movement_gpu_us = copy_dur

    def _compute_wall_metrics(self, unit):
        """CPU wall span across all phases for this layer, and its ratio to GPU time.
        A ratio >1.5× flags host-boundness — either pipeline serialisation (CPU
        submits this layer, moves on, returns later) or CUDA dispatch overhead.
        Early/late layers in the 8B trace naturally have ratios of 3-10× due to
        pipeline stagger — not necessarily pathological.
        """
        all_events = (unit.all_gather_fwd + unit.fwd_compute + unit.all_gather_bwd
                      + unit.bwd_compute + unit.reduce_scatter)
        if all_events:
            cpu_start = min(n.start_time for n in all_events)
            cpu_end = max(n.end_time for n in all_events)
            self.cpu_wall_us = cpu_end - cpu_start
            self.cpu_wall_to_gpu_ratio = self.cpu_wall_us / self.total_gpu if self.total_gpu > 0 else 0.0
        else:
            self.cpu_wall_us = 0.0
            self.cpu_wall_to_gpu_ratio = 0.0

    def _compute_gpu_kernel_classification(self, unit):
        """Classify all GPU kernels across compute phases (fwd+bwd) into
        compute vs NCCL vs copy, and track their count and mean duration.

        These are used for HBM-bandwidth-bound detection: if compute kernels
        have low average duration (< 5µs) they are likely launch-latency-bound
        rather than arithmetic-bound, indicating HBM BW may be the limiter.
        """
        comp_kernels = _collect_gpu_kernels(unit.fwd_compute + unit.bwd_compute)
        self.comp_kernel_count = len(comp_kernels)
        self.comp_kernel_avg_dur_us = (
            sum(k.get('dur', 0) for k in comp_kernels) / self.comp_kernel_count
            if self.comp_kernel_count > 0 else 0.0
        )
        # Count NCCL and copy kernels in compute phases
        self.nccl_in_comp_count = sum(
            1 for k in comp_kernels if 'nccl' in k.get('name', '').lower()
        )
        self.copy_in_comp_count = sum(
            1 for k in comp_kernels
            if any(p in k.get('name', '').lower() for p in ('copy', 'memcpy', 'memset'))
        )
        # ``estimated_bytes_moved`` removed — the previous heuristic was
        # meaningless (kernel_count × avg_dur × 2000).  Accurate estimation
        # requires knowledge of kernel input shapes (not available in traces).

    def to_dict(self) -> Dict[str, float]:
        return {
            "ag_fwd_gpu_us": self.ag_fwd_gpu,
            "fwd_cmp_gpu_us": self.fwd_cmp_gpu,
            "ag_bwd_gpu_us": self.ag_bwd_gpu,
            "bwd_cmp_gpu_us": self.bwd_cmp_gpu,
            "rs_gpu_us": self.rs_gpu,
            "ar_in_rs_gpu_us": self.ar_in_rs_gpu,
            "rs_overhead_gpu_us": self.rs_overhead_gpu,
            "optimizer_gpu_us": self.optimizer_gpu,
            "tp_ag_gpu_us": self.tp_ag_gpu,
            "tp_rs_gpu_us": self.tp_rs_gpu,
            "tp_ar_gpu_us": self.tp_ar_gpu,
            "tp_total_gpu_us": self.tp_total_gpu,
            "tp_contention_inflation": self.tp_contention_inflation,
            "tp_effective_gpu_us": self.tp_effective_gpu_us,
            "total_gpu_us": self.total_gpu,
            "total_cpu_us": self.total_cpu,
            "comm_ratio": self.comm_ratio,
            "comp_ratio": self.comp_ratio,
            "optimizer_ratio": self.optimizer_ratio,
            "gpu_busy": self.gpu_busy,
            "fwd_busy": self.fwd_busy,
            "bwd_busy": self.bwd_busy,
            "layer_span_us": self.layer_span,
            "overlap_ratio": self.overlap_ratio,
            "serial_ratio": self.serial_ratio,
            "idle_ratio": self.idle_ratio,
            "memory_peak": self.memory_peak,
            "memory_allocated": self.memory_allocated,
            "memory_freed": self.memory_freed,
            "fsdp_comm_ratio": self.fsdp_comm_ratio,
            "tp_comm_ratio": self.tp_comm_ratio,
            "fwd_comp_comm_overlap": self.fwd_comp_comm_overlap,
            "bwd_comp_comm_overlap": self.bwd_comp_comm_overlap,
            "pipeline_overlap_ratio": self.pipeline_overlap_ratio,
            "kernel_count": self.kernel_count,
            "avg_kernel_dur_us": self.avg_kernel_dur_us,
            "nccl_comm_gpu_us": self.nccl_comm_gpu_us,
            "copy_data_movement_gpu_us": self.copy_data_movement_gpu_us,
            "cpu_wall_us": self.cpu_wall_us,
            "cpu_wall_to_gpu_ratio": self.cpu_wall_to_gpu_ratio,
            # New universal metrics
            "compute_to_comm_ratio": self.compute_to_comm_ratio,
            "ag_fwd_exposed_ratio": self.ag_fwd_exposed_ratio,
            "rs_exposed_ratio": self.rs_exposed_ratio,
            "ag_bwd_exposed_ratio": self.ag_bwd_exposed_ratio,
            "avg_exposed_ratio": self.avg_exposed_ratio,
            "comp_kernel_count": self.comp_kernel_count,
            "comp_kernel_avg_dur_us": self.comp_kernel_avg_dur_us,
            "nccl_in_comp_count": self.nccl_in_comp_count,
            "copy_in_comp_count": self.copy_in_comp_count,
            # Throughput (populated by compute_throughput_metrics)
            "tokens_per_second_per_gpu": getattr(self, 'tokens_per_second_per_gpu', 0.0),
            "mfu": getattr(self, 'mfu', 0.0),
            "hfu": getattr(self, 'hfu', 0.0),
            "estimated_flops_per_step": getattr(self, 'estimated_flops_per_step', 0.0),
        }


def _format_us(v: float) -> str:
    """Human-readable time: µs / ms / s.  Thresholds at 1000 and 1,000,000.
    NB: May need minutes/hours for very long training steps.
    """
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}s"
    if v >= 1_000:
        return f"{v / 1_000:.2f}ms"
    return f"{v:.1f}us"


class Bottlenecks:
    """Threshold-based bottleneck classification.

    Each class attribute is a heuristic with a physical justification derived
    from the roofline model, NCCL bandwidth model, Amdahl's law, and GPU
    architecture specs (H100-SXM unless noted).  Should be revisited for
    different architectures or hardware.
    """

    # ====================================================================
    # Section A — Classical bottleneck thresholds
    # ====================================================================
    # Physical basis (Amdahl's law / Pareto principle):
    # When ≥ 70 % of GPU time is compute (fwd + bwd) and ≤ 30 % is
    # communication + overhead, further comm optimisation yields at most
    # 1 / (1 − 0.70) ≈ 1.43× speedup.  The bottleneck has shifted to compute.
    COMP_HEAVY_THRESHOLD = 0.70

    # Physical basis (NCCL bandwidth model):
    # On H100 NVLink (≈ 900 GB/s aggregate), large-message all-reduce
    # bandwidth plateaus at ≈ 60 – 70 GB/s.  When communication occupies
    # ≥ 40 % of GPU time, the all-reduce operates in the bandwidth-limited
    # regime: doubling network BW gives < 2× speedup (Amdahl).
    COMM_HEAVY_THRESHOLD = 0.40

    # Physical basis (Amdahl's law):
    # With 15 % serial / idle fraction, maximum achievable speedup is
    # 1 / 0.15 ≈ 6.7× regardless of GPU count.  Beyond this threshold,
    # I/O or pipeline bubbles materially limit strong scaling efficiency.
    IO_HEAVY_THRESHOLD = 0.15

    # Physical basis (NCCL all-gather BW model):
    # Same BW ceiling as all-reduce on NVLink (≈ 60 – 70 GB/s large-message).
    # When FSDP all-gather ≥ 40 % of FSDP-comm time, unshard dominates the
    # communication profile.
    AG_HEAVY_THRESHOLD = 0.40

    # Physical basis (NCCL reduce-scatter BW model):
    # Reduce-scatter has identical bandwidth characteristics to all-gather
    # (algorithmic equivalents in NCCL ring / tree).  ≥ 40 % indicates
    # gradient synchronisation dominates FSDP comm.
    RS_HEAVY_THRESHOLD = 0.40

    # Physical basis (NVLink BW model):
    # TP collectives run intra-node over NVLink at ≈ 900 GB/s (H100).
    # Lower threshold because TP (unlike FSDP) is synchronous on the
    # critical path and cannot be overlapped across layers.
    # Even 15 % GPU time in TP materially limits scaling efficiency.
    TP_HEAVY_THRESHOLD = 0.15

    # Physical basis (HBM memory-bandwidth model):
    # ADAMW is memory-bandwidth-bound (reads g + m, writes m + params).
    # On H100 HBM3 (3.35 TB/s), 15 % threshold corresponds to the point
    # where optimizer memory traffic becomes a dominant fraction of the
    # HBM budget, competing with compute kernels for BW.
    OPTIMIZER_HEAVY_THRESHOLD = 0.15

    # Physical basis (utilisation heuristic):
    # Below 50 % GPU utilisation means the GPU is idle more than it is
    # computing — the pipeline delivers < half its achievable throughput.
    # Standard industry heuristic (cf. NVIDIA GPU utilisation guidance).
    UTIL_LOW_THRESHOLD = 0.50

    # ====================================================================
    # Section B — FSDP2 / TP / async TP specific thresholds
    # ====================================================================

    # Physical basis (CUDA launch-latency model):
    # GPU kernel launch latency on CUDA is ≈ 3 – 8 µs (driver + scheduler
    # overhead).  When avg kernel duration ≤ 5 µs, execution is comparable
    # to launch latency → launch-latency-bound (cf. NVIDIA guidance:
    # kernels should be > 10 µs for efficient utilisation).
    SMALL_KERNEL_AVG_DUR_US = 5.0

    # Physical basis (empirical — transformer layer composition):
    # A typical transformer layer has ≈ 50 – 80 CUDA kernels (attention +
    # MLP + norms).  When count ≥ 100 with avg < 5 µs, the decomposition
    # is so fine-grained that launch overhead dominates execution.
    SMALL_KERNEL_COUNT = 100

    # Physical basis (async TP pipeline model):
    # In an ideal async TP implementation, TP collectives overlap with
    # compute for > 70 % of forward wall time.  Below 30 % the overlap
    # mechanism is present but underutilised — the async pipeline hides
    # less than a third of TP latency.
    ASYNC_TP_OVERLAP_LOW = 0.30

    # Physical basis (pipeline symmetry):
    # |fwd − bwd| overlap ≥ 40 pp indicates the pipeline stagger is tuned
    # for one direction at the expense of the other.  Forward has one AG,
    # backward has AG + RS → inherently asymmetric; large asymmetry means
    # the stagger is poorly balanced.
    OVERLAP_ASYMMETRY_DIFF = 0.40

    # Physical basis (CPU dispatch overhead model):
    # For efficient GPU utilisation, CPU dispatch overhead should be a small
    # fraction of GPU time.  A ratio of 3.0 means every 1 µs of GPU work
    # costs 3 µs of CPU serialisation — a strong indicator of Python
    # runtime / GIL / PyTorch dispatch overhead.  Empirically, well-optimised
    # FSDP2 training loops achieve CPU:GPU ≤ 1.5×.
    HOST_BOUND_RATIO = 3.0

    # Physical basis (CPU memory bandwidth — DDR5):
    # In FSDP2 AG, non-NCCL time is dominated by split_with_sizes_copy
    # (CPU-side memory copies for tensor slicing).  When ≥ 50 % of AG time
    # is copy, CPU data-preparation (DDR5 ≈ 50 – 80 GB/s) bottlenecks the
    # AG faster than NCCL (NVLink ≈ 450 GB/s unidirectional).
    COPY_HEAVY_RATIO = 0.50

    # Physical basis (theoretical fwd : bwd FLOPs ratio):
    # Standard transformer layer with activation checkpointing: theoretical
    # fwd : bwd FLOPs ≈ 1 : 2.  Deviation > 25 % from this ratio
    # (i.e. |fwd − bwd| / max ≥ 0.75) signals an imbalance from act-ckpt
    # or custom gradient computation.
    FWD_BWD_IMBALANCE = 0.75

    # Physical basis (FSDP2 pipeline model):
    # With N pipeline stages, the theoretical max overlap fraction is
    # (N − 1) / N.  Even a modest 2-deep pipeline should achieve 50 %
    # overlap.  Serial > 85 % means < 15 % overlap — the stagger is
    # effectively defeated by synchronisation or insufficient in-flight
    # layers.
    SERIAL_RATIO_HIGH = 0.85

    # ====================================================================
    # Section C — Communication-hiding / BW bottleneck thresholds
    # ====================================================================

    # Physical basis (NCCL AG BW model — inter-node):
    # AG fwd GPU ≥ 80 % of fwd compute GPU → the all-gather is mostly
    # exposed (GPU stalls waiting for AG).  On NVLink (≈ 450 GB/s
    # unidirectional), at this ratio doubling AG BW would reduce step
    # time by ≤ 20 % (Amdahl's law: serial fraction = 0.80 / 1.80 ≈ 0.44,
    # max speedup = 1 / (1 − 0.44 + 0.44 / 2) ≈ 1.22×).
    AG_LATENCY_EXPOSED_RATIO = 0.80

    # Physical basis (roofline model — H100 HBM3):
    # For H100 (HBM3: 3.35 TB/s, compute: 989 TFLOPS bf16), the ridge
    # point (compute / memory ceiling intersection) is at arithmetic
    # intensity = 989 TFLOPS / 3.35 TB/s ≈ 295 FLOPs/byte (bf16).
    # A kernel of 8 µs at typical grid size (≈ 256 blocks = 2 waves on 128
    # SMs) suggests an instruction mix well below the ridge point, placing
    # it in the memory-bandwidth-bound regime.
    HBM_BOUND_AVG_KERNEL_US = 8.0

    # Physical basis (utilisation cross-validation):
    # GPU utilisation < 50 % AND avg kernel < 8 µs → kernel execution is
    # memory-bandwidth-starved rather than compute-bound.  Short kernels
    # issue rapidly but spend most of their time waiting on HBM.
    HBM_BOUND_GPU_UTIL = 0.50

    # Physical basis (injection BW model):
    # RS GPU ≥ 100 % of bwd compute GPU means gradient synchronisation
    # takes as long as backward computation.  On H100 NVLink (≈ 60 – 70
    # GB/s large-message all-reduce BW), this is the point where the
    # all-reduce BW limits backward-pass throughput as much as the compute
    # itself.  Uses GPU-time ratio (not overlap efficiency) because async
    # NCCL's CPU_wall is just dispatch time, making overlap meaningless.
    RS_INJECTION_RATIO = 1.0

    # Physical basis (critical-path analysis):
    # Even 10 % TP GPU time relative to compute is significant because TP
    # collectives are synchronous per transformer layer (unlike FSDP which
    # pipelines across layers).  Synchronous collectives > 10 % of compute
    # directly limit scaling efficiency via Amdahl's law.
    TP_ON_CRITICAL_PATH_RATIO = 0.10

    # Physical basis (async TP profitability model):
    # When TP overlap < 20 % AND TP ≥ 10 % of compute, the async TP
    # mechanism hides < 20 % of TP latency — effectively synchronous.
    # An overlap of 20 % means only 1/5 of the collective latency is hidden.
    TP_ON_CRITICAL_PATH_OVERLAP = 0.20

    # Physical basis (NVLink message-efficiency model):
    # NVLink bandwidth efficiency drops significantly for small messages
    # (< 1 KB).  On H100 (18 NVLink 4.0 links × 450 GB/s uni), small-
    # message throughput can be as low as 5 – 10 % of peak.  ≥ 50 small
    # TP kernels per layer indicates message fragmentation that prevents
    # NVLink BW saturation.
    NVLINK_SAT_COUNT = 50

    # Physical basis (NVLink BW model):
    # For a 128 KB TP message on H100 NVLink (≈ 70 GB/s achievable large-
    # message BW per link pair), expected transfer ≲ 2 µs.  Average TP
    # kernel < 10 µs indicates kernels are so short they never saturate
    # NVLink bandwidth — the launch overhead dominates.
    NVLINK_SAT_AVG_US = 10.0

    # Physical basis (pipeline utilisation model):
    # GPU utilisation < 50 % is the general low-utilisation threshold (see
    # UTIL_LOW_THRESHOLD).  When combined with AG ≥ 10 % of fwd compute,
    # the AG is a material contributor to the idle — it is not hidden by
    # the pipeline stagger.
    COMM_COMPUTE_UTIL_THRESHOLD = 0.50

    # Physical basis (signal-to-noise filter):
    # AG < 10 % of fwd compute is negligible — low utilisation in that
    # case is driven by other factors (data loading, sync).  The 10 % floor
    # ensures we only flag communication-mediated low utilisation.
    AG_VS_FWD_RATIO = 0.10

    # Physical basis (utilisation-derived idle model):
    # GPU idle ≥ 50 % of wall time AND the layer has both fwd and bwd
    # (ruling out pipeline warm-down / warm-up) → communication exposure
    # is the dominant limiter.  Uses 1 − gpu_busy, which replaces the
    # old overlap-efficiency formula that broke for async NCCL where
    # CPU_wall is just dispatch, not execution.
    EXPOSED_COMM_IDLE_THRESHOLD = 0.50

    # Physical basis (cross-layer GPU pipeline model):
    # Consecutive layers' GPU spans overlapping < 20 % means the FSDP2
    # pipeline stagger delivers negligible GPU-level parallelism.
    # Theoretical ideal overlap = (N − 1) / N; even a 2-layer pipeline
    # should achieve 50 %.  Below 20 % indicates either insufficient
    # micro-batches or synchronisation that serialises the GPU stream.
    # Cross-check against CPU overlap_ratio (sweep-line) which measures
    # CPU dispatch pipelining — the GPU overlap is the real signal.
    PIPELINE_OVERLAP_LOW = 0.20

    # Physical basis (CUPTI contention model):
    # When TP collectives overlap with compute kernels on the GPU, CUPTI
    # records inflated wall durations because the TP kernel contends with
    # compute for SM resources. The uncontested baseline is the 25th
    # percentile of TP kernel durations (rejects NCCL setup/control kernels).
    # An inflation ratio ≥ 3.0 means a significant subset of TP kernels are
    # much longer than the baseline — a sign of compute-contention artifacts.
    TP_CONTENTION_INFLATION_HIGH = 3.0

    @classmethod
    def detect(cls, metrics: Metrics) -> List[str]:
        """All bottleneck checks → list of descriptive labels (empty = no bottleneck).
        Additive: a single layer can have multiple issues.  Section A = classic
        (compute/comm/IO/phase-dominates), Section B = FSDP2/TP-specific
        (small-kernel, pipeline, async TP, host-bound, copy-heavy, fwd-bwd asymmetry).
        """
        issues = []
        total_gpu = metrics.total_gpu + metrics.tp_total_gpu
        if total_gpu == 0:
            return issues

        # ================================================================
        # Section A — Classic bottlenecks
        # ================================================================

        # --- Pipeline bubble (idle between layers) ---
        # Sweep-line found wall time where NO FSDP shard group was active —
        # a pipeline bubble, data-loading stall, or synchronisation point.
        if metrics.idle_ratio >= cls.IO_HEAVY_THRESHOLD:
            issues.append(f"I/O or pipeline bubble ({metrics.idle_ratio:.1%} idle)")

        # --- Communication-bound (FSDP NCCL + TP) ---
        if metrics.comm_ratio >= cls.COMM_HEAVY_THRESHOLD:
            issues.append(f"comm-bound (comm={metrics.comm_ratio:.1%})")

        # --- All-gather / reduce-scatter heavy within FSDP comm ---
        fsdp_comm = metrics.ag_fwd_gpu + metrics.ag_bwd_gpu + metrics.rs_gpu
        if fsdp_comm > 0:
            ag_ratio = (metrics.ag_fwd_gpu + metrics.ag_bwd_gpu) / fsdp_comm
            rs_ratio = metrics.rs_gpu / fsdp_comm
            if ag_ratio >= cls.AG_HEAVY_THRESHOLD:
                issues.append(f"all-gather-heavy ({ag_ratio:.1%} of FSDP comm)")
            if rs_ratio >= cls.RS_HEAVY_THRESHOLD:
                issues.append(f"reduce-scatter-heavy ({rs_ratio:.1%} of FSDP comm)")

        # --- TP-heavy (mesh_tp collectives dominate GPU) ---
        tp_total = metrics.tp_ag_gpu + metrics.tp_rs_gpu + metrics.tp_ar_gpu
        if tp_total > 0 and tp_total / total_gpu >= cls.TP_HEAVY_THRESHOLD:
            issues.append(f"TP-heavy ({tp_total/total_gpu:.1%} of total GPU)")

        # --- Optimizer-heavy (ADAMW step dominates GPU) ---
        if metrics.optimizer_gpu > 0 and metrics.optimizer_ratio >= cls.OPTIMIZER_HEAVY_THRESHOLD:
            issues.append(f"optimizer-heavy ({metrics.optimizer_ratio:.1%} of total GPU)")

        # --- Dominant phase (>35% of GPU) ---
        # First phase exceeding 35% of total GPU time gets flagged.
        # Fwd cmp / Bwd cmp excluded — they are expected to dominate in FSDP.
        phases = [("AG fwd", metrics.ag_fwd_gpu), ("AG bwd", metrics.ag_bwd_gpu),
                  ("RS", metrics.rs_gpu), ("Optimizer", metrics.optimizer_gpu),
                  ("TP", tp_total)]
        for name, val in phases:
            if total_gpu > 0 and val / total_gpu > 0.35:
                issues.append(f"{name} dominates ({val/total_gpu:.1%} of total GPU)")
                break

        # --- GPU utilisation ---
        if 0 < metrics.gpu_busy < cls.UTIL_LOW_THRESHOLD:
            issues.append(f"low GPU utilization ({metrics.gpu_busy:.1%})")
        # gpu_busy is always ≤ 1.0 (capped); values near 1.0 are normal for
        # well-pipelined FSDP2 steps and do NOT indicate a bottleneck.

        # ================================================================
        # Section B — FSDP2 / TP / async TP specific bottlenecks
        # ================================================================

        # 1. Small-kernel bound (CUDA launch latency)
        # FSDP2's all_gather_copy_in launches 300+ split_with_sizes / aten::empty
        # kernels averaging <4µs — the dominant cost is CUDA driver launch latency,
        # not GPU execution.  Fires when count ≥ 100 and average <5µs.
        if metrics.kernel_count >= cls.SMALL_KERNEL_COUNT and metrics.avg_kernel_dur_us < cls.SMALL_KERNEL_AVG_DUR_US:
            issues.append(f"small-kernel-bound ({metrics.kernel_count} kernels, avg {metrics.avg_kernel_dur_us:.1f}us)")

        # 2. Serial pipeline (FSDP2 shard groups not overlapping)
        # High serial_ratio (≥85%) means layers run sequentially
        # with little overlap — GPU under-utilised.  In a well-pipelined
        # FSDP2 step, most time should be in the overlap region.
        if metrics.serial_ratio >= cls.SERIAL_RATIO_HIGH:
            issues.append(f"serial pipeline ({metrics.serial_ratio:.1%} serial)")

        # 3. Async TP overlap gap
        # Only meaningful when TP is configured (mesh_tp).  Low
        # fwd_comp_comm_overlap or bwd_comp_comm_overlap means the
        # async TP all-gather/reduce-scatter is NOT hiding behind compute,
        # defeating the purpose of async TP.
        if tp_total > 0:
            if 0 < metrics.fwd_comp_comm_overlap < cls.ASYNC_TP_OVERLAP_LOW:
                issues.append(f"low async TP overlap (fwd: {metrics.fwd_comp_comm_overlap:.1%})")
            if 0 < metrics.bwd_comp_comm_overlap < cls.ASYNC_TP_OVERLAP_LOW:
                issues.append(f"low async TP overlap (bwd: {metrics.bwd_comp_comm_overlap:.1%})")

            # Fwd vs bwd overlap asymmetry ≥40pp → pipeline unevenly utilised.
            if metrics.fwd_comp_comm_overlap > 0 and metrics.bwd_comp_comm_overlap > 0:
                diff = abs(metrics.fwd_comp_comm_overlap - metrics.bwd_comp_comm_overlap)
                if diff >= cls.OVERLAP_ASYMMETRY_DIFF:
                    issues.append(f"async TP asymmetry (fwd={metrics.fwd_comp_comm_overlap:.1%} vs bwd={metrics.bwd_comp_comm_overlap:.1%})")

        # 4. Host-bound (CPU dispatch overhead / pipeline serialisation)
        # CPU wall span > 1.5× GPU time indicates either:
        # (a) pipeline serialisation — CPU submits this layer, starts the
        #     next, and only returns much later, or
        # (b) genuine CPU dispatch overhead (CUDA kernel launch latency).
        # In fully-pipelined FSDP2, early/late layers naturally have wide
        # spans (3-10×), so this is a conservative threshold.
        if metrics.cpu_wall_to_gpu_ratio >= cls.HOST_BOUND_RATIO:
            issues.append(f"host-bound (CPU wall {metrics.cpu_wall_to_gpu_ratio:.1f}x GPU time)")

        # 5. Copy-heavy all-gather
        # Data-movement kernels (copy_in/copy_out from fsdp::split_with_sizes_copy)
        # exceed NCCL all-gather time inside the AG phase.  Common when the
        # all-gather buffer is small — memcpy dominates over network latency.
        total_copy_nccl = metrics.copy_data_movement_gpu_us + metrics.nccl_comm_gpu_us
        if total_copy_nccl > 0:
            copy_ratio = metrics.copy_data_movement_gpu_us / total_copy_nccl
            if copy_ratio >= cls.COPY_HEAVY_RATIO:
                issues.append(f"copy-heavy all-gather ({copy_ratio:.1%} copy vs NCCL)")

        # 6. Fwd / Bwd compute imbalance
        fwd_total = metrics.ag_fwd_gpu + metrics.fwd_cmp_gpu
        bwd_total = metrics.ag_bwd_gpu + metrics.bwd_cmp_gpu + metrics.rs_gpu + metrics.ar_in_rs_gpu
        max_gpu = max(fwd_total, bwd_total)
        if max_gpu > 0:
            imbalance = abs(fwd_total - bwd_total) / max_gpu
            if imbalance >= cls.FWD_BWD_IMBALANCE:
                heavier = "fwd" if fwd_total > bwd_total else "bwd"
                issues.append(f"fwd-bwd imbalance ({heavier}={imbalance:.1%} heavier)")

        # ================================================================
        # Section C — Communication-hiding / BW bottlenecks
        # ================================================================

        # 7. Inter-node bandwidth (all-gather not hidden behind compute)
        # If AG fwd GPU time ≥ 80% of fwd compute GPU, the all-gather is
        # mostly exposed — the GPU stalls waiting for AG to finish before
        # compute can start.  This is the classic inter-node BW bottleneck.
        if metrics.fwd_cmp_gpu > 0:
            ag_exposed = metrics.ag_fwd_gpu / metrics.fwd_cmp_gpu
            if ag_exposed >= cls.AG_LATENCY_EXPOSED_RATIO:
                issues.append(f"exposed all-gather (AG={ag_exposed:.1%} of fwd compute)")

        # 8. HBM bandwidth bound
        # Low GPU utilisation despite a compute-heavy profile suggests the
        # compute kernels themselves are memory-bandwidth-limited (short
        # kernels hitting HBM BW ceiling rather than compute-bound).
        if (metrics.comp_ratio >= cls.COMP_HEAVY_THRESHOLD
                and metrics.gpu_busy < cls.HBM_BOUND_GPU_UTIL
                and metrics.comp_kernel_avg_dur_us < cls.HBM_BOUND_AVG_KERNEL_US):
            issues.append(f"HBM bandwidth-bound "
                          f"(comp kernels avg {metrics.comp_kernel_avg_dur_us:.1f}us, "
                          f"busy {metrics.gpu_busy:.1%})")

        # 9. Gradient accumulation / injection bandwidth pressure
        # Reduce-scatter consumes significant GPU time relative to backward
        # compute.  RS GPU time ≥ 30% of backward compute → injection pressure
        # (the RS kernel is large enough to contend with bwd compute for GPU
        # resources, even if both run on separate streams async).
        if metrics.bwd_cmp_gpu > 0:
            rs_injection = metrics.rs_gpu / metrics.bwd_cmp_gpu
            if rs_injection >= cls.RS_INJECTION_RATIO:
                issues.append(f"RS exceeds bwd compute "
                              f"(RS={rs_injection:.1%} of bwd compute)")

        # 10. Synchronous TP on critical path
        # TP collectives exist but have low overlap with compute → they block
        # the critical path (defeating async TP's purpose).
        if metrics.tp_total_gpu > 0 and metrics.fwd_cmp_gpu > 0:
            tp_over_fwd = metrics.tp_total_gpu / metrics.fwd_cmp_gpu
            if (tp_over_fwd >= cls.TP_ON_CRITICAL_PATH_RATIO
                    and metrics.fwd_comp_comm_overlap < cls.TP_ON_CRITICAL_PATH_OVERLAP):
                issues.append(f"synchronous TP on critical path "
                              f"(TP/fwd={tp_over_fwd:.1%}, overlap={metrics.fwd_comp_comm_overlap:.1%})")

        # 11. NVLink saturation (TP kernel spray)
        # High count of very small TP kernels suggests NVLink bandwidth
        # fragmentation — too many small messages, not saturating NVLink.
        tp_kernel_count = getattr(metrics, 'tp_kernel_count', metrics.nccl_in_comp_count)
        tp_avg_us = getattr(metrics, 'tp_avg_kernel_us', metrics.comp_kernel_avg_dur_us)
        if (metrics.tp_total_gpu > 0
                and tp_kernel_count >= cls.NVLINK_SAT_COUNT
                and tp_avg_us < cls.NVLINK_SAT_AVG_US):
            issues.append(f"NVLink saturation "
                          f"({tp_kernel_count} small TP kernels, avg {tp_avg_us:.1f}us)")

        # 11b. TP contention inflation (CUPTI wall-clock artifact)
        # Only flag on the first layer to avoid redundant per-layer tags.
        # This is a trace-level artifact, not a per-layer issue.
        pass  # reported at aggregated level only

        # 12. No comm/compute overlap (FSDP2 pipeline idle)
        # GPU utilisation is low AND AG fwd is a meaningful fraction of forward
        # compute → the all-gather is not being hidden behind compute from other
        # layers (pipeline stagger is ineffective).  Uses GPU-time ratio instead
        # of overlap efficiency because async NCCL's CPU_wall is just dispatch.
        if metrics.fwd_cmp_gpu > 0:
            ag_vs_fwd = metrics.ag_fwd_gpu / metrics.fwd_cmp_gpu
            if ag_vs_fwd >= cls.AG_VS_FWD_RATIO and metrics.gpu_busy < cls.COMM_COMPUTE_UTIL_THRESHOLD:
                issues.append(f"no comm/compute overlap "
                              f"(busy={metrics.gpu_busy:.1%}, AG={ag_vs_fwd:.1%} of fwd)")

        # 13. Exposed communication (general overlap quality)
        # Fraction of wall time with no GPU activity of any kind.  Uses GPU
        # utilisation as a proxy: if most of the wall time has zero GPU kernels
        # running, the work is exposed (not hidden behind other GPU work).
        # Only fires when the layer has both forward AND backward activity
        # (otherwise low utilisation is pipeline warmup, not exposed comm).
        # Replaces the old overlap-efficiency-based formula which broke for
        # async NCCL operations (CPU_wall = dispatch time, not execution).
        gpu_idle = 1.0 - metrics.gpu_busy
        if (gpu_idle > cls.EXPOSED_COMM_IDLE_THRESHOLD
                and metrics.fwd_cmp_gpu > 0 and metrics.bwd_cmp_gpu > 0):
            issues.append(f"exposed communication "
                          f"(GPU idle={gpu_idle:.1%} of wall)")

        # 14. Low cross-layer GPU pipeline overlap
        # Adjacent layers' GPU execution spans overlap by less than 20%.
        # In a well-pipelined FSDP2 step, the GPU kernel of layer N+1's
        # all-gather (or fwd compute) should overlap with layer N's GPU
        # work.  Low overlap means layers run mostly sequentially on
        # the GPU — the pipeline stagger is not achieving its goal even
        # if CPU dispatch spans overlap.
        if 0 < metrics.pipeline_overlap_ratio < cls.PIPELINE_OVERLAP_LOW:
            issues.append(f"low cross-layer GPU overlap "
                          f"({metrics.pipeline_overlap_ratio:.1%})")

        return issues


class Report:
    """Aggregates per-layer metrics into a human-readable text report and JSON markers.
    Builds Metrics objects, computes overlap/pipeline decomposition, formats the report,
    and serialises per-unit data + bottlenecks.
    """

    def __init__(self, fsdp: FSDP, root_nodes: List[LogicalOperation],
                 output_path: Optional[str] = None,
                 model_config: Optional[ModelConfig] = None,
                 ag_per_layer: Optional[Dict[str, Tuple[float, float]]] = None):
        self.fsdp = fsdp
        self.root_nodes = root_nodes
        self.output_path = output_path
        self.model_config = model_config
        self.metrics_list: List[Metrics] = []
        self.aggregated: Dict[str, float] = {}
        self.throughput_metrics: Dict[str, float] = {}
        self._ag_per_layer = ag_per_layer

    def generate_report(self):
        """Build metrics, compute aggregates, format text + JSON, write to file if configured.
        Returns ``(report_text, markers_list)``.
        """
        num_units = len(self.fsdp.units)
        opt_gpu = self.fsdp.optimizer_gpu
        opt_cpu = self.fsdp.optimizer_cpu
        tp_ag = self.fsdp.tp_all_gather_gpu
        tp_rs = self.fsdp.tp_reduce_scatter_gpu
        tp_ar = self.fsdp.tp_all_reduce_gpu
        tp_kernels = list(self.fsdp.tp_all_gather + self.fsdp.tp_reduce_scatter + self.fsdp.tp_all_reduce)
        ag_per_layer = self._ag_per_layer if self._ag_per_layer is not None else _compute_ag_per_layer(self.root_nodes)
        rs_global_seen: Set[Tuple[float, float]] = set()
        for unit in self.fsdp.units:
            self.metrics_list.append(Metrics(unit, opt_gpu, opt_cpu, num_units,
                                             tp_ag, tp_rs, tp_ar, tp_kernels=tp_kernels,
                                             ag_per_layer=ag_per_layer,
                                             rs_global_seen=rs_global_seen))

        self._compute_aggregated()
        self._compute_throughput()
        report = self._build_report_text()

        if self.output_path:
            with open(self.output_path, 'w') as f:
                f.write(report)

        markers = self._build_json_markers()
        if self.output_path:
            json_path = self.output_path.replace('.txt', '_markers.json') if '.txt' in self.output_path else 'markers.json'
            with open(json_path, 'w') as f:
                json.dump(markers, f, indent=2)

        return report, markers

    def _compute_aggregated(self):
        """Cross-unit aggregates: sweep-line overlap decomposition + per-phase averages.
        Averages are stored in ``self.aggregated`` AND back-propagated to each ``Metrics``
        instance so bottleneck classification can use global overlap/step_wall values.
        """
        keys = ["ag_fwd_gpu_us", "fwd_cmp_gpu_us", "ag_bwd_gpu_us", "bwd_cmp_gpu_us",
                "rs_gpu_us", "ar_in_rs_gpu_us", "optimizer_gpu_us", "tp_ag_gpu_us", "tp_rs_gpu_us",
                "tp_ar_gpu_us", "tp_total_gpu_us",
                "total_gpu_us", "total_cpu_us",
                "comm_ratio", "comp_ratio", "optimizer_ratio",
                "overlap_ratio", "serial_ratio", "idle_ratio"]
        self.aggregated = {k: 0.0 for k in keys}
        count = len(self.metrics_list)

        # Compute overlap metrics from actual unit spans (needed even when
        # metrics_list is empty so _build_report_text doesn't crash)
        ov = _compute_overlap_metrics(self.fsdp.units)
        self.overlap_metrics = ov

        if count == 0:
            return

        for m in self.metrics_list:
            d = m.to_dict()
            for k in keys:
                self.aggregated[k] += d.get(k, 0.0)
            # Set overlap fields on each Metrics instance
            m.overlap_ratio = ov['overlap_ratio']
            m.serial_ratio = ov['serial_ratio']
            m.idle_ratio = ov['idle_ratio']
            m.step_wall = ov['step_wall']

        for k in keys:
            self.aggregated[k] /= count
        self.aggregated["num_units"] = count
        self.aggregated["overlap_ratio"] = ov['overlap_ratio']
        self.aggregated["serial_ratio"] = ov['serial_ratio']
        self.aggregated["idle_ratio"] = ov['idle_ratio']
        self.aggregated["step_wall"] = ov['step_wall']
        self.aggregated["overlap_time"] = ov['overlap_time']
        self.aggregated["serial_time"] = ov['serial_time']

        # Cross-layer GPU pipeline overlap: consecutive layers' GPU spans
        cross_layer_avg, per_layer_overlaps = _compute_cross_layer_overlap(self.fsdp.units)
        self.aggregated["pipeline_overlap_ratio"] = cross_layer_avg
        for i, m in enumerate(self.metrics_list):
            if i < len(per_layer_overlaps):
                m.pipeline_overlap_ratio = per_layer_overlaps[i]

        # TP contention inflation: compare average TP kernel duration to
        # the 25th-percentile TP kernel duration (uncontested baseline).
        # The 25th percentile rejects NCCL setup/control kernels (<20 µs)
        # while still capturing uncontested data-movement kernels.  When
        # all TP kernels are inflated (high compute overlap), the 25th
        # percentile is also inflated and the ratio stays low — this is
        # intentional: the inflation metric measures dispersion, not
        # absolute inflation versus an external baseline.
        tp_kernels = (self.fsdp.tp_all_gather
                      + self.fsdp.tp_reduce_scatter
                      + self.fsdp.tp_all_reduce)
        if tp_kernels and len(tp_kernels) > 0:
            durs = sorted(k.get('dur', 0) for k in tp_kernels)
            baseline = durs[len(durs) // 4] if len(durs) >= 4 else durs[0]
            avg_dur = sum(durs) / len(durs)
            inflation = avg_dur / baseline if baseline > 0 else 1.0
            self.aggregated["tp_contention_inflation"] = inflation
            for m in self.metrics_list:
                m.tp_contention_inflation = inflation
                m.tp_effective_gpu_us = m.tp_total_gpu / inflation if inflation > 1.0 else m.tp_total_gpu

    def _compute_throughput(self):
        """Compute MFU / HFU / tokens-per-second from model config and step wall.

        Populates ``self.throughput_metrics`` dict and calls
        ``compute_throughput_metrics`` on each per-layer Metrics instance.
        """
        cfg = self.model_config
        self.throughput_metrics = {
            'tokens_per_second_per_gpu': 0.0,
            'mfu': 0.0,
            'hfu': 0.0,
            'estimated_flops_per_step': 0.0,
        }
        if cfg is None or not cfg.is_configured:
            return

        step_wall_us = self.aggregated.get('step_wall', 0.0)
        if step_wall_us <= 0:
            return

        # Compute throughput on first layer (step-level metrics, identical across all layers)
        first = self.metrics_list[0] if self.metrics_list else None
        if first is None:
            return
        first.compute_throughput_metrics(cfg)

        self.throughput_metrics = {
            'tokens_per_second_per_gpu': first.tokens_per_second_per_gpu,
            'mfu': first.mfu,
            'hfu': first.hfu,
            'estimated_flops_per_step': first.estimated_flops_per_step,
        }

    def _build_report_text(self) -> str:
        """Assemble the text report: step summary → phase metrics → efficiency →
        overlap & pipeline → memory → per-unit table → bottleneck summary → timelines.
        Simple string concatenation, zero templating dependencies.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("FSDP Trace Analysis Report")
        lines.append("=" * 70)
        lines.append("")

        num_units = self.aggregated.get("num_units", 0)
        opt_gpu = self.fsdp.optimizer_gpu
        step_wall = self.aggregated.get("step_wall", self.fsdp.step_wall)

        if num_units == 0 or not self.metrics_list:
            lines.append("  No FSDP units detected — trace may be CPU-only or non-FSDP.")
            if step_wall > 0:
                lines.append(f"  Step wall time:       {_format_us(step_wall)}")
            return "\n".join(lines), []

        # Step Summary
        lines.append("--- Step Summary ---")
        lines.append(f"  Number of layers:     {int(num_units)}")
        lines.append(f"  Step wall time:       {_format_us(step_wall)}")
        if step_wall > 0:
            lines.append(f"  Estimated throughput: {1_000_000 / step_wall:.1f} steps/s")
        lines.append("")

        # Aggregated phase metrics
        # All values are AVERAGE GPU time per layer (sum of kernel durations,
        # NOT a union — so phases can overlap on different CUDA streams and
        # percentages reflect GPU-cycle share, not wall-time share).
        lines.append("--- Phase Metrics (avg GPU cycles per layer) ---")
        phase_keys = [
            ("ag_fwd_gpu_us", "All-gather fwd"),
            ("fwd_cmp_gpu_us", "Fwd compute"),
            ("tp_ag_gpu_us", "  TP all-gather"),
            ("tp_ar_gpu_us", "  TP all-reduce"),
            ("ag_bwd_gpu_us", "All-gather bwd"),
            ("bwd_cmp_gpu_us", "Bwd compute"),
            ("rs_gpu_us", "Reduce-scatter"),
            ("ar_in_rs_gpu_us", "  All-reduce in RS"),
            ("tp_rs_gpu_us", "  TP reduce-scatter"),
            ("optimizer_gpu_us", "Optimizer step"),
        ]
        total_gpu = self.aggregated.get("total_gpu_us", 0)
        tp_total = self.aggregated.get("tp_total_gpu_us", 0)
        lines.append(f"  {'Phase':25s} {'Avg':>10s} {'% GPU cycles':>14s}")
        lines.append(f"  {'-----':25s} {'---':>10s} {'------------':>14s}")
        for key, label in phase_keys:
            avg = self.aggregated.get(key, 0)
            pct = avg / (total_gpu + tp_total) * 100 if (total_gpu + tp_total) > 0 else 0
            lines.append(f"  {label:25s} {_format_us(avg):>10s} {pct:>13.1f}%")
        lines.append(f"  {'-----':25s} {'---':>10s} {'------------':>14s}")
        lines.append(f"  {'FSDP total':25s} {_format_us(total_gpu):>10s}")
        total_with_tp = total_gpu + tp_total
        lines.append(f"  {'TP total':25s} {_format_us(tp_total):>10s} {tp_total / total_with_tp * 100:>13.1f}%" if total_with_tp > 0 else f"  {'TP total':25s} {_format_us(tp_total):>10s} {'N/A':>13s}")
        lines.append(f"  {'Total CPU (dispatch)':25s} {_format_us(self.aggregated.get('total_cpu_us', 0)):>10s}")
        lines.append("")

        # Throughput (MFU / HFU / tokens/sec) — requires ModelConfig
        tp = self.throughput_metrics
        if tp.get('mfu', 0) > 0:
            lines.append("--- Throughput ---")
            lines.append(f"  MFU:                   {tp['mfu']:.1%}")
            if tp.get('hfu', 0) > 0 and abs(tp['hfu'] - tp['mfu']) > 0.001:
                lines.append(f"  HFU:                   {tp['hfu']:.1%} (w/ activation checkpointing)")
            lines.append(f"  Tokens/sec/GPU:        {tp['tokens_per_second_per_gpu']:.1f}")
            lines.append(f"  Estimated FLOPs/step:  {tp['estimated_flops_per_step']:.2e}")
            lines.append(f"  Step wall:             {_format_us(self.aggregated.get('step_wall', 0))}")
            lines.append("")

        # Efficiency metrics
        lines.append("--- Efficiency ---")
        comp_gpu = self.aggregated.get("fwd_cmp_gpu_us", 0) + self.aggregated.get("bwd_cmp_gpu_us", 0)
        tp_total = self.aggregated.get("tp_total_gpu_us", 0)
        fsdp_comm = (self.aggregated.get("ag_fwd_gpu_us", 0) + self.aggregated.get("ag_bwd_gpu_us", 0)
                     + self.aggregated.get("rs_gpu_us", 0) + self.aggregated.get("ar_in_rs_gpu_us", 0))
        opt_ratio = self.aggregated.get('optimizer_ratio', 0)
        total = total_gpu + tp_total
        if total > 0:
            lines.append(f"  Compute time:         {comp_gpu / total:.1%}")
            lines.append(f"  TP communication:     {tp_total / total:.1%}")
            lines.append(f"  FSDP communication:   {fsdp_comm / total:.1%}")
        else:
            lines.append(f"  GPU time:             0.0 (no GPU events)")
        lines.append(f"  Optimizer:            {opt_ratio:.1%}")
        avg_busy = sum(m.gpu_busy for m in self.metrics_list) / len(self.metrics_list) if self.metrics_list else 0.0
        lines.append(f"  Avg GPU busy:         {avg_busy:.1%} (per-layer GPU active / layer span)")
        max_span = max(m.layer_span for m in self.metrics_list) if self.metrics_list else 0.0
        non_zero = [m.layer_span for m in self.metrics_list if m.layer_span > 0]
        min_span = min(non_zero) if non_zero else max_span
        lines.append(f"  Layer span imbalance: {max_span / min_span:.1f}x (max/min layer span ratio)" if max_span > 0 and min_span > 0 else "  Layer span imbalance: N/A (no nonzero span)")

        # Compute-to-communicate ratio
        avg_ctc = sum(m.compute_to_comm_ratio for m in self.metrics_list) / len(self.metrics_list) if self.metrics_list else 0.0
        lines.append(f"  Compute-to-comm:      {avg_ctc:.2f}× (compute GPU / comm GPU)")
        lines.append("")

        # Per-phase exposed ratio
        lines.append("--- Comm Exposed Ratio ---")
        lines.append(f"  {'Phase':25s} {'Avg exp':>10s} {'Min':>10s} {'Max':>10s} {'Interpretation':>35s}")
        lines.append(f"  {'-----':25s} {'--------':>10s} {'---':>10s} {'---':>10s} {'---------------':>35s}")
        for label, attr in [("AG fwd", "ag_fwd_exposed_ratio"),
                            ("AG bwd", "ag_bwd_exposed_ratio"),
                            ("RS", "rs_exposed_ratio")]:
            vals = [getattr(m, attr) for m in self.metrics_list]
            avg = sum(vals) / len(vals) if vals else 0.0
            mn = min(vals) if vals else 0.0
            mx = max(vals) if vals else 0.0
            interp = "exposed" if avg >= 0.7 else "hidden" if avg < 0.3 else "partial"
            lines.append(f"  {label:25s} {avg:>9.1%} {mn:>9.1%} {mx:>9.1%}  {interp:>35s}")
        avg_exp = sum(m.avg_exposed_ratio for m in self.metrics_list) / len(self.metrics_list) if self.metrics_list else 0.0
        lines.append(f"  {'Avg exposed ratio':25s} {avg_exp:>9.1%}")
        lines.append("")

        # Overlap & Pipeline
        lines.append("--- Overlap & Pipeline ---")
        ov = self.overlap_metrics
        lines.append(f"  Overlap time:         {_format_us(ov['overlap_time'])} ({ov['overlap_ratio']:.1%} of non-idle)")
        lines.append(f"  Serial execution:     {_format_us(ov['serial_time'])} ({ov['serial_ratio']:.1%} of step)")
        lines.append(f"  Idle/Gap time:        {_format_us(ov['idle_time'])} ({ov['idle_ratio']:.1%} of step)")
        lines.append(f"  Communication ratio:  {fsdp_comm / total:.1%} FSDP + {tp_total / total:.1%} TP = {self.aggregated.get('comm_ratio', 0):.1%} total")
        avg_fwd_ov = sum(m.fwd_comp_comm_overlap for m in self.metrics_list) / max(len(self.metrics_list), 1)
        avg_bwd_ov = sum(m.bwd_comp_comm_overlap for m in self.metrics_list) / max(len(self.metrics_list), 1)
        lines.append(f"  Avg Fwd comp-comm:    {avg_fwd_ov:.1%} (avg TP overlap during fwd compute)")
        lines.append(f"  Avg Bwd comp-comm:    {avg_bwd_ov:.1%} (avg TP overlap during bwd compute)")
        avg_po = self.aggregated.get("pipeline_overlap_ratio", 0.0)
        lines.append(f"  Pipeline cross-layer: {avg_po:.1%} (GPU overlap between adjacent layers)")
        lines.append("")

        # Memory
        lines.append("--- Memory ---")
        mem_available = any(m.memory_has_data for m in self.metrics_list)
        if mem_available:
            total_alloc = sum(m.memory_allocated for m in self.metrics_list)
            total_free = sum(m.memory_freed for m in self.metrics_list)
            peak = max(m.memory_peak for m in self.metrics_list)
            lines.append(f"  Peak memory:          {peak / (1024**3):.2f} GiB")
            lines.append(f"  Total allocated:      {total_alloc / (1024**3):.2f} GiB")
            lines.append(f"  Total freed:          {total_free / (1024**3):.2f} GiB")
        else:
            lines.append(f"  {_format_us(0):>10s}  Memory profiling not enabled in this trace. No allocation/free events recorded.")
        lines.append("")

        # Per-unit table
        lines.append("--- Per-Unit Metrics ---")
        mem_available = any(m.memory_has_data for m in self.metrics_list)
        mem_col = " Mem" if mem_available else ""
        header = (f"{'Layer':25s} {'AG fwd':>10s} {'Fwd cmp':>10s} {'TP AG':>9s} {'TP RS':>9s} "
                  f"{'TP AR':>9s} {'AG bwd':>10s} {'Bwd cmp':>10s} {'RS':>10s} "
                  f"{'Opt':>10s} {'Total':>10s} "
                  f"{'Busy':>6s}{'CtC':>6s}{'ExpC':>6s}"
                  f"{'Span':>9s}{'F-Ovl':>8s}{'B-Ovl':>8s}{'P-Ovl':>7s} "
                  f"{'K#':>7s}{'AvgK':>7s}{mem_col:>10s}  {'Bottleneck'}")
        lines.append(header)
        lines.append("-" * len(header))
        for m in self.metrics_list:
            issues = Bottlenecks.detect(m)
            d = m.to_dict()
            label = "; ".join(issues) if issues else "OK"
            mem_str = ""
            if mem_available and d.get('memory_peak', 0) > 0:
                mem_str = f"{d['memory_peak']/(1024**3):>9.1f}G"
            elif mem_available:
                mem_str = f"{'N/A':>9s}"
            ctc = d.get('compute_to_comm_ratio', 0)
            ctc_str = f"{ctc:.1f}x" if ctc != float('inf') else "inf"
            exp_comm = d.get('avg_exposed_ratio', 0)
            lines.append(
                f"{m.layer_name:25s} "
                f"{_format_us(d['ag_fwd_gpu_us']):>10s} "
                f"{_format_us(d['fwd_cmp_gpu_us']):>10s} "
                f"{_format_us(d['tp_ag_gpu_us']):>9s} "
                f"{_format_us(d['tp_rs_gpu_us']):>9s} "
                f"{_format_us(d['tp_ar_gpu_us']):>9s} "
                f"{_format_us(d['ag_bwd_gpu_us']):>10s} "
                f"{_format_us(d['bwd_cmp_gpu_us']):>10s} "
                f"{_format_us(d['rs_gpu_us']):>10s} "
                f"{_format_us(d['optimizer_gpu_us']):>10s} "
                f"{_format_us(d['total_gpu_us']):>10s} "
                f"{d['gpu_busy']:>5.1%} "
                f"{ctc_str:>5s} "
                f"{exp_comm:>5.1%} "
                f"{_format_us(d['layer_span_us']):>9s}"
                f"{d['fwd_comp_comm_overlap']:>7.1%} "
                f"{d['bwd_comp_comm_overlap']:>7.1%} "
                f"{m.pipeline_overlap_ratio:>6.1%} "
                f"{d['kernel_count']:>7d} "
                f"{d['avg_kernel_dur_us']:>5.1f}us "
                f"{mem_str:>10s}  "
                f"{label}"
            )
        lines.append("")

        # Bottleneck summary
        lines.append("--- Bottleneck Summary ---")
        all_issues = defaultdict(list)
        for m in self.metrics_list:
            issues = Bottlenecks.detect(m)
            for iss in issues:
                all_issues[iss].append(m.layer_name)

        if all_issues:
            for iss, layers in sorted(all_issues.items()):
                lines.append(f"  {iss}: {len(layers)} units ({', '.join(layers[:5])}{'...' if len(layers) > 5 else ''})")
        else:
            lines.append("  No bottlenecks detected.")
        lines.append("")

        # Chronological timeline
        lines.append("--- Chronological Timeline ---")
        timeline = self.get_fsdp_chronological_timeline(self.root_nodes)
        lines.append(timeline)

        # Aggregated timeline
        lines.append("--- Aggregated Timeline ---")
        agg_str = self.get_fsdp_timeline_aggregated_string(self.aggregated)
        lines.append(agg_str)

        return "\n".join(lines)

    def _build_json_markers(self) -> List[dict]:
        """One dict per layer: metrics + bottleneck labels for ``_markers.json`` output.
        Includes aggregated throughput metrics (MFU, HFU, TPS) in the first entry.
        """
        markers = []
        for i, m in enumerate(self.metrics_list):
            d = m.to_dict()
            issues = Bottlenecks.detect(m)
            entry = {
                "layer": m.layer_name,
                "metrics": d,
                "bottlenecks": issues,
            }
            if i == 0 and self.throughput_metrics.get('mfu', 0) > 0:
                entry["throughput"] = {
                    "mfu": self.throughput_metrics['mfu'],
                    "hfu": self.throughput_metrics['hfu'],
                    "tokens_per_second_per_gpu": self.throughput_metrics['tokens_per_second_per_gpu'],
                    "estimated_flops_per_step": self.throughput_metrics['estimated_flops_per_step'],
                }
                entry["aggregated"] = {k: v for k, v in self.aggregated.items()
                                       if isinstance(v, (int, float))}
            markers.append(entry)
        return markers

    @staticmethod
    def get_fsdp_timeline_aggregated_string(agg: Dict[str, float]) -> str:
        """One-line summary of average phase GPU times + overlap ratio, appended to the report.
        """
        parts = []
        for key in ["ag_fwd_gpu_us", "fwd_cmp_gpu_us", "ag_bwd_gpu_us", "bwd_cmp_gpu_us",
                     "rs_gpu_us", "optimizer_gpu_us"]:
            label = key.replace("_gpu_us", "").replace("_", " ")
            parts.append(f"{label}: {_format_us(agg.get(key, 0))}")
        ov = agg.get("overlap_ratio", 0)
        parts.append(f"overlap: {ov:.1%}")
        return " | ".join(parts)

    @staticmethod
    def get_fsdp_chronological_timeline(roots: List[LogicalOperation]) -> str:
        """First 80 FSDP events sorted by CPU start time with offsets, for debugging
        phase boundaries.  Skips ``backward_prefetch`` noise.
        """
        import heapq
        from trace_parser import TraceParserHelper

        all_nodes = sorted(TraceParserHelper.iter_nodes(roots), key=lambda n: n.start_time)
        fsdp_events = [(n.start_time, n.name, n) for n in all_nodes
                       if (_is_fsdp_name(n.name) or n.name.startswith('Optimizer.'))
                       and 'backward_prefetch' not in n.name]

        if not fsdp_events:
            return "  No FSDP events found in tree."

        base_time = min(t for t, _, _ in fsdp_events)

        lines = []
        for ts, name, node in fsdp_events[:80]:
            offset = ts - base_time
            label = name
            for p in FSDP_PREFIXES:
                if label.startswith(p):
                    label = label[len(p):]
                    break
            lines.append(f"  t+{offset:>12.1f}us  {label}  (cpu={node.cpu_duration:.0f}us gpu={node.gpu_duration:.0f}us)")

        if len(fsdp_events) > 80:
            lines.append(f"  ... and {len(fsdp_events) - 80} more events")

        return "\n".join(lines)
