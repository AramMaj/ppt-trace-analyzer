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
   inter-node BW, HBM bandwidth, sync TP on critical path, NVLink saturation, etc.)
7. Report class — aggregation, formatting, JSON markers
"""

import json
from typing import Dict, List, Optional, Set, Iterator, Tuple, Any
from collections import defaultdict

from trace_parser import LogicalOperation
from fsdp_detector import FSDP, FSDPUnit, FSDP_PREFIXES

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

def _phase_gpu_time(nodes: List[LogicalOperation]) -> float:
    """Sum of inclusive GPU duration across nodes (aggregate CUDA kernel time,
    NCCL collective time, memcpy).  ``gpu_duration`` is already the inclusive
    sum per node, so this is not a unioned time range.
    """
    return sum(n.gpu_duration for n in nodes)


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
                'step_wall': 0.0, 'overlap_ratio': 0.0, 'serial_exec_efficiency': 1.0}

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
        'serial_exec_efficiency': serial_time / total if total > 0 else 1.0,
        'idle_ratio': idle_time / total if total > 0 else 0.0,
    }


def _collect_memory(unit: 'FSDPUnit', metrics: 'Metrics'):
    """Sum ``memory_delta`` (activation memory, parameter memory, gradient buffers)
    across all FSDP phase nodes.  Peak is max(allocated, freed), approximating
    ``torch.cuda.max_memory_allocated`` per layer.  Only meaningful when
    ``profile_memory=True`` in ``torch.profiler`` — guarded by ``memory_has_data``.
    """
    all_nodes = (unit.all_gather_fwd + unit.fwd_compute + unit.all_gather_bwd
                 + unit.bwd_compute + unit.reduce_scatter)
    total_alloc = 0
    total_free = 0
    for n in all_nodes:
        delta = n.memory_delta
        if delta > 0:
            total_alloc += delta
        elif delta < 0:
            total_free += -delta
    if total_alloc > 0 or total_free > 0:
        metrics.memory_has_data = True
        metrics.memory_allocated = total_alloc
        metrics.memory_freed = total_free
        metrics.memory_peak = max(total_alloc, total_free)


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
        attn_flops = 4 * h * h          # QKV projection + output projection (ignoring heads)
        mlp_flops = 3 * 2 * h * ffn     # gate_proj + up_proj + down_proj (×2 for FWD)
        per_layer_flops = attn_flops + mlp_flops

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
                 tp_kernels: Optional[List[dict]] = None):
        self.layer_name = unit.layer_name

        # --- Phase raw times — these are the building blocks for everything ---
        self.ag_fwd_gpu = _phase_gpu_time(unit.all_gather_fwd)
        self.ag_fwd_cpu = _phase_cpu_time(unit.all_gather_fwd)
        self.ag_fwd_wall = _phase_wall_time(unit.all_gather_fwd)

        self.fwd_cmp_gpu = _phase_gpu_time(unit.fwd_compute)
        self.fwd_cmp_cpu = _phase_cpu_time(unit.fwd_compute)
        self.fwd_cmp_wall = _phase_wall_time(unit.fwd_compute, unit.fwd_compute_span)

        self.ag_bwd_gpu = _phase_gpu_time(unit.all_gather_bwd) + unit.ag_bwd_supplement_us
        self.ag_bwd_cpu = _phase_cpu_time(unit.all_gather_bwd)
        self.ag_bwd_wall = _phase_wall_time(unit.all_gather_bwd)

        self.bwd_cmp_gpu = _phase_gpu_time(unit.bwd_compute)
        self.bwd_cmp_cpu = _phase_cpu_time(unit.bwd_compute)
        self.bwd_cmp_wall = _phase_wall_time(unit.bwd_compute, unit.bwd_compute_span)

        self.rs_gpu = _phase_gpu_time(unit.reduce_scatter)
        self.rs_cpu = _phase_cpu_time(unit.reduce_scatter)
        self.rs_wall = _phase_wall_time(unit.reduce_scatter)

        # --- Optimizer (evenly split across layers) ---
        self.optimizer_gpu = global_optimizer_gpu / num_units if num_units > 0 else 0.0
        self.optimizer_cpu = global_optimizer_cpu / num_units if num_units > 0 else 0.0

        # --- Totals (all FSDP phases) ---
        self.total_gpu = (self.ag_fwd_gpu + self.fwd_cmp_gpu + self.ag_bwd_gpu
                          + self.bwd_cmp_gpu + self.rs_gpu + self.optimizer_gpu)
        self.total_cpu = (self.ag_fwd_cpu + self.fwd_cmp_cpu + self.ag_bwd_cpu
                          + self.bwd_cmp_cpu + self.rs_cpu + self.optimizer_cpu)

        # --- Communication vs compute ratio ---
        comm_gpu = self.ag_fwd_gpu + self.ag_bwd_gpu + self.rs_gpu  # FSDP NCCL
        comp_gpu = self.fwd_cmp_gpu + self.bwd_cmp_gpu  # attn/MLP/norm
        self.comm_ratio = comm_gpu / self.total_gpu if self.total_gpu > 0 else 0.0
        self.comp_ratio = comp_gpu / self.total_gpu if self.total_gpu > 0 else 0.0

        self.optimizer_ratio = self.optimizer_gpu / self.total_gpu if self.total_gpu > 0 else 0.0

        # --- Per-phase overlap efficiency ---
        # How well each collective is hidden behind compute:
        #   GPU time / CPU wall span → 1.0 = fully exposed (no overlap),
        #   << 1.0 = well hidden (GPU execution overlaps with other work).
        self.ag_fwd_overlap_efficiency = (
            self.ag_fwd_gpu / self.ag_fwd_wall if self.ag_fwd_wall > 0 else 0.0
        )
        self.rs_overlap_efficiency = (
            self.rs_gpu / self.rs_wall if self.rs_wall > 0 else 0.0
        )
        self.ag_bwd_overlap_efficiency = (
            self.ag_bwd_gpu / self.ag_bwd_wall if self.ag_bwd_wall > 0 else 0.0
        )

        # --- Layer wall span + GPU utilisation ---
        all_events = (unit.all_gather_fwd + unit.fwd_compute + unit.all_gather_bwd
                      + unit.bwd_compute + unit.reduce_scatter)
        self.layer_span = _phase_wall_time(all_events) if all_events else 0.0
        self.gpu_util = self.total_gpu / self.layer_span if self.layer_span > 0 else 0.0

        # --- TP metrics (evenly split across layers) ---
        self.tp_ag_gpu = global_tp_ag_gpu / num_units if num_units > 0 else 0.0
        self.tp_rs_gpu = global_tp_rs_gpu / num_units if num_units > 0 else 0.0
        self.tp_ar_gpu = global_tp_ar_gpu / num_units if num_units > 0 else 0.0
        self.tp_total_gpu = self.tp_ag_gpu + self.tp_rs_gpu + self.tp_ar_gpu

        # --- Kernel counts per phase (small-kernel detection) ---
        self.ag_fwd_count = len(unit.all_gather_fwd_gpu_kernels)
        self.ag_bwd_count = len(unit.all_gather_bwd_gpu_kernels)
        self.rs_count = len(unit.reduce_scatter_gpu_kernels)

        # --- Per-layer async TP overlap ---
        self.fwd_comp_comm_overlap = 0.0
        self.bwd_comp_comm_overlap = 0.0
        self.pipeline_overlap_ratio = 0.0
        if tp_kernels and unit.fwd_compute and unit.bwd_compute:
            fwd_start = unit.fwd_compute[0].start_time
            fwd_end = unit.fwd_compute[-1].end_time
            bwd_start = unit.bwd_compute[0].start_time
            bwd_end = unit.bwd_compute[-1].end_time
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
        self.serial_exec_efficiency = 1.0
        self.idle_ratio = 0.0
        self.step_wall = 0.0

        # --- Communication ratio including TP ---
        total_gpu = self.total_gpu + self.tp_total_gpu
        fsdp_comm = self.ag_fwd_gpu + self.ag_bwd_gpu + self.rs_gpu
        self.fsdp_comm_ratio = fsdp_comm / total_gpu if total_gpu > 0 else 0.0
        self.tp_comm_ratio = self.tp_total_gpu / total_gpu if total_gpu > 0 else 0.0
        self.comm_ratio = (fsdp_comm + self.tp_total_gpu) / total_gpu if total_gpu > 0 else 0.0

        # --- Compute-to-communicate ratio ---
        # Arithmetic intensity proxy: compute GPU time ÷ all comm GPU time.
        # Higher = more compute-heavy; <1.0 = comm-dominated.
        all_comm = fsdp_comm + self.tp_total_gpu
        comp = self.fwd_cmp_gpu + self.bwd_cmp_gpu
        self.compute_to_comm_ratio = comp / all_comm if all_comm > 0 else float('inf')

        # --- Exposed communication fraction ---
        # Fraction of FSDP communication that is NOT hidden behind compute.
        # Computed from overlap efficiency: if AG runs in 100µs but spans 900µs
        # of wall time, only 100/900 = 11% is "exposed" (the rest overlaps
        # with other streams / compute).  Exposed = 1 - efficiency.
        # Higher exposed_comm → worse overlap → more pipeline bubble.
        self.exposed_comm_fraction = 0.0
        effs = [self.ag_fwd_overlap_efficiency, self.rs_overlap_efficiency,
                self.ag_bwd_overlap_efficiency]
        valid = [e for e in effs if e > 0]
        if valid:
            self.exposed_comm_fraction = sum(valid) / len(valid)

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
        # Total bytes estimate: assume ~2 bytes per FLOP for bf16 matmul
        self.estimated_bytes_moved = self.comp_kernel_count * self.comp_kernel_avg_dur_us * 2e9 / 1e6  # rough

    def to_dict(self) -> Dict[str, float]:
        return {
            "ag_fwd_gpu_us": self.ag_fwd_gpu,
            "fwd_cmp_gpu_us": self.fwd_cmp_gpu,
            "ag_bwd_gpu_us": self.ag_bwd_gpu,
            "bwd_cmp_gpu_us": self.bwd_cmp_gpu,
            "rs_gpu_us": self.rs_gpu,
            "optimizer_gpu_us": self.optimizer_gpu,
            "tp_ag_gpu_us": self.tp_ag_gpu,
            "tp_rs_gpu_us": self.tp_rs_gpu,
            "tp_ar_gpu_us": self.tp_ar_gpu,
            "tp_total_gpu_us": self.tp_total_gpu,
            "total_gpu_us": self.total_gpu,
            "total_cpu_us": self.total_cpu,
            "comm_ratio": self.comm_ratio,
            "comp_ratio": self.comp_ratio,
            "optimizer_ratio": self.optimizer_ratio,
            "gpu_util": self.gpu_util,
            "layer_span_us": self.layer_span,
            "overlap_ratio": self.overlap_ratio,
            "serial_exec_efficiency": self.serial_exec_efficiency,
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
            "ag_fwd_overlap_efficiency": self.ag_fwd_overlap_efficiency,
            "rs_overlap_efficiency": self.rs_overlap_efficiency,
            "ag_bwd_overlap_efficiency": self.ag_bwd_overlap_efficiency,
            "exposed_comm_fraction": self.exposed_comm_fraction,
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
    """Threshold-based bottleneck classification.  Each class attribute is a
    tunable heuristic derived from FSDP2 + TP workloads.  Should be revisited
    for different architectures or hardware.
    """

    # --- Classical bottleneck thresholds ---
    COMP_HEAVY_THRESHOLD = 0.70
    COMM_HEAVY_THRESHOLD = 0.40
    IO_HEAVY_THRESHOLD = 0.15
    AG_HEAVY_THRESHOLD = 0.40
    RS_HEAVY_THRESHOLD = 0.40
    TP_HEAVY_THRESHOLD = 0.15
    OPTIMIZER_HEAVY_THRESHOLD = 0.15
    UTIL_LOW_THRESHOLD = 0.50

    # --- FSDP2 / TP / async TP specific thresholds ---
    SMALL_KERNEL_AVG_DUR_US = 5.0
    SMALL_KERNEL_COUNT = 100
    ASYNC_TP_OVERLAP_LOW = 0.30
    OVERLAP_ASYMMETRY_DIFF = 0.40
    HOST_BOUND_RATIO = 1.5
    COPY_HEAVY_RATIO = 0.50
    FWD_BWD_IMBALANCE = 0.30
    SERIAL_EXEC_HIGH = 0.85

    # --- New FSDP2 communication bottleneck thresholds ---
    # Inter-node bandwidth: AG fwd GPU time ≥ 80% of fwd compute → BW-limited
    AG_LATENCY_EXPOSED_RATIO = 0.80
    # HBM bandwidth: compute kernels avg < 8µs and GPU util < 50%
    # suggests memory-bandwidth-bound (kernels too short to be arithmetic-heavy)
    HBM_BOUND_AVG_KERNEL_US = 8.0
    HBM_BOUND_GPU_UTIL = 0.50
    # Gradient accumulation / injection pressure: RS not hidden behind compute,
    # RS overlap efficiency ≥ 0.70 means RS GPU time dominates its wall span
    RS_EXPOSED_THRESHOLD = 0.70
    # Synchronous TP: TP GPU time ≥ 10% of compute, but TP overlap < 0.20
    # (TP collectives block on critical path, defeating async TP)
    TP_ON_CRITICAL_PATH_RATIO = 0.10
    TP_ON_CRITICAL_PATH_OVERLAP = 0.20
    # NVLink saturation: many (≥ 50) small (< 10µs avg) TP kernels
    NVLINK_SAT_COUNT = 50
    NVLINK_SAT_AVG_US = 10.0

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

        # --- Compute-bound ---
        # attn/MLP/norm (fwd_cmp + bwd_cmp) dominates total GPU.
        comp_gpu = metrics.fwd_cmp_gpu + metrics.bwd_cmp_gpu
        comp_ratio = comp_gpu / total_gpu if total_gpu > 0 else 0.0
        if comp_ratio >= cls.COMP_HEAVY_THRESHOLD:
            issues.append(f"compute-bound (comp={comp_ratio:.1%})")

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
        phases = [("AG fwd", metrics.ag_fwd_gpu), ("AG bwd", metrics.ag_bwd_gpu),
                  ("RS", metrics.rs_gpu), ("Fwd cmp", metrics.fwd_cmp_gpu),
                  ("Bwd cmp", metrics.bwd_cmp_gpu), ("Optimizer", metrics.optimizer_gpu),
                  ("TP", tp_total)]
        for name, val in phases:
            if total_gpu > 0 and val / total_gpu > 0.35:
                issues.append(f"{name} dominates ({val/total_gpu:.1%} of total GPU)")
                break

        # --- GPU utilisation ---
        if 0 < metrics.gpu_util < cls.UTIL_LOW_THRESHOLD:
            issues.append(f"low GPU utilization ({metrics.gpu_util:.1%})")
        if metrics.gpu_util > 1.0:
            issues.append(f"high compute-comm overlap ({metrics.gpu_util:.1%} util)")

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
        # High serial_exec_efficiency (≥85%) means layers run sequentially
        # with little overlap — GPU under-utilised.  In a well-pipelined
        # FSDP2 step, most time should be in the overlap region.
        if metrics.serial_exec_efficiency >= cls.SERIAL_EXEC_HIGH:
            issues.append(f"serial pipeline ({metrics.serial_exec_efficiency:.1%} serial)")

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
        bwd_total = metrics.ag_bwd_gpu + metrics.bwd_cmp_gpu + metrics.rs_gpu
        max_gpu = max(fwd_total, bwd_total)
        if max_gpu > 0:
            imbalance = abs(fwd_total - bwd_total) / max_gpu
            if imbalance >= cls.FWD_BWD_IMBALANCE:
                heavier = "fwd" if fwd_total > bwd_total else "bwd"
                issues.append(f"fwd-bwd imbalance ({heavier}={imbalance:.1%} heavier)")

        # ================================================================
        # Section C — Communication-hiding / BW bottlenecks (new)
        # ================================================================

        # 7. Inter-node bandwidth (all-gather not hidden behind compute)
        # If AG fwd GPU time ≥ 80% of fwd compute GPU, the all-gather is
        # mostly exposed — the GPU stalls waiting for AG to finish before
        # compute can start.  This is the classic inter-node BW bottleneck.
        if metrics.fwd_cmp_gpu > 0:
            ag_exposed = metrics.ag_fwd_gpu / metrics.fwd_cmp_gpu
            if ag_exposed >= cls.AG_LATENCY_EXPOSED_RATIO:
                issues.append(f"inter-node BW (AG={ag_exposed:.1%} of fwd compute)")

        # 8. HBM bandwidth bound
        # Low GPU utilisation despite a compute-heavy profile suggests the
        # compute kernels themselves are memory-bandwidth-limited (short
        # kernels hitting HBM BW ceiling rather than compute-bound).
        if (metrics.comp_ratio >= cls.COMP_HEAVY_THRESHOLD
                and metrics.gpu_util < cls.HBM_BOUND_GPU_UTIL
                and metrics.comp_kernel_avg_dur_us < cls.HBM_BOUND_AVG_KERNEL_US):
            issues.append(f"HBM bandwidth-bound "
                          f"(comp kernels avg {metrics.comp_kernel_avg_dur_us:.1f}us, "
                          f"util {metrics.gpu_util:.1%})")

        # 9. Gradient accumulation / injection bandwidth pressure
        # Reduce-scatter is not hidden behind backward compute → RS GPU time
        # dominates its wall span.  High RS overlap efficiency means RS is
        # exposed on the critical path rather than overlapping with bwd compute.
        if metrics.rs_overlap_efficiency >= cls.RS_EXPOSED_THRESHOLD:
            issues.append(f"RS injection pressure "
                          f"(RS overlap efficiency={metrics.rs_overlap_efficiency:.1%})")

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

        # 12. No comm/compute overlap (FSDP2 pipeline idle)
        # AG fwd overlap efficiency close to 1.0 means the all-gather runs
        # entirely serially with no overlap → classic FSDP2 pipeline bubble.
        if metrics.ag_fwd_overlap_efficiency > 0 and metrics.ag_fwd_overlap_efficiency >= 0.85:
            issues.append(f"no comm/compute overlap "
                          f"(AG overlap efficiency={metrics.ag_fwd_overlap_efficiency:.1%})")

        # 13. Exposed communication (general overlap quality)
        # Composite metric across all comm phases.  >50% exposed → poor hiding.
        if metrics.exposed_comm_fraction > 0.50:
            issues.append(f"exposed communication "
                          f"(comm hiding efficiency={1-metrics.exposed_comm_fraction:.1%})")

        return issues


class Report:
    """Aggregates per-layer metrics into a human-readable text report and JSON markers.
    Builds Metrics objects, computes overlap/pipeline decomposition, formats the report,
    and serialises per-unit data + bottlenecks.
    """

    def __init__(self, fsdp: FSDP, root_nodes: List[LogicalOperation],
                 output_path: Optional[str] = None,
                 model_config: Optional[ModelConfig] = None):
        self.fsdp = fsdp
        self.root_nodes = root_nodes
        self.output_path = output_path
        self.model_config = model_config
        self.metrics_list: List[Metrics] = []
        self.aggregated: Dict[str, float] = {}
        self.throughput_metrics: Dict[str, float] = {}

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
        for unit in self.fsdp.units:
            self.metrics_list.append(Metrics(unit, opt_gpu, opt_cpu, num_units,
                                             tp_ag, tp_rs, tp_ar, tp_kernels=tp_kernels))

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
                "rs_gpu_us", "optimizer_gpu_us", "tp_ag_gpu_us", "tp_rs_gpu_us",
                "tp_ar_gpu_us", "tp_total_gpu_us",
                "total_gpu_us", "total_cpu_us",
                "comm_ratio", "comp_ratio", "optimizer_ratio",
                "overlap_ratio", "serial_exec_efficiency", "idle_ratio"]
        self.aggregated = {k: 0.0 for k in keys}
        count = len(self.metrics_list)
        if count == 0:
            return

        # Compute overlap metrics from actual unit spans
        ov = _compute_overlap_metrics(self.fsdp.units)
        self.overlap_metrics = ov

        for m in self.metrics_list:
            d = m.to_dict()
            for k in keys:
                self.aggregated[k] += d.get(k, 0.0)
            # Set overlap fields on each Metrics instance
            m.overlap_ratio = ov['overlap_ratio']
            m.serial_exec_efficiency = ov['serial_exec_efficiency']
            m.idle_ratio = ov['idle_ratio']
            m.step_wall = ov['step_wall']

        for k in keys:
            self.aggregated[k] /= count
        self.aggregated["num_units"] = count
        self.aggregated["overlap_ratio"] = ov['overlap_ratio']
        self.aggregated["serial_exec_efficiency"] = ov['serial_exec_efficiency']
        self.aggregated["idle_ratio"] = ov['idle_ratio']
        self.aggregated["step_wall"] = ov['step_wall']
        self.aggregated["overlap_time"] = ov['overlap_time']
        self.aggregated["serial_time"] = ov['serial_time']

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

        for m in self.metrics_list:
            m.compute_throughput_metrics(cfg)

        tps = sum(getattr(m, 'tokens_per_second_per_gpu', 0.0) for m in self.metrics_list)
        avg_tps = tps / len(self.metrics_list) if self.metrics_list else 0.0
        avg_mfu = sum(getattr(m, 'mfu', 0.0) for m in self.metrics_list) / len(self.metrics_list) if self.metrics_list else 0.0
        avg_hfu = sum(getattr(m, 'hfu', 0.0) for m in self.metrics_list) / len(self.metrics_list) if self.metrics_list else 0.0
        flops = self.metrics_list[0].estimated_flops_per_step if self.metrics_list else 0.0

        self.throughput_metrics = {
            'tokens_per_second_per_gpu': avg_tps,
            'mfu': avg_mfu,
            'hfu': avg_hfu,
            'estimated_flops_per_step': flops,
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

        # Step Summary
        lines.append("--- Step Summary ---")
        lines.append(f"  Number of layers:     {int(num_units)}")
        lines.append(f"  Step wall time:       {_format_us(step_wall)}")
        if step_wall > 0:
            lines.append(f"  Estimated throughput: {1_000_000 / step_wall:.1f} steps/s")
        lines.append("")

        # Aggregated phase metrics
        lines.append("--- Phase Metrics ---")
        phase_keys = [
            ("ag_fwd_gpu_us", "All-gather fwd"),
            ("fwd_cmp_gpu_us", "Fwd compute"),
            ("tp_ag_gpu_us", "  TP all-gather"),
            ("tp_ar_gpu_us", "  TP all-reduce"),
            ("ag_bwd_gpu_us", "All-gather bwd"),
            ("bwd_cmp_gpu_us", "Bwd compute"),
            ("rs_gpu_us", "Reduce-scatter"),
            ("tp_rs_gpu_us", "  TP reduce-scatter"),
            ("optimizer_gpu_us", "Optimizer step"),
        ]
        total_gpu = self.aggregated.get("total_gpu_us", 0)
        lines.append(f"  {'Phase':25s} {'Per unit':>10s} {'Total':>10s} {'% GPU':>8s}")
        lines.append(f"  {'-----':25s} {'--------':>10s} {'-----':>10s} {'-----':>8s}")
        for key, label in phase_keys:
            per_unit = self.aggregated.get(key, 0)
            total = per_unit * num_units
            pct = per_unit / total_gpu * 100 if total_gpu > 0 else 0
            lines.append(f"  {label:25s} {_format_us(per_unit):>10s} {_format_us(total):>10s} {pct:>7.1f}%")
        lines.append(f"  {'-----':25s} {'--------':>10s} {'-----':>10s} {'-----':>8s}")
        tp_per_unit = self.aggregated.get("tp_total_gpu_us", 0)
        lines.append(f"  {'FSDP total':25s} {_format_us(total_gpu):>10s} {_format_us(total_gpu * num_units):>10s}")
        lines.append(f"  {'TP total':25s} {_format_us(tp_per_unit):>10s} {_format_us(tp_per_unit * num_units):>10s}")
        lines.append(f"  {'Total (incl. TP)':25s} {_format_us(total_gpu + tp_per_unit):>10s} {_format_us((total_gpu + tp_per_unit) * num_units):>10s} {100.0:>7.1f}%")
        lines.append(f"  {'Total CPU':25s} {_format_us(self.aggregated.get('total_cpu_us', 0)):>10s}")
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
        fsdp_comm = self.aggregated.get("ag_fwd_gpu_us", 0) + self.aggregated.get("ag_bwd_gpu_us", 0) + self.aggregated.get("rs_gpu_us", 0)
        opt_ratio = self.aggregated.get('optimizer_ratio', 0)
        true_comp = comp_gpu - tp_total
        total = total_gpu + tp_total
        lines.append(f"  True compute:         {true_comp / total:.1%} (ex-TP)")
        lines.append(f"  TP communication:     {tp_total / total:.1%}")
        lines.append(f"  FSDP communication:   {fsdp_comm / total:.1%}")
        lines.append(f"  Optimizer:            {opt_ratio:.1%}")
        avg_util = sum(m.gpu_util for m in self.metrics_list) / len(self.metrics_list) if self.metrics_list else 0.0
        lines.append(f"  Avg GPU utilization:  {avg_util:.1%} (per-layer GPU busy / layer span)")
        max_span = max(m.layer_span for m in self.metrics_list) if self.metrics_list else 0.0
        min_span = min(m.layer_span for m in self.metrics_list if m.layer_span > 0) or max_span
        lines.append(f"  Layer span imbalance: {max_span / min_span:.1f}x (max/min layer span ratio)")

        # Compute-to-communicate ratio
        avg_ctc = sum(m.compute_to_comm_ratio for m in self.metrics_list) / len(self.metrics_list) if self.metrics_list else 0.0
        lines.append(f"  Compute-to-comm:      {avg_ctc:.2f}× (compute GPU / comm GPU)")
        lines.append("")

        # Per-phase overlap efficiency
        lines.append("--- Comm Overlap Efficiency ---")
        lines.append(f"  {'Phase':25s} {'Avg eff':>10s} {'Min':>10s} {'Max':>10s} {'Interpretation':>35s}")
        lines.append(f"  {'-----':25s} {'--------':>10s} {'---':>10s} {'---':>10s} {'---------------':>35s}")
        for label, attr in [("AG fwd", "ag_fwd_overlap_efficiency"),
                            ("AG bwd", "ag_bwd_overlap_efficiency"),
                            ("RS", "rs_overlap_efficiency")]:
            vals = [getattr(m, attr) for m in self.metrics_list]
            avg = sum(vals) / len(vals) if vals else 0.0
            mn = min(vals) if vals else 0.0
            mx = max(vals) if vals else 0.0
            interp = "exposed" if avg >= 0.7 else "hidden" if avg < 0.3 else "partial"
            lines.append(f"  {label:25s} {avg:>9.1%} {mn:>9.1%} {mx:>9.1%}  {interp:>35s}")
        exp = sum(m.exposed_comm_fraction for m in self.metrics_list) / len(self.metrics_list) if self.metrics_list else 0.0
        lines.append(f"  {'Exposed comm (avg)':25s} {exp:>9.1%}")
        lines.append("")

        # Overlap / Serial Execution Efficiency
        lines.append("--- Overlap & Pipeline ---")
        ov = self.overlap_metrics
        lines.append(f"  Overlap time:         {_format_us(ov['overlap_time'])} ({ov['overlap_ratio']:.1%} of non-idle)")
        lines.append(f"  Serial execution:     {_format_us(ov['serial_time'])} ({ov['serial_exec_efficiency']:.1%} of step)")
        lines.append(f"  Idle/Gap time:        {_format_us(ov['idle_time'])} ({ov['idle_ratio']:.1%} of step)")
        lines.append(f"  Communication ratio:  {fsdp_comm / total:.1%} FSDP + {tp_total / total:.1%} TP = {self.aggregated.get('comm_ratio', 0):.1%} total")
        avg_fwd_ov = sum(m.fwd_comp_comm_overlap for m in self.metrics_list) / max(len(self.metrics_list), 1)
        avg_bwd_ov = sum(m.bwd_comp_comm_overlap for m in self.metrics_list) / max(len(self.metrics_list), 1)
        lines.append(f"  Avg Fwd comp-comm:    {avg_fwd_ov:.1%} (avg TP overlap during fwd compute)")
        lines.append(f"  Avg Bwd comp-comm:    {avg_bwd_ov:.1%} (avg TP overlap during bwd compute)")
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
                  f"{'Util':>6s}{'CtC':>6s}{'ExpC':>6s}"
                  f"{'Span':>9s}{'F-Ovl':>8s}{'B-Ovl':>8s} "
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
            exp_comm = d.get('exposed_comm_fraction', 0)
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
                f"{d['gpu_util']:>5.1%} "
                f"{ctc_str:>5s} "
                f"{exp_comm:>5.1%} "
                f"{_format_us(d['layer_span_us']):>9s}"
                f"{d['fwd_comp_comm_overlap']:>7.1%} "
                f"{d['bwd_comp_comm_overlap']:>7.1%} "
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
