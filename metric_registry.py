"""METRIC_REGISTRY: documentation for every metric in Metrics.to_dict().
THRESHOLD_REGISTRY: documentation for every threshold in Bottlenecks.

Each entry documents one metric key with:
- description: plain-English explanation
- used_by: list of bottleneck short names whose detection logic reads this metric
- unit: µs, ratio, count, bytes, FLOPs, tokens/s/GPU
- calculation: how the metric is derived from raw data

Each threshold entry documents one threshold constant with:
- value: the numeric threshold
- rationale: plain-English explanation of what it detects
- physical_justification: the physical / architectural model it derives from
- used_by: list of bottleneck short names that use this threshold

This file is imported by html_report.py to render the collapsible metric
registry table and threshold documentation. It lives in its own module so
that CLI tools, comparison reports, and other consumers can reference it
without importing the full detection pipeline.
"""

METRIC_REGISTRY = {
    # Phase GPU times (µs)
    "ag_fwd_gpu_us": {
        "description": "All-gather forward GPU time — NCCL kernel duration for the forward all-gather",
        "used_by": ["all-gather-heavy", "reduce-scatter-heavy", "fwd-bwd imbalance", "exposed all-gather", "no comm/compute overlap", "dominant phase"],
        "unit": "µs",
        "calculation": "_phase_gpu_time(unit.all_gather_fwd) — sum of NCCL kernel durations in all-gather forward",
    },
    "fwd_cmp_gpu_us": {
        "description": "Forward compute GPU time — attention, MLP, and layer norm kernels (direct, non-overlapping)",
        "used_by": ["exposed all-gather", "fwd-bwd imbalance", "synchronous TP on critical path", "no comm/compute overlap", "exposed communication"],
        "unit": "µs",
        "calculation": "_phase_gpu_time_direct(unit.fwd_compute) — sum of direct GPU kernel durations in forward compute",
    },
    "ag_bwd_gpu_us": {
        "description": "All-gather backward GPU time — NCCL kernel duration for the backward all-gather (ancestry-based _compute_ag_per_layer + ac2g supplement)",
        "used_by": ["all-gather-heavy", "reduce-scatter-heavy", "fwd-bwd imbalance", "dominant phase"],
        "unit": "µs",
        "calculation": "_compute_ag_per_layer[layer] (bwd) + unit.ag_bwd_supplement_us — ancestry-based AG bwd + ac2g supplement",
    },
    "bwd_cmp_gpu_us": {
        "description": "Backward compute GPU time — gradient computation kernels (direct, non-overlapping)",
        "used_by": ["fwd-bwd imbalance", "RS exceeds bwd compute", "exposed communication"],
        "unit": "µs",
        "calculation": "_phase_gpu_time_direct(unit.bwd_compute) — sum of direct GPU kernel durations in backward compute",
    },
    "rs_gpu_us": {
        "description": "Reduce-scatter GPU time — NCCL ReduceScatter kernel duration only (excludes AllReduce kernels in the post_backward_reduce subtree)",
        "used_by": ["all-gather-heavy", "reduce-scatter-heavy", "fwd-bwd imbalance", "RS exceeds bwd compute", "dominant phase"],
        "unit": "µs",
        "calculation": "_phase_gpu_time_breakdown(unit.reduce_scatter) — sum of NCCL ReduceScatter kernel durations in post_backward_reduce subtree, deduplicated globally",
    },
    "ar_in_rs_gpu_us": {
        "description": "All-reduce GPU time in the reduce-scatter subtree — NCCL AllReduce kernels under post_backward_reduce (embedding gradient sync, tied weights)",
        "used_by": ["AR in RS on critical path"],
        "unit": "µs",
        "calculation": "_phase_gpu_time_breakdown(unit.reduce_scatter) — sum of NCCL AllReduce kernel durations in post_backward_reduce subtree, deduplicated globally",
    },
    "optimizer_gpu_us": {
        "description": "Optimizer GPU time — evenly split across layers from the global optimizer step (ADAMW parameter update)",
        "used_by": ["optimizer-heavy", "dominant phase"],
        "unit": "µs",
        "calculation": "global_optimizer_gpu / num_units — optimizer GPU time evenly split across layers",
    },
    "tp_ag_gpu_us": {
        "description": "Tensor-parallel all-gather GPU time",
        "used_by": ["TP-heavy", "dominant phase"],
        "unit": "µs",
        "calculation": "global_tp_ag_gpu / num_units — TP all-gather evenly split across layers",
    },
    "tp_rs_gpu_us": {
        "description": "Tensor-parallel reduce-scatter GPU time",
        "used_by": ["TP-heavy", "dominant phase"],
        "unit": "µs",
        "calculation": "global_tp_rs_gpu / num_units — TP reduce-scatter evenly split across layers",
    },
    "tp_ar_gpu_us": {
        "description": "Tensor-parallel all-reduce GPU time",
        "used_by": ["TP-heavy", "dominant phase"],
        "unit": "µs",
        "calculation": "global_tp_ar_gpu / num_units — TP all-reduce evenly split across layers",
    },
    "tp_total_gpu_us": {
        "description": "Total tensor-parallel GPU time (ag + rs + ar)",
        "used_by": ["TP-heavy", "synchronous TP on critical path", "NVLink saturation", "dominant phase"],
        "unit": "µs",
        "calculation": "tp_ag_gpu + tp_rs_gpu + tp_ar_gpu — sum of TP all-gather, reduce-scatter, all-reduce",
    },
    # Totals and ratios
    "total_gpu_us": {
        "description": "Total GPU time across all FSDP phases (compute + comm + optimizer)",
        "used_by": ["TP-heavy", "dominant phase"],
        "unit": "µs",
        "calculation": "ag_fwd_gpu + fwd_cmp_gpu + ag_bwd_gpu + bwd_cmp_gpu + rs_gpu + optimizer_gpu — sum of all FSDP phase GPU times",
    },
    "total_cpu_us": {
        "description": "Total CPU dispatch time across all phases",
        "used_by": [],
        "unit": "µs",
        "calculation": "ag_fwd_cpu + fwd_cmp_cpu + ag_bwd_cpu + bwd_cmp_cpu + rs_cpu + optimizer_cpu — sum of all phase CPU dispatch times",
    },
    "comm_ratio": {
        "description": "Fraction of total GPU time spent on communication (FSDP NCCL + TP collectives)",
        "used_by": ["comm-bound"],
        "unit": "ratio",
        "calculation": "(FSDP_comm + TP_comm) / (total_gpu + tp_total_gpu) — communication / total GPU",
    },
    "comp_ratio": {
        "description": "Fraction of total GPU time spent on compute (fwd_cmp + bwd_cmp)",
        "used_by": ["HBM bandwidth-bound"],
        "unit": "ratio",
        "calculation": "(fwd_cmp_gpu + bwd_cmp_gpu) / total_gpu — compute / total GPU",
    },
    "optimizer_ratio": {
        "description": "Fraction of total GPU time spent on the optimizer step",
        "used_by": ["optimizer-heavy"],
        "unit": "ratio",
        "calculation": "optimizer_gpu / total_gpu — optimizer / total GPU",
    },
    "gpu_busy": {
        "description": "Fraction of the layer's wall span where at least one GPU kernel was executing (union of kernel intervals, capped at 1.0)",
        "used_by": ["low GPU utilization", "HBM bandwidth-bound", "no comm/compute overlap", "exposed communication"],
        "unit": "ratio",
        "calculation": "merged_gpu_kernel_intervals / max(layer_span, gpu_kernel_span) — union of GPU kernel intervals / denominator",
    },
    "fwd_busy": {
        "description": "GPU utilization during the forward pass only",
        "used_by": [],
        "unit": "ratio",
        "calculation": "merged_fwd_kernel_intervals / max(fwd_span, fwd_gpu_kernel_span) — GPU utilization during forward",
    },
    "bwd_busy": {
        "description": "GPU utilization during the backward pass only",
        "used_by": [],
        "unit": "ratio",
        "calculation": "merged_bwd_kernel_intervals / max(bwd_span, bwd_gpu_kernel_span) — GPU utilization during backward",
    },
    "layer_span_us": {
        "description": "Sum of forward and backward CPU wall spans (not union — excludes the pipeline gap between fwd and bwd)",
        "used_by": [],
        "unit": "µs",
        "calculation": "fwd_span + bwd_span — sum of forward and backward CPU wall spans",
    },
    "overlap_ratio": {
        "description": "Fraction of non-idle time where multiple FSDP layers overlapped on GPU (sweep-line decomposition)",
        "used_by": [],
        "unit": "ratio",
        "calculation": "overlap_time / (serial_time + overlap_time) — fraction of non-idle time with multiple layers active",
    },
    "serial_ratio": {
        "description": "Fraction of total step wall time where only one FSDP layer was active on GPU",
        "used_by": ["serial pipeline"],
        "unit": "ratio",
        "calculation": "serial_time / total_time — fraction of step with only one layer active",
    },
    "idle_ratio": {
        "description": "Fraction of total step wall time with no GPU activity from any FSDP layer — pipeline bubbles or data-loading stalls",
        "used_by": ["I/O or pipeline bubble"],
        "unit": "ratio",
        "calculation": "idle_time / total_time — fraction of step with zero layers active",
    },
    # Memory
    "memory_peak": {
        "description": "Peak GPU memory usage observed during this layer (needs profile_memory=True)",
        "used_by": [],
        "unit": "bytes",
        "calculation": "max(running memory counter after chronological memory deltas) — peak GPU memory usage",
    },
    "memory_allocated": {
        "description": "Total GPU memory allocated during this layer (needs profile_memory=True)",
        "used_by": [],
        "unit": "bytes",
        "calculation": "sum(positive memory delta values) — total GPU memory allocated",
    },
    "memory_freed": {
        "description": "Total GPU memory freed during this layer (needs profile_memory=True)",
        "used_by": [],
        "unit": "bytes",
        "calculation": "sum(|negative memory delta|) — total GPU memory freed (absolute)",
    },
    # Communication breakdown
    "fsdp_comm_ratio": {
        "description": "Fraction of total GPU time spent on FSDP communication (AG fwd + AG bwd + RS)",
        "used_by": [],
        "unit": "ratio",
        "calculation": "(ag_fwd_gpu + ag_bwd_gpu + rs_gpu) / (total_gpu + tp_total_gpu) — FSDP comm / total GPU",
    },
    "tp_comm_ratio": {
        "description": "Fraction of total GPU time spent on TP communication",
        "used_by": [],
        "unit": "ratio",
        "calculation": "tp_total_gpu / (total_gpu + tp_total_gpu) — TP comm / total GPU",
    },
    # Async TP overlap
    "fwd_comp_comm_overlap": {
        "description": "Fraction of forward compute wall time where TP kernels (all-gather/reduce-scatter) were executing concurrently — measures async TP overlap quality during forward",
        "used_by": ["low async TP overlap", "async TP asymmetry", "synchronous TP on critical path"],
        "unit": "ratio",
        "calculation": "TP kernel duration during fwd / fwd_wall_time — fraction of forward wall time with TP kernels",
    },
    "bwd_comp_comm_overlap": {
        "description": "Fraction of backward compute wall time where TP kernels were executing concurrently",
        "used_by": ["low async TP overlap", "async TP asymmetry"],
        "unit": "ratio",
        "calculation": "TP kernel duration during bwd / bwd_wall_time — fraction of backward wall time with TP kernels",
    },
    "pipeline_overlap_ratio": {
        "description": "GPU-span overlap ratio between adjacent layers — measures cross-layer GPU pipeline parallelism",
        "used_by": ["low cross-layer GPU overlap"],
        "unit": "ratio",
        "calculation": "cross-layer GPU span overlap ratio per adjacent pair — measures GPU pipeline parallelism",
    },
    # Kernel statistics
    "kernel_count": {
        "description": "Total number of GPU kernels across all phases for this layer",
        "used_by": ["small-kernel-bound"],
        "unit": "count",
        "calculation": "len(collected_gpu_kernels across all phases) — total GPU kernel count",
    },
    "avg_kernel_dur_us": {
        "description": "Average duration of all GPU kernels in this layer",
        "used_by": ["small-kernel-bound"],
        "unit": "µs",
        "calculation": "sum(kernel_durations) / kernel_count — average GPU kernel duration",
    },
    "nccl_comm_gpu_us": {
        "description": "Total NCCL kernel time inside all-gather phases",
        "used_by": ["copy-heavy all-gather"],
        "unit": "µs",
        "calculation": "sum(NCCL kernel durations inside all-gather nodes) — total NCCL time in AG phases",
    },
    "copy_data_movement_gpu_us": {
        "description": "Total data-movement (memcpy) kernel time inside all-gather phases (split_with_sizes_copy, aten::copy_)",
        "used_by": ["copy-heavy all-gather"],
        "unit": "µs",
        "calculation": "sum(copy kernel durations inside all-gather nodes) — total copy time in AG phases",
    },
    "cpu_wall_us": {
        "description": "CPU wall span from first to last event across all phases for this layer",
        "used_by": [],
        "unit": "µs",
        "calculation": "max(end_time) - min(start_time) across all phase events — CPU wall span",
    },
    "cpu_wall_to_gpu_ratio": {
        "description": "Ratio of CPU wall span to total GPU time — values >1.5x indicate host-boundness (CPU serialization or dispatch overhead)",
        "used_by": ["host-bound"],
        "unit": "ratio",
        "calculation": "cpu_wall_us / total_gpu — CPU wall / total GPU time",
    },
    # Universal exposure / efficiency metrics
    "compute_to_comm_ratio": {
        "description": "Arithmetic intensity proxy: compute GPU time ÷ all comm GPU time. Higher = more compute-heavy; <1.0 = comm-dominated",
        "used_by": [],
        "unit": "ratio",
        "calculation": "(fwd_cmp_gpu + bwd_cmp_gpu) / (FSDP_comm + TP_comm) — compute / all communication",
    },
    "ag_fwd_exposed_ratio": {
        "description": "All-gather forward GPU time ÷ its CPU wall span. Near 1.0 = fully exposed (no overlap with other work); low = well hidden",
        "used_by": [],
        "unit": "ratio",
        "calculation": "min(1.0, ag_fwd_gpu / ag_fwd_wall) — AG fwd GPU / AG fwd wall span",
    },
    "rs_exposed_ratio": {
        "description": "Reduce-scatter GPU time ÷ its CPU wall span",
        "used_by": [],
        "unit": "ratio",
        "calculation": "min(1.0, rs_gpu / rs_wall) — RS GPU / RS wall span",
    },
    "ag_bwd_exposed_ratio": {
        "description": "All-gather backward GPU time ÷ its CPU wall span",
        "used_by": [],
        "unit": "ratio",
        "calculation": "min(1.0, ag_bwd_gpu / ag_bwd_wall) — AG bwd GPU / AG bwd wall span",
    },
    "avg_exposed_ratio": {
        "description": "Average of ag_fwd_exposed_ratio, rs_exposed_ratio, and ag_bwd_exposed_ratio (only counting phases with nonzero GPU time)",
        "used_by": [],
        "unit": "ratio",
        "calculation": "mean of nonzero ag_fwd, rs, and ag_bwd exposed ratios — average collection exposure",
    },
    # Compute-phase kernel classification
    "comp_kernel_count": {
        "description": "Number of GPU kernels in compute phases (fwd + bwd) only — excludes NCCL/copy inside AG phases",
        "used_by": [],
        "unit": "count",
        "calculation": "len(kernels in fwd and bwd compute phases) — GPU kernels in compute phases only",
    },
    "comp_kernel_avg_dur_us": {
        "description": "Average duration of compute-phase kernels — short averages (<5µs) suggest launch-latency or HBM-bandwidth bound",
        "used_by": ["HBM bandwidth-bound", "NVLink saturation"],
        "unit": "µs",
        "calculation": "sum(comp_kernel_durs) / comp_kernel_count — avg duration of compute-phase kernels",
    },
    "nccl_in_comp_count": {
        "description": "Number of NCCL kernels found inside compute phases (e.g. fused RS in bwd)",
        "used_by": ["NVLink saturation"],
        "unit": "count",
        "calculation": "count of NCCL kernels found inside compute phases — NCCL inside compute",
    },
    "copy_in_comp_count": {
        "description": "Number of copy/memcpy kernels found inside compute phases",
        "used_by": [],
        "unit": "count",
        "calculation": "count of copy kernels found inside compute phases — copy inside compute",
    },
    # Throughput (populated by compute_throughput_metrics)
    "tokens_per_second_per_gpu": {
        "description": "Throughput in tokens processed per second per GPU (needs ModelConfig)",
        "used_by": [],
        "unit": "tokens/s/GPU",
        "calculation": "tokens_per_step / step_wall_s / num_gpus — tokens/s/GPU throughput",
    },
    "mfu": {
        "description": "Model FLOPs Utilization — achieved FLOPs as fraction of peak theoretical FLOPs (needs ModelConfig)",
        "used_by": [],
        "unit": "ratio",
        "calculation": "observed_flops / (gpu_peak_flops * num_gpus) — Model FLOPs Utilization",
    },
    "hfu": {
        "description": "Hardware FLOPs Utilization — MFU adjusted for activation checkpointing overhead (needs ModelConfig)",
        "used_by": [],
        "unit": "ratio",
        "calculation": "MFU * 3 / (2 + activation_checkpointing_fraction) — Hardware FLOPs Utilization",
    },
    "estimated_flops_per_step": {
        "description": "Estimated FLOPs per training step based on ModelConfig parameters (needs ModelConfig)",
        "used_by": [],
        "unit": "FLOPs",
        "calculation": "Kaplan et al. FLOPs formula from ModelConfig hyperparameters — estimated FLOPs per step",
    },
}


# =========================================================================
# THRESHOLD_REGISTRY — physical justification for every Bottlenecks threshold
# =========================================================================
# Each entry documents one threshold constant with:
#   value                  — numeric threshold value
#   rationale              — what this threshold detects / signals
#   physical_justification — the physical or architectural model that
#                            motivates the number (roofline, NCCL BW model,
#                            Amdahl's law, GPU spec, empirical measurement)
#   used_by                — list of bottleneck short names that check this
# =========================================================================

THRESHOLD_REGISTRY = {
    # --- Section A — Classical bottleneck thresholds ---
    "COMP_HEAVY_THRESHOLD": {
        "value": 0.70,
        "rationale": "Fraction of total GPU time spent on compute (fwd + bwd) above which the bottleneck is compute-bound rather than communication-bound",
        "physical_justification": "Amdahl's law: if ≥70% of GPU time is compute, further communication optimisation yields at most 1/(1−0.70)≈1.43× speedup. The bottleneck has shifted to compute.",
        "used_by": ["HBM bandwidth-bound"],
    },
    "COMM_HEAVY_THRESHOLD": {
        "value": 0.40,
        "rationale": "Fraction of total GPU time spent on communication above which the step is communication-bound",
        "physical_justification": "NCCL bandwidth model on H100 NVLink (≈900 GB/s aggregate): large-message all-reduce BW plateaus at ≈60–70 GB/s. When comm ≥40% of GPU time, the all-reduce is in the bandwidth-limited regime; doubling network BW gives <2× speedup (Amdahl).",
        "used_by": ["comm-bound"],
    },
    "IO_HEAVY_THRESHOLD": {
        "value": 0.15,
        "rationale": "Fraction of wall time with no active GPU work (idle / pipeline bubble) above which I/O or bubbles limit scaling",
        "physical_justification": "Amdahl's law: with 15% serial/idle fraction, maximum achievable speedup is 1/0.15≈6.7× regardless of GPU count. Beyond this, I/O or pipeline bubbles materially limit strong scaling.",
        "used_by": ["I/O or pipeline bubble"],
    },
    "AG_HEAVY_THRESHOLD": {
        "value": 0.40,
        "rationale": "Fraction of FSDP communication time spent on all-gather (fwd + bwd) above which AG dominates the comm profile",
        "physical_justification": "NCCL all-gather BW model: same bandwidth ceiling as all-reduce on NVLink (≈60–70 GB/s large-message). When AG ≥40% of FSDP comm, unshard dominates the communication profile.",
        "used_by": ["all-gather-heavy"],
    },
    "RS_HEAVY_THRESHOLD": {
        "value": 0.40,
        "rationale": "Fraction of FSDP communication time spent on reduce-scatter above which RS dominates the comm profile",
        "physical_justification": "NCCL reduce-scatter BW model: RS has identical BW characteristics to all-gather (algorithmic equivalents in NCCL ring/tree). ≥40% indicates gradient synchronisation dominates FSDP comm.",
        "used_by": ["reduce-scatter-heavy"],
    },
    "TP_HEAVY_THRESHOLD": {
        "value": 0.15,
        "rationale": "Fraction of total GPU time spent on tensor-parallel collectives above which TP is a bottleneck",
        "physical_justification": "NVLink BW model: TP collectives run intra-node over NVLink at ≈900 GB/s (H100). Lower threshold because TP (unlike FSDP) is synchronous on the critical path and cannot be overlapped across layers. Even 15% GPU time in TP materially limits scaling efficiency.",
        "used_by": ["TP-heavy"],
    },
    "OPTIMIZER_HEAVY_THRESHOLD": {
        "value": 0.15,
        "rationale": "Fraction of total GPU time spent on the ADAMW optimizer step above which the optimizer dominates",
        "physical_justification": "HBM memory-bandwidth model: ADAMW is memory-bandwidth-bound (reads g + m, writes m + params). On H100 HBM3 (3.35 TB/s), 15% corresponds to the point where optimizer memory traffic becomes a dominant fraction of the HBM budget.",
        "used_by": ["optimizer-heavy"],
    },
    "UTIL_LOW_THRESHOLD": {
        "value": 0.50,
        "rationale": "Minimum GPU utilisation below which the layer is flagged as low-utilisation",
        "physical_justification": "Standard industry heuristic (cf. NVIDIA GPU utilisation guidance): below 50% the GPU is idle more than it is computing — the pipeline delivers less than half its achievable throughput.",
        "used_by": ["low GPU utilization"],
    },

    # --- Section B — FSDP2 / TP / async TP specific thresholds ---
    "SMALL_KERNEL_AVG_DUR_US": {
        "value": 5.0,
        "rationale": "Average GPU kernel duration (µs) below which kernels are small enough to be launch-latency-bound",
        "physical_justification": "CUDA launch-latency model: kernel launch latency is ≈3–8 µs (driver + scheduler overhead). When avg kernel duration ≤5 µs, execution is comparable to launch latency → launch-latency-bound. NVIDIA guidance recommends kernels >10 µs for efficient utilisation.",
        "used_by": ["small-kernel-bound"],
    },
    "SMALL_KERNEL_COUNT": {
        "value": 100,
        "rationale": "Minimum number of small kernels to trigger small-kernel-bound classification",
        "physical_justification": "Empirical — transformer layer composition: a typical layer has ≈50–80 CUDA kernels (attention + MLP + norms). When count ≥100 with avg <5 µs, the decomposition is fine-grained enough for launch overhead to dominate.",
        "used_by": ["small-kernel-bound"],
    },
    "ASYNC_TP_OVERLAP_LOW": {
        "value": 0.30,
        "rationale": "Fraction of forward (or backward) wall time with concurrent TP kernel execution below which async TP overlap is considered poor",
        "physical_justification": "Async TP pipeline model: in an ideal async TP implementation, collectives overlap with compute for >70% of forward wall time. Below 30% the overlap mechanism is present but hides less than a third of TP latency.",
        "used_by": ["low async TP overlap"],
    },
    "OVERLAP_ASYMMETRY_DIFF": {
        "value": 0.40,
        "rationale": "Absolute difference (in percentage points) between fwd and bwd TP overlap above which the asymmetry is flagged",
        "physical_justification": "Pipeline symmetry: |fwd−bwd| ≥40 pp indicates the pipeline stagger is tuned for one direction at the expense of the other. Forward has one AG; backward has AG+RS — inherently asymmetric; large asymmetry means poor stagger balance.",
        "used_by": ["async TP asymmetry"],
    },
    "HOST_BOUND_RATIO": {
        "value": 3.0,
        "rationale": "Ratio of CPU wall span to total GPU time above which CPU dispatch overhead is considered a bottleneck",
        "physical_justification": "CPU dispatch overhead model: for efficient GPU utilisation, CPU dispatch overhead should be a small fraction of GPU time. A ratio of 3.0 means 1 µs of GPU work costs 3 µs of CPU serialisation — Python runtime / GIL / PyTorch dispatch overhead. Well-optimised FSDP2 achieves CPU:GPU ≤1.5×.",
        "used_by": ["host-bound"],
    },
    "COPY_HEAVY_RATIO": {
        "value": 0.50,
        "rationale": "Fraction of all-gather phase GPU time spent on memcpy (vs NCCL) above which CPU-side memory copy dominates the AG",
        "physical_justification": "CPU memory bandwidth model (DDR5): in FSDP2 AG, non-NCCL time is dominated by split_with_sizes_copy (CPU-side memory copies for tensor slicing). When ≥50% of AG time is copy, CPU data-preparation (DDR5 ≈50–80 GB/s) bottlenecks the AG faster than NCCL (NVLink ≈450 GB/s unidirectional).",
        "used_by": ["copy-heavy all-gather"],
    },
    "FWD_BWD_IMBALANCE": {
        "value": 0.75,
        "rationale": "Normalised |fwd−bwd| / max(fwd,bwd) above which the forward/backward GPU time split is considered imbalanced",
        "physical_justification": "Theoretical fwd:bwd FLOPs ratio: standard transformer layer with activation checkpointing has fwd:bwd ≈ 1:2. Deviation >25% (i.e. imbalance ≥0.75) signals intervention from act-ckpt or custom gradient computation.",
        "used_by": ["fwd-bwd imbalance"],
    },
    "SERIAL_RATIO_HIGH": {
        "value": 0.85,
        "rationale": "Fraction of total wall time with only one FSDP shard group active above which the pipeline is considered serial",
        "physical_justification": "FSDP2 pipeline model: with N pipeline stages, theoretical max overlap = (N−1)/N. Even a 2-deep pipeline should achieve 50% overlap. Serial >85% means <15% overlap — the stagger is defeated by synchronisation or insufficient in-flight layers.",
        "used_by": ["serial pipeline"],
    },

    # --- Section C — Communication-hiding / BW bottleneck thresholds ---
    "AG_LATENCY_EXPOSED_RATIO": {
        "value": 0.80,
        "rationale": "Ratio of all-gather fwd GPU time to fwd compute GPU time above which AG is considered exposed (inter-node BW-limited)",
        "physical_justification": "NCCL AG BW model (inter-node): on NVLink (≈450 GB/s unidirectional), AG fwd GPU ≥80% of fwd compute GPU means the GPU stalls waiting for AG. At this ratio, doubling AG BW would reduce step time by ≤20% (Amdahl: serial fraction = 0.80/1.80≈0.44; max speedup = 1/(1−0.44+0.44/2)≈1.22×).",
        "used_by": ["exposed all-gather"],
    },
    "HBM_BOUND_AVG_KERNEL_US": {
        "value": 8.0,
        "rationale": "Average compute-phase kernel duration below which kernels are short enough to suggest HBM bandwidth binding",
        "physical_justification": "Roofline model (H100 HBM3): ridge point at 295 FLOPs/byte (bf16). A kernel of 8 µs at typical grid size (≈256 blocks = 2 waves on 128 SMs) has arithmetic intensity well below the ridge point, placing it in the memory-bandwidth-bound regime.",
        "used_by": ["HBM bandwidth-bound"],
    },
    "HBM_BOUND_GPU_UTIL": {
        "value": 0.50,
        "rationale": "GPU utilisation below which (combined with short compute kernels) HBM bandwidth binding is flagged",
        "physical_justification": "Utilisation cross-validation: GPU utilisation <50% AND avg kernel <8 µs → kernel execution is memory-bandwidth-starved rather than compute-bound. Short kernels issue rapidly but spend most time waiting on HBM.",
        "used_by": ["HBM bandwidth-bound"],
    },
    "RS_INJECTION_RATIO": {
        "value": 1.0,
        "rationale": "Ratio of RS GPU time to bwd compute GPU time above which RS injection pressure is flagged",
        "physical_justification": "Injection BW model: RS GPU ≥100% of bwd compute GPU means gradient synchronisation takes as long as backward computation. On H100 NVLink (≈60–70 GB/s large-message all-reduce BW), this is the point where all-reduce BW limits backward throughput as much as compute. Uses GPU-time ratio (not overlap) because async NCCL CPU_wall is just dispatch time.",
        "used_by": ["RS exceeds bwd compute"],
    },
    "TP_ON_CRITICAL_PATH_RATIO": {
        "value": 0.10,
        "rationale": "Ratio of TP GPU time to fwd compute GPU time above which synchronous TP on the critical path is flagged",
        "physical_justification": "Critical-path analysis: even 10% TP GPU time relative to compute is significant because TP collectives are synchronous per transformer layer (unlike FSDP which pipelines across layers). Synchronous collectives >10% of compute directly limit scaling efficiency via Amdahl's law.",
        "used_by": ["synchronous TP on critical path"],
    },
    "TP_ON_CRITICAL_PATH_OVERLAP": {
        "value": 0.20,
        "rationale": "Forward TP overlap below which (combined with significant TP time) the async TP mechanism is defeated",
        "physical_justification": "Async TP profitability model: when TP overlap <20% AND TP ≥10% of compute, the async TP mechanism hides <20% of collective latency — effectively synchronous. Only 1/5 of collective latency is hidden.",
        "used_by": ["synchronous TP on critical path"],
    },
    "NVLINK_SAT_COUNT": {
        "value": 50,
        "rationale": "Minimum number of small TP kernels per layer above which NVLink fragmentation is flagged",
        "physical_justification": "NVLink message-efficiency model: NVLink BW efficiency drops significantly for small messages (<1 KB). On H100 (18 NVLink 4.0 links × 450 GB/s uni), small-message throughput can be 5–10% of peak. ≥50 small TP kernels indicates message fragmentation that prevents NVLink BW saturation.",
        "used_by": ["NVLink saturation"],
    },
    "NVLINK_SAT_AVG_US": {
        "value": 10.0,
        "rationale": "Average TP kernel duration (µs) below which TP kernels are too short to saturate NVLink bandwidth",
        "physical_justification": "NVLink BW model: for a 128 KB TP message on H100 NVLink (≈70 GB/s achievable large-message BW per link pair), expected transfer ≲2 µs. Avg TP kernel <10 µs means launch overhead dominates and NVLink BW is never saturated.",
        "used_by": ["NVLink saturation"],
    },
    "COMM_COMPUTE_UTIL_THRESHOLD": {
        "value": 0.50,
        "rationale": "GPU utilisation below which (combined with meaningful AG time) no comm/compute overlap is flagged",
        "physical_justification": "Pipeline utilisation model: same physical basis as UTIL_LOW_THRESHOLD (50% is the standard low-utilisation boundary). When combined with AG ≥10% of fwd compute, the AG is a material contributor to the idle — not hidden by pipeline stagger.",
        "used_by": ["no comm/compute overlap"],
    },
    "AG_VS_FWD_RATIO": {
        "value": 0.10,
        "rationale": "Ratio of AG fwd GPU time to fwd compute GPU time below which AG is negligible for low-utilisation diagnosis",
        "physical_justification": "Signal-to-noise filter: AG <10% of fwd compute is negligible — low utilisation in that case is driven by other factors (data loading, synchronisation). The 10% floor ensures we only flag communication-mediated low utilisation.",
        "used_by": ["no comm/compute overlap"],
    },
    "EXPOSED_COMM_IDLE_THRESHOLD": {
        "value": 0.50,
        "rationale": "GPU idle fraction (1 − gpu_busy) above which communication exposure is flagged (requires fwd + bwd activity)",
        "physical_justification": "Utilisation-derived idle model: GPU idle ≥50% of wall time with both fwd and bwd active → communication exposure is the dominant limiter. Uses 1−gpu_busy which replaces the old overlap-efficiency formula that broke for async NCCL (CPU_wall = dispatch, not execution).",
        "used_by": ["exposed communication"],
    },
    "PIPELINE_OVERLAP_LOW": {
        "value": 0.20,
        "rationale": "Cross-layer GPU span overlap ratio below which the FSDP2 pipeline stagger is considered ineffective",
        "physical_justification": "Cross-layer GPU pipeline model: ideal overlap = (N−1)/N. Even a 2-layer pipeline should achieve 50%. Below 20% indicates insufficient micro-batches or synchronisation that serialises the GPU stream. Cross-check against CPU overlap_ratio (sweep-line) — GPU overlap is the real signal.",
        "used_by": ["low cross-layer GPU overlap"],
    },
}
