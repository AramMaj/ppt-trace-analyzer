"""METRIC_REGISTRY: documentation for every metric in Metrics.to_dict().

Each entry documents one metric key with:
- description: plain-English explanation
- used_by: list of bottleneck short names whose detection logic reads this metric
- unit: µs, ratio, count, bytes, FLOPs, tokens/s/GPU
- calculation: how the metric is derived from raw data

This file is imported by html_report.py to render the collapsible metric
registry table. It lives in its own module so that CLI tools, comparison
reports, and other consumers can reference it without importing the full
detection pipeline.
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
        "description": "All-gather backward GPU time — NCCL kernel duration for the backward all-gather (including NCCL kernels found inside bwd compute phases and ac2g supplement)",
        "used_by": ["all-gather-heavy", "reduce-scatter-heavy", "fwd-bwd imbalance", "dominant phase"],
        "unit": "µs",
        "calculation": "_phase_gpu_time(unit.all_gather_bwd) + _phase_gpu_time(unit.all_gather_bwd_nccl) + unit.ag_bwd_supplement_us + _collect_nccl_kernel_time(unit.bwd_compute) — NCCL AG bwd + NCCL in bwd compute",
    },
    "bwd_cmp_gpu_us": {
        "description": "Backward compute GPU time — gradient computation kernels (direct, non-overlapping)",
        "used_by": ["fwd-bwd imbalance", "RS exceeds bwd compute", "exposed communication"],
        "unit": "µs",
        "calculation": "_phase_gpu_time_direct(unit.bwd_compute) — sum of direct GPU kernel durations in backward compute",
    },
    "rs_gpu_us": {
        "description": "Reduce-scatter GPU time — NCCL kernel duration for the reduce-scatter",
        "used_by": ["all-gather-heavy", "reduce-scatter-heavy", "fwd-bwd imbalance", "RS exceeds bwd compute", "dominant phase"],
        "unit": "µs",
        "calculation": "_phase_gpu_time(unit.reduce_scatter) — sum of NCCL kernel durations for reduce-scatter",
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
