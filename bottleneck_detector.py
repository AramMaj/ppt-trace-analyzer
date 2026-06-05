"""
Bottleneck und Metriken

Rückgabe
=> Vorherige JSON + Erweiterungen über User Annotations und Bottlenecks
=> Textueller Report
    < Bottleneck Freitext Report >
    => Metriken für gesamte Unit
    => Metriken pro Unit
    => Unit Auflistung
    => Chronlogical

"""

import json
from typing import Dict, List, Optional, Tuple, Set, Iterator, Any
from collections import defaultdict
from dataclasses import dataclass, field

from trace_parser import LogicalOperation
from fsdp_detector import FSDP, FSDPUnit, FSDP_PREFIXES


def _is_fsdp_name(name: str) -> bool:
    return name.startswith(FSDP_PREFIXES)


def _phase_gpu_time(nodes: List[LogicalOperation]) -> float:
    return sum(n.gpu_duration for n in nodes)


def _phase_cpu_time(nodes: List[LogicalOperation]) -> float:
    return sum(n.cpu_duration for n in nodes)


def _phase_wall_time(nodes: List[LogicalOperation]) -> float:
    if not nodes:
        return 0.0
    start = min(n.start_time for n in nodes)
    end = max(n.end_time for n in nodes)
    return end - start


def _compute_overlap_metrics(units: List['FSDPUnit']) -> dict:
    """Compute overlap and serial execution efficiency from layer wall spans."""
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
    """Collect memory metrics from the unit's tree nodes.
    memory_delta is cumulative (includes children) after attribute_memory propagation,
    so we only sum the top-level phase nodes for each unit.
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


class Metrics:
    def __init__(self, unit: FSDPUnit, global_optimizer_gpu: float = 0.0, global_optimizer_cpu: float = 0.0,
                 num_units: int = 1, global_tp_ag_gpu: float = 0.0, global_tp_rs_gpu: float = 0.0,
                 global_tp_ar_gpu: float = 0.0, tp_kernels: Optional[List[dict]] = None):
        self.layer_name = unit.layer_name

        self.ag_fwd_gpu = _phase_gpu_time(unit.all_gather_fwd)
        self.ag_fwd_cpu = _phase_cpu_time(unit.all_gather_fwd)
        self.ag_fwd_wall = _phase_wall_time(unit.all_gather_fwd)

        self.fwd_cmp_gpu = _phase_gpu_time(unit.fwd_compute)
        self.fwd_cmp_cpu = _phase_cpu_time(unit.fwd_compute)
        self.fwd_cmp_wall = _phase_wall_time(unit.fwd_compute)

        self.ag_bwd_gpu = _phase_gpu_time(unit.all_gather_bwd)
        self.ag_bwd_cpu = _phase_cpu_time(unit.all_gather_bwd)
        self.ag_bwd_wall = _phase_wall_time(unit.all_gather_bwd)

        self.bwd_cmp_gpu = _phase_gpu_time(unit.bwd_compute)
        self.bwd_cmp_cpu = _phase_cpu_time(unit.bwd_compute)
        self.bwd_cmp_wall = _phase_wall_time(unit.bwd_compute)

        self.rs_gpu = _phase_gpu_time(unit.reduce_scatter)
        self.rs_cpu = _phase_cpu_time(unit.reduce_scatter)
        self.rs_wall = _phase_wall_time(unit.reduce_scatter)

        self.optimizer_gpu = global_optimizer_gpu / num_units if num_units > 0 else 0.0
        self.optimizer_cpu = global_optimizer_cpu / num_units if num_units > 0 else 0.0

        self.total_gpu = self.ag_fwd_gpu + self.fwd_cmp_gpu + self.ag_bwd_gpu + self.bwd_cmp_gpu + self.rs_gpu + self.optimizer_gpu
        self.total_cpu = self.ag_fwd_cpu + self.fwd_cmp_cpu + self.ag_bwd_cpu + self.bwd_cmp_cpu + self.rs_cpu + self.optimizer_cpu

        comm_gpu = self.ag_fwd_gpu + self.ag_bwd_gpu + self.rs_gpu
        comp_gpu = self.fwd_cmp_gpu + self.bwd_cmp_gpu
        self.comm_ratio = comm_gpu / self.total_gpu if self.total_gpu > 0 else 0.0
        self.comp_ratio = comp_gpu / self.total_gpu if self.total_gpu > 0 else 0.0

        self.optimizer_ratio = self.optimizer_gpu / self.total_gpu if self.total_gpu > 0 else 0.0

        # Layer wall span: time from first phase start to last phase end for this unit
        all_events = (unit.all_gather_fwd + unit.fwd_compute + unit.all_gather_bwd
                      + unit.bwd_compute + unit.reduce_scatter)
        self.layer_span = _phase_wall_time(all_events) if all_events else 0.0
        self.gpu_util = self.total_gpu / self.layer_span if self.layer_span > 0 else 0.0

        self.tp_ag_gpu = global_tp_ag_gpu / num_units if num_units > 0 else 0.0
        self.tp_rs_gpu = global_tp_rs_gpu / num_units if num_units > 0 else 0.0
        self.tp_ar_gpu = global_tp_ar_gpu / num_units if num_units > 0 else 0.0
        self.tp_total_gpu = self.tp_ag_gpu + self.tp_rs_gpu + self.tp_ar_gpu

        self.ag_fwd_count = len(unit.all_gather_fwd_gpu_kernels)
        self.ag_bwd_count = len(unit.all_gather_bwd_gpu_kernels)
        self.rs_count = len(unit.reduce_scatter_gpu_kernels)

        # Per-layer compute-communication overlap (TP kernels overlapping compute phases)
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
                         if k.get('ts', 0) >= fwd_start and k.get('ts', 0) + k.get('dur', 0) <= fwd_end)
            bwd_tp = sum(k.get('dur', 0) for k in tp_kernels
                         if k.get('ts', 0) >= bwd_start and k.get('ts', 0) + k.get('dur', 0) <= bwd_end)
            self.fwd_comp_comm_overlap = fwd_tp / fwd_wall if fwd_wall > 0 else 0.0
            self.bwd_comp_comm_overlap = bwd_tp / bwd_wall if bwd_wall > 0 else 0.0

        # Memory — from tree nodes if available
        self.memory_peak = 0
        self.memory_allocated = 0
        self.memory_freed = 0
        self.memory_has_data = False
        _collect_memory(unit, self)

        # Overlap — global metrics shared across units
        self.overlap_ratio = 0.0
        self.serial_exec_efficiency = 1.0
        self.idle_ratio = 0.0
        self.step_wall = 0.0

        # True communication ratio (FSDP + TP comm / total GPU)
        total_gpu = self.total_gpu + self.tp_total_gpu
        fsdp_comm = self.ag_fwd_gpu + self.ag_bwd_gpu + self.rs_gpu
        self.fsdp_comm_ratio = fsdp_comm / total_gpu if total_gpu > 0 else 0.0
        self.tp_comm_ratio = self.tp_total_gpu / total_gpu if total_gpu > 0 else 0.0
        self.comm_ratio = (fsdp_comm + self.tp_total_gpu) / total_gpu if total_gpu > 0 else 0.0

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
        }


def _format_us(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}s"
    if v >= 1_000:
        return f"{v / 1_000:.2f}ms"
    return f"{v:.1f}us"


class Bottlenecks:
    COMP_HEAVY_THRESHOLD = 0.70
    COMM_HEAVY_THRESHOLD = 0.40
    IO_HEAVY_THRESHOLD = 0.15
    AG_HEAVY_THRESHOLD = 0.40
    RS_HEAVY_THRESHOLD = 0.40
    TP_HEAVY_THRESHOLD = 0.15
    OPTIMIZER_HEAVY_THRESHOLD = 0.15
    UTIL_LOW_THRESHOLD = 0.50

    @classmethod
    def detect(cls, metrics: Metrics) -> List[str]:
        issues = []
        total_gpu = metrics.total_gpu + metrics.tp_total_gpu
        if total_gpu == 0:
            return issues

        # ----- bottleneck types: compute, i/o, gpu rank -----
        # Compute-bound
        comp_gpu = metrics.fwd_cmp_gpu + metrics.bwd_cmp_gpu
        comp_ratio = comp_gpu / total_gpu if total_gpu > 0 else 0.0
        if comp_ratio >= cls.COMP_HEAVY_THRESHOLD:
            issues.append(f"compute-bound (comp={comp_ratio:.1%})")

        # I/O-bound: significant idle time between GPU activities
        if metrics.idle_ratio >= cls.IO_HEAVY_THRESHOLD:
            issues.append(f"I/O-bound ({metrics.idle_ratio:.1%} idle)")

        # GPU-rank bottleneck: which phase dominates and limits throughput
        if metrics.comm_ratio >= cls.COMM_HEAVY_THRESHOLD:
            issues.append(f"comm-bound (comm={metrics.comm_ratio:.1%})")
        fsdp_comm = metrics.ag_fwd_gpu + metrics.ag_bwd_gpu + metrics.rs_gpu
        if fsdp_comm > 0:
            ag_ratio = (metrics.ag_fwd_gpu + metrics.ag_bwd_gpu) / fsdp_comm
            rs_ratio = metrics.rs_gpu / fsdp_comm
            if ag_ratio >= cls.AG_HEAVY_THRESHOLD:
                issues.append(f"all-gather-heavy ({ag_ratio:.1%} of FSDP comm)")
            if rs_ratio >= cls.RS_HEAVY_THRESHOLD:
                issues.append(f"reduce-scatter-heavy ({rs_ratio:.1%} of FSDP comm)")

        tp_total = metrics.tp_ag_gpu + metrics.tp_rs_gpu + metrics.tp_ar_gpu
        if tp_total > 0 and tp_total / total_gpu >= cls.TP_HEAVY_THRESHOLD:
            issues.append(f"TP-heavy ({tp_total/total_gpu:.1%} of total GPU)")

        if metrics.optimizer_gpu > 0 and metrics.optimizer_ratio >= cls.OPTIMIZER_HEAVY_THRESHOLD:
            issues.append(f"optimizer-heavy ({metrics.optimizer_ratio:.1%} of total GPU)")

        # GPU-rank: which phase dominates total GPU time
        phases = [("AG fwd", metrics.ag_fwd_gpu), ("AG bwd", metrics.ag_bwd_gpu),
                  ("RS", metrics.rs_gpu), ("Fwd cmp", metrics.fwd_cmp_gpu),
                  ("Bwd cmp", metrics.bwd_cmp_gpu), ("Optimizer", metrics.optimizer_gpu),
                  ("TP", tp_total)]
        for name, val in phases:
            if total_gpu > 0 and val / total_gpu > 0.35:
                issues.append(f"{name} dominates ({val/total_gpu:.1%} of total GPU)")
                break  # only report the top phase

        # GPU utilization
        if 0 < metrics.gpu_util < cls.UTIL_LOW_THRESHOLD:
            issues.append(f"low GPU utilization ({metrics.gpu_util:.1%})")
        if metrics.gpu_util > 1.0:
            issues.append(f"high compute-comm overlap ({metrics.gpu_util:.1%} util)")

        return issues


class Report:
    def __init__(self, fsdp: FSDP, root_nodes: List[LogicalOperation], output_path: Optional[str] = None):
        self.fsdp = fsdp
        self.root_nodes = root_nodes
        self.output_path = output_path
        self.metrics_list: List[Metrics] = []
        self.aggregated: Dict[str, float] = {}

    def generate_report(self):
        num_units = len(self.fsdp.units)
        opt_gpu = self.fsdp.optimizer_gpu
        opt_cpu = self.fsdp.optimizer_cpu
        tp_ag = self.fsdp.tp_all_gather_gpu
        tp_rs = self.fsdp.tp_reduce_scatter_gpu
        tp_ar = self.fsdp.tp_all_reduce_gpu
        # Collect all TP kernel events for per-layer overlap attribution
        tp_kernels = list(self.fsdp.tp_all_gather + self.fsdp.tp_reduce_scatter + self.fsdp.tp_all_reduce)
        for unit in self.fsdp.units:
            self.metrics_list.append(Metrics(unit, opt_gpu, opt_cpu, num_units,
                                             tp_ag, tp_rs, tp_ar, tp_kernels=tp_kernels))

        self._compute_aggregated()
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

    def _build_report_text(self) -> str:
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

        # Efficiency metrics
        lines.append("--- Efficiency ---")
        comp_gpu = self.aggregated.get("fwd_cmp_gpu_us", 0) + self.aggregated.get("bwd_cmp_gpu_us", 0)
        tp_total = self.aggregated.get("tp_total_gpu_us", 0)
        fsdp_comm = self.aggregated.get("ag_fwd_gpu_us", 0) + self.aggregated.get("ag_bwd_gpu_us", 0) + self.aggregated.get("rs_gpu_us", 0)
        opt_ratio = self.aggregated.get('optimizer_ratio', 0)
        true_comp = comp_gpu - tp_total  # TP kernels are counted inside comp GPU time
        total = total_gpu + tp_total  # include TP in total for true ratio
        lines.append(f"  True compute:         {true_comp / total:.1%} (ex-TP)")
        lines.append(f"  TP communication:     {tp_total / total:.1%}")
        lines.append(f"  FSDP communication:   {fsdp_comm / total:.1%}")
        lines.append(f"  Optimizer:            {opt_ratio:.1%}")
        avg_util = sum(m.gpu_util for m in self.metrics_list) / len(self.metrics_list) if self.metrics_list else 0.0
        lines.append(f"  Avg GPU utilization:  {avg_util:.1%} (per-layer GPU busy / layer span)")
        max_span = max(m.layer_span for m in self.metrics_list) if self.metrics_list else 0.0
        min_span = min(m.layer_span for m in self.metrics_list if m.layer_span > 0) or max_span
        lines.append(f"  Layer span imbalance: {max_span / min_span:.1f}x (max/min layer span ratio)")
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
                  f"{'Util':>7s}{'Span':>9s}{'F-Ovl':>8s}{'B-Ovl':>8s}{mem_col:>10s}  {'Bottleneck'}")
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
                f"{d['gpu_util']:>6.1%} "
                f"{_format_us(d['layer_span_us']):>9s}"
                f"{d['fwd_comp_comm_overlap']:>7.1%} "
                f"{d['bwd_comp_comm_overlap']:>7.1%} "
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
        markers = []
        for m in self.metrics_list:
            d = m.to_dict()
            issues = Bottlenecks.detect(m)
            markers.append({
                "layer": m.layer_name,
                "metrics": d,
                "bottlenecks": issues,
            })
        return markers

    @staticmethod
    def get_fsdp_timeline_aggregated_string(agg: Dict[str, float]) -> str:
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
