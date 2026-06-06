"""
Chrome Trace annotation generator for PPT trace analysis.

Extracts FSDP phases, TP kernels, flow arrows, GPU counters,
and bottleneck markers as Chrome Trace user events.
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

from trace_parser import TraceParser, TraceParserHelper
from fsdp_detector import StandardFSDPDetector
from bottleneck_detector import Report, Bottlenecks, _phase_gpu_time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PPT_PID_BASE = 9999
PPT_CAT = "ppt_analyzer"
TID_BLOCK = 200  # Fallback block size when tid_offset is not provided

PHASE_COLORS = {
    "AG fwd": "#4CAF50",
    "Fwd cmp": "#2196F3",
    "AG bwd": "#FF9800",
    "Bwd cmp": "#F44336",
    "RS": "#9C27B0",
    "Optimizer": "#607D8B",
    "TP": "#FFEB3B",
}

PHASE_LABELS = ["AG fwd", "Fwd cmp", "AG bwd", "Bwd cmp", "RS"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase_wall_span(nodes):
    if not nodes:
        return None
    start = min(n.start_time for n in nodes)
    end = max(n.end_time for n in nodes)
    return (start, end)


def _get_phase_spans(unit):
    phases = []
    phase_src = [
        ("AG fwd", unit.all_gather_fwd),
        ("Fwd cmp", unit.fwd_compute),
        ("AG bwd", unit.all_gather_bwd),
        ("Bwd cmp", unit.bwd_compute),
        ("RS", unit.reduce_scatter),
    ]
    for label, node_list in phase_src:
        span = _phase_wall_span(node_list)
        if span is not None:
            phases.append((label, span[0], span[1], node_list))
    return phases


def _find_profiler_steps(trace_file: str) -> List[Tuple[str, int, int, int]]:
    """Scan the raw trace file for ProfilerStep#N events.

    Returns a list of ``(event_name, ts_us, end_us, tid)`` sorted by ts.
    Each entry corresponds to one ProfilerStep complete event (ph == 'X')
    with a ``cat`` in ``('cpu_op', 'user_annotation')``.
    """
    steps: List[Tuple[str, int, int, int]] = []
    with open(trace_file) as f:
        data = json.load(f)
    for ev in data.get("traceEvents", []):
        name = ev.get("name", "")
        if not name.startswith("ProfilerStep#"):
            continue
        ph = ev.get("ph", "")
        cat = ev.get("cat", "")
        if ph == "X" and cat in ("cpu_op", "user_annotation"):
            ts = ev.get("ts", 0)
            dur = ev.get("dur", 0)
            tid = ev.get("tid", 0)
            steps.append((name, ts, ts + dur, tid))
    steps.sort(key=lambda x: x[1])

    # Deduplicate entries that share the same name & time range (same step
    # recorded on multiple threads / pids); keep only the first occurrence.
    seen = set()
    unique = []
    for name, ts, end, tid in steps:
        key = (name, ts, end)
        if key not in seen:
            seen.add(key)
            unique.append((name, ts, end, tid))
    return unique


def _filter_gpu_events(parser, step_start: int, step_end: int):
    """Filter the parser's GPU / memory events to *step_start..step_end*."""
    step_dur = step_end - step_start
    margin = max(step_dur * 0.05, 2000.0)
    parser.gpu_events = [
        ev
        for ev in parser.gpu_events
        if ev.get("ts", 0) >= step_start - margin
        and ev.get("ts", 0) + ev.get("dur", 0) <= step_end + margin
    ]
    parser.memory_events = [
        ev
        for ev in parser.memory_events
        if step_start - margin <= ev.get("ts", 0) <= step_end + margin
    ]


def _prune_roots_for_step(roots, step_name, step_start, step_end):
    """Keep only the ProfilerStep subtree and roots within the step time range.

    Returns a new root list where:
    - The matching ProfilerStep root (and its entire subtree) is kept.
    - Roots on other threads that fall entirely within ``[step_start, step_end]``
      are kept.
    - All other roots (other steps, setup code, etc.) are discarded.
    """
    profiler_root = None
    for root in roots:
        raw = root.raw_event or {}
        if raw.get("name", "") == step_name and raw.get("ph") == "X":
            if abs(root.start_time - step_start) < 100:
                profiler_root = root
                break

    result = []
    if profiler_root is not None:
        result.append(profiler_root)

    for root in roots:
        if root is profiler_root:
            continue
        if root.start_time >= step_start and root.end_time <= step_end:
            result.append(root)

    return result


# ---------------------------------------------------------------------------
# Step-level analysis
# ---------------------------------------------------------------------------

def _analyze_step(
    trace_file: str,
    step_name: str,
    step_start: int,
    step_end: int,
):
    """Run the analysis pipeline for *one* ProfilerStep.

    Returns ``(fsdp_orch, report)`` or ``(None, None)`` on failure.
    """
    parser = TraceParser(trace_file)
    if not parser.load():
        return None, None

    _filter_gpu_events(parser, step_start, step_end)
    roots = parser.build_tree()
    roots = _prune_roots_for_step(roots, step_name, step_start, step_end)
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)

    detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
    fsdp = detector.extract_fsdp_phases(roots)

    fsdp.optimizer_step = [
        n for n in fsdp.optimizer_step
        if step_start <= n.start_time <= step_end
    ]
    fsdp.optimizer_zero_grad = [
        n for n in fsdp.optimizer_zero_grad
        if step_start <= n.start_time <= step_end
    ]

    report = Report(fsdp, roots, output_path=None)
    report.generate_report()

    return fsdp, report


# ---------------------------------------------------------------------------
# Cross-step annotation generation (one TID per layer, steps go right)
# ---------------------------------------------------------------------------

def annotate_trace(trace_file: str, output_file: str, max_steps: int = 0):
    """Read *trace_file*, detect all ProfilerStep events, run the full
    analysis pipeline for **each** step, and produce an annotated Chrome
    Trace JSON.

    All steps share a single ``PID = PPT_PID_BASE``.  Each **layer** gets
    exactly one TID; phases from *all* steps for that layer are placed on
    the same TID so they flow **rightward** in time on the same row.

    Parameters
    ----------
    trace_file:
        Path to the input Chrome Trace JSON.
    output_file:
        Path for the annotated output.
    max_steps:
        Maximum number of steps to annotate (0 = all).
    """
    # 1. Load original trace data
    with open(trace_file) as f:
        data = json.load(f)

    # 2. Find ProfilerSteps
    steps = _find_profiler_steps(trace_file)
    if not steps:
        steps = [("Trace", 0, 0, 0)]
        all_ts = [
            ev.get("ts", 0)
            for ev in data.get("traceEvents", [])
            if ev.get("ph") == "X"
        ]
        if all_ts:
            steps[0] = ("Trace", min(all_ts), max(all_ts), 0)
        print("  No ProfilerStep markers found, annotating full trace as one step.")
    else:
        print(f"  Found {len(steps)} ProfilerStep(s)")

    if max_steps and max_steps < len(steps):
        steps = steps[-max_steps:]

    # 3. Analyse all steps
    step_analyses = []
    for step_name, step_start, step_end, _ in steps:
        sname = step_name.replace("ProfilerStep#", "Step#")
        print(f"  Analysing {sname} …")
        fsdp, report = _analyze_step(trace_file, step_name, step_start, step_end)
        if fsdp is None:
            continue
        step_analyses.append((step_name, fsdp, report))

    if not step_analyses:
        print("  No steps could be analysed.")
        return

    max_layers = max(len(fsdp.units) for _, fsdp, _ in step_analyses)
    pid = PPT_PID_BASE
    annotations = []

    # -- Layer threads: all steps' phases on the layer's TID --------------
    for layer_idx in range(max_layers):
        tid = layer_idx + 1

        # Pick a step that has this layer for the thread name
        layer_name = f"Layer {layer_idx}"
        for _, fsdp, _ in step_analyses:
            if layer_idx < len(fsdp.units):
                raw = fsdp.units[layer_idx].layer_name
                if raw:
                    layer_name = f"Layer {layer_idx}: {raw}"
                    break

        annotations.append({
            "ph": "M", "pid": pid, "tid": tid,
            "name": "thread_name",
            "args": {"name": layer_name},
        })

        for step_idx, (step_name, fsdp, _) in enumerate(step_analyses):
            if layer_idx >= len(fsdp.units):
                continue
            unit = fsdp.units[layer_idx]

            phases = _get_phase_spans(unit)
            for pi, (label, start, end, nodes) in enumerate(phases):
                gpu_dur = _phase_gpu_time(nodes)
                annotations.append({
                    "ph": "X", "pid": pid, "tid": tid,
                    "ts": start, "dur": end - start,
                    "cat": PPT_CAT,
                    "name": f"{label} ({step_name})",
                    "args": {
                        "phase": label,
                        "layer": layer_name,
                        "gpu_us": round(gpu_dur, 1),
                        "wall_us": round(end - start, 1),
                        "num_nodes": len(nodes),
                        "step": step_name,
                    },
                    "cname": PHASE_COLORS.get(label, "#888"),
                })

                # Flow arrow: phase transition within the same step
                if pi + 1 < len(phases):
                    next_label, next_start, _, _ = phases[pi + 1]
                    flow_id = f"flow_{step_idx}_{layer_idx}_{pi}"
                    annotations.append({
                        "ph": "s", "pid": pid, "tid": tid,
                        "ts": end, "cat": PPT_CAT,
                        "id": flow_id,
                        "name": f"{label} -> {next_label} ({step_name})",
                        "args": {"step": step_name},
                    })
                    annotations.append({
                        "ph": "f", "pid": pid, "tid": tid,
                        "ts": next_start, "cat": PPT_CAT,
                        "id": flow_id,
                        "name": f"{label} -> {next_label} ({step_name})",
                        "bp": "e",
                        "args": {"step": step_name},
                    })

    # -- Optimizer thread (all steps merged) ------------------------------
    opt_tid = max_layers + 1
    if any(fsdp.optimizer_step for _, fsdp, _ in step_analyses):
        annotations.append({
            "ph": "M", "pid": pid, "tid": opt_tid,
            "name": "thread_name",
            "args": {"name": "Optimizer (all steps)"},
        })
        for step_name, fsdp, _ in step_analyses:
            for opt_node in fsdp.optimizer_step:
                annotations.append({
                    "ph": "X", "pid": pid, "tid": opt_tid,
                    "ts": opt_node.start_time,
                    "dur": opt_node.end_time - opt_node.start_time,
                    "cat": PPT_CAT,
                    "name": "Optimizer.step",
                    "args": {
                        "phase": "Optimizer",
                        "gpu_us": round(opt_node.gpu_duration, 1),
                        "cpu_us": round(opt_node.cpu_duration, 1),
                        "step": step_name,
                    },
                    "cname": PHASE_COLORS["Optimizer"],
                })

    # -- TP collectives thread (all steps merged) -------------------------
    tp_tid = max_layers + 2
    has_tp = any(
        fsdp.tp_all_gather or fsdp.tp_reduce_scatter or fsdp.tp_all_reduce
        for _, fsdp, _ in step_analyses
    )
    if has_tp:
        annotations.append({
            "ph": "M", "pid": pid, "tid": tp_tid,
            "name": "thread_name",
            "args": {"name": "TP collectives (all steps)"},
        })
        tp_kernels = []
        for step_name, fsdp, _ in step_analyses:
            for k in fsdp.tp_all_gather + fsdp.tp_reduce_scatter + fsdp.tp_all_reduce:
                kc = dict(k)
                kc["_step"] = step_name
                tp_kernels.append(kc)
        tp_kernels.sort(key=lambda k: k.get("ts", 0))
        for k in tp_kernels:
            label = k.get("_coll_name", "") or k.get("name", "")
            annotations.append({
                "ph": "X", "pid": pid, "tid": tp_tid,
                "ts": k.get("ts", 0), "dur": k.get("dur", 0),
                "cat": PPT_CAT,
                "name": f"TP {label}",
                "args": {
                    "phase": "TP",
                    "gpu_us": round(k.get("dur", 0), 1),
                    "step": k.get("_step", ""),
                },
                "cname": PHASE_COLORS["TP"],
            })

    # -- GPU activity counter (all layers, all steps) ---------------------
    gpu_tid = max_layers + 4
    # Collect combined push-pop events from all steps
    all_events = []
    for _, fsdp, _ in step_analyses:
        for unit in fsdp.units:
            for _, start, end, _ in _get_phase_spans(unit):
                all_events.append((start, 1))
                all_events.append((end, -1))
    if all_events:
        all_events.sort(key=lambda x: x[0])
        annotations.append({
            "ph": "M", "pid": pid, "tid": gpu_tid,
            "name": "thread_name",
            "args": {"name": "GPU activity (all steps)"},
        })
        active = 0
        for ts, delta in all_events:
            annotations.append({
                "ph": "C", "pid": pid, "tid": gpu_tid,
                "ts": ts, "cat": PPT_CAT,
                "name": "Active layers",
                "args": {"active_layers": active + (1 if delta > 0 else 0),
                         "step": "all"},
            })
            active += delta

    # -- Bottleneck markers (all steps merged) ----------------------------
    bneck_tid = max_layers + 3
    annotations.append({
        "ph": "M", "pid": pid, "tid": bneck_tid,
        "name": "thread_name",
        "args": {"name": "Bottlenecks (all steps)"},
    })
    for step_name, fsdp, report in step_analyses:
        for metric, unit in zip(report.metrics_list, fsdp.units):
            issues = Bottlenecks.detect(metric)
            if not issues:
                continue
            phases = _get_phase_spans(unit)
            if not phases:
                continue
            longest = max(phases, key=lambda p: p[2] - p[1])
            ts = longest[1] + (longest[2] - longest[1]) * 0.3

            short = []
            for iss in issues:
                if "dominates" in iss or "bound" in iss:
                    short.append(iss.split("(")[0].strip())
                else:
                    short.append(iss)
            annotations.append({
                "ph": "i", "pid": pid, "tid": bneck_tid,
                "ts": ts, "cat": PPT_CAT,
                "name": "; ".join(short[:3]),
                "args": {
                    "layer": metric.layer_name,
                    "issues_full": "; ".join(issues),
                    "comm_ratio": round(metric.comm_ratio, 3),
                    "comp_ratio": round(metric.comp_ratio, 3),
                    "gpu_util": round(metric.gpu_util, 3),
                    "comm_ratio_pct": f"{metric.comm_ratio:.1%}",
                    "comp_ratio_pct": f"{metric.comp_ratio:.1%}",
                    "step": step_name,
                },
                "s": "t", "cname": "#FF5722",
            })

    # 4. Merge and write
    data.setdefault("traceEvents", []).extend(annotations)
    num_orig = len(data["traceEvents"]) - len(annotations)
    with open(output_file, "w") as f:
        json.dump(data, f)
    print(f"  Annotated trace written to {output_file}")
    print(f"  {num_orig} original events + {len(annotations)} annotations")

    # Stats per step
    for step_name, _, _, _ in steps:
        step_anns = [a for a in annotations
                     if a.get("args", {}).get("step") == step_name]
        n_phases = sum(
            1 for a in step_anns
            if a.get("ph") == "X"
            and a.get("args", {}).get("phase") not in ("TP", "Optimizer")
        )
        n_tp = sum(1 for a in step_anns
                   if a.get("args", {}).get("phase") == "TP")
        n_bneck = sum(1 for a in step_anns if a.get("ph") == "i")
        n_flows = sum(1 for a in step_anns if a.get("ph") == "s")
        n_opt = sum(1 for a in step_anns
                    if a.get("args", {}).get("phase") == "Optimizer")
        name = step_name.replace("ProfilerStep#", "#")
        print(
            f"    {name}: {n_phases} phases, {n_flows} flows, "
            f"{n_tp} TP, {n_opt} optimizer, {n_bneck} bottlenecks"
        )
