"""

"""
import sys
import csv
import os
import json
from trace_parser import TraceParser
from fsdp_detector import StandardFSDPDetector
from bottleneck_detector import Report, Metrics, Bottlenecks, _format_us, _phase_gpu_time


def _process_trace(trace_file: str):
    """Run the full pipeline on one trace. Returns (aggregated, metrics_list, fsdp, report_text) or None."""
    parser = TraceParser(trace_file)
    if not parser.load():
        return None

    roots = parser.build_tree()
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)

    detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
    fsdp = detector.extract_fsdp_phases(roots)

    report = Report(fsdp, roots, output_path=None)
    text, markers = report.generate_report()

    return report.aggregated, report.metrics_list, fsdp, report, text


def _format_bottleneck_summary(metrics_list, aggregated) -> str:
    """Short one-line bottleneck summary for a trace."""
    from collections import defaultdict
    from bottleneck_detector import Bottlenecks
    all_issues = defaultdict(list)
    for m in metrics_list:
        issues = Bottlenecks.detect(m)
        for iss in issues:
            all_issues[iss].append(m.layer_name)
    if not all_issues:
        return "OK"
    parts = []
    for iss in list(all_issues.keys())[:3]:
        parts.append(iss)
    return "; ".join(parts)


PPT_PID = 9999
PPT_CAT = 'ppt_analyzer'


def _phase_wall_span(nodes):
    """Return (start, end) spanning a list of LogicalOperation nodes."""
    if not nodes:
        return None
    start = min(n.start_time for n in nodes)
    end = max(n.end_time for n in nodes)
    return (start, end)


def _annotate_trace(trace_file, output_file):
    """Add user annotations to a copy of the trace, showing detected FSDP phases,
    optimizer step, and bottleneck markers."""
    with open(trace_file, 'r') as f:
        data = json.load(f)

    # Run analysis
    parser = TraceParser(trace_file)
    if not parser.load():
        print("Failed to load trace for annotation.")
        return
    roots = parser.build_tree()
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)
    detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
    fsdp = detector.extract_fsdp_phases(roots)
    report = Report(fsdp, roots, output_path=None)
    text, markers = report.generate_report()

    annotations = []

    # Process name
    annotations.append({
        'ph': 'M', 'pid': PPT_PID, 'tid': 0,
        'name': 'process_name', 'args': {'name': 'PPT Analyzer'}
    })

    all_phases = []

    # ----- Unit phases (one thread per layer) -----
    for idx, unit in enumerate(fsdp.units):
        tid = idx + 1
        annotations.append({
            'ph': 'M', 'pid': PPT_PID, 'tid': tid,
            'name': 'thread_name', 'args': {'name': unit.layer_name}
        })

        phase_defs = [
            ('AG fwd', unit.all_gather_fwd, '#4CAF50'),
            ('Fwd cmp', unit.fwd_compute, '#2196F3'),
            ('AG bwd', unit.all_gather_bwd, '#FF9800'),
            ('Bwd cmp', unit.bwd_compute, '#F44336'),
            ('RS', unit.reduce_scatter, '#9C27B0'),
        ]
        for label, nodes, color in phase_defs:
            span = _phase_wall_span(nodes)
            if span is None:
                continue
            gpu_dur = _phase_gpu_time(nodes)
            annotations.append({
                'ph': 'X', 'pid': PPT_PID, 'tid': tid,
                'ts': span[0], 'dur': span[1] - span[0],
                'cat': PPT_CAT, 'name': f'{label} ({unit.layer_name})',
                'args': {
                    'phase': label,
                    'layer': unit.layer_name,
                    'gpu_us': round(gpu_dur, 1),
                    'wall_us': round(span[1] - span[0], 1),
                    'num_nodes': len(nodes),
                },
                'cname': color,
            })
            all_phases.append((span[0], label, unit.layer_name))

    # ----- Optimizer step -----
    opt_tid = len(fsdp.units) + 1
    annotations.append({
        'ph': 'M', 'pid': PPT_PID, 'tid': opt_tid,
        'name': 'thread_name', 'args': {'name': 'Optimizer'}
    })
    for opt_node in fsdp.optimizer_step:
        annotations.append({
            'ph': 'X', 'pid': PPT_PID, 'tid': opt_tid,
            'ts': opt_node.start_time,
            'dur': opt_node.end_time - opt_node.start_time,
            'cat': PPT_CAT, 'name': 'Optimizer.step',
            'args': {
                'phase': 'Optimizer',
                'gpu_us': round(opt_node.gpu_duration, 1),
                'cpu_us': round(opt_node.cpu_duration, 1),
            },
            'cname': '#607D8B',
        })
        all_phases.append((opt_node.start_time, 'Optimizer', ''))

    # ----- TP kernels -----
    tp_tid = len(fsdp.units) + 2
    annotations.append({
        'ph': 'M', 'pid': PPT_PID, 'tid': tp_tid,
        'name': 'thread_name', 'args': {'name': 'TP collectives'}
    })
    tp_kernels = list(fsdp.tp_all_gather + fsdp.tp_reduce_scatter + fsdp.tp_all_reduce)
    tp_kernels.sort(key=lambda k: k.get('ts', 0))
    for k in tp_kernels[:500]:  # cap to avoid bloat
        kname = k.get('name', '')
        coll = k.get('_coll_name', '')
        label = coll or kname
        annotations.append({
            'ph': 'X', 'pid': PPT_PID, 'tid': tp_tid,
            'ts': k.get('ts', 0), 'dur': k.get('dur', 0),
            'cat': PPT_CAT, 'name': f'TP {label}',
            'args': {
                'phase': 'TP',
                'gpu_us': round(k.get('dur', 0), 1),
            },
            'cname': '#FFEB3B',
        })

    # ----- Bottleneck markers -----
    bneck_tid = len(fsdp.units) + 3
    annotations.append({
        'ph': 'M', 'pid': PPT_PID, 'tid': bneck_tid,
        'name': 'thread_name', 'args': {'name': 'Bottlenecks'}
    })
    for m in report.metrics_list:
        issues = Bottlenecks.detect(m)
        if not issues:
            continue
        # Find midpoint of the unit's span
        all_nodes = [n for lst in [
            fsdp.units[report.metrics_list.index(m)].all_gather_fwd,
            fsdp.units[report.metrics_list.index(m)].fwd_compute,
            fsdp.units[report.metrics_list.index(m)].all_gather_bwd,
            fsdp.units[report.metrics_list.index(m)].bwd_compute,
            fsdp.units[report.metrics_list.index(m)].reduce_scatter,
        ] for n in lst]
        if not all_nodes:
            continue
        mid = (min(n.start_time for n in all_nodes) + max(n.end_time for n in all_nodes)) / 2
        annotations.append({
            'ph': 'i', 'pid': PPT_PID, 'tid': bneck_tid,
            'ts': mid,
            'cat': PPT_CAT, 'name': '; '.join(issues),
            'args': {
                'layer': m.layer_name,
                'issues': '; '.join(issues),
                'comm_ratio': round(m.comm_ratio, 3),
                'comp_ratio': round(m.comp_ratio, 3),
                'gpu_util': round(m.gpu_util, 3),
                'overlap_ratio': round(m.overlap_ratio, 3),
            },
            's': 't',  # instant event scope: thread
            'cname': '#FF5722',
        })

    # ----- Merge and write -----
    data.setdefault('traceEvents', []).extend(annotations)
    num_orig = len(data['traceEvents']) - len(annotations)
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"Annotated trace written to {output_file}")
    print(f"  Added {len(annotations)} events ({num_orig} original + {len(annotations)} annotations)")


def _compare_traces(trace_files, output_file=None):
    results = []
    for tf in trace_files:
        label = os.path.basename(tf)
        print(f"Processing {label}...", file=sys.stderr)
        r = _process_trace(tf)
        if r is None:
            print(f"  FAILED to load {tf}", file=sys.stderr)
            continue
        results.append((label, r))

    if not results:
        print("No traces processed successfully.")
        sys.exit(1)

    # ----- Build comparison table -----
    rows = []
    headers = [
        "Trace", "Layers",
        "Step wall", "AG fwd", "Fwd cmp", "AG bwd", "Bwd cmp",
        "RS", "Opt", "TP total", "Total GPU",
        "Util", "Comm", "FSDP comm", "TP comm",
        "Overlap", "Serial eff", "Idle",
        "F-Ovl", "B-Ovl",
        "Mem peak",
        "Bottlenecks",
    ]

    for label, (agg, metrics_list, fsdp, report, text) in results:
        num_units = len(metrics_list)
        step_wall = agg.get("step_wall", 0)

        # GPU times (per-unit average)
        ag_fwd = agg.get("ag_fwd_gpu_us", 0)
        fwd_cmp = agg.get("fwd_cmp_gpu_us", 0)
        ag_bwd = agg.get("ag_bwd_gpu_us", 0)
        bwd_cmp = agg.get("bwd_cmp_gpu_us", 0)
        rs = agg.get("rs_gpu_us", 0)
        opt = agg.get("optimizer_gpu_us", 0)
        tp_total = agg.get("tp_total_gpu_us", 0)
        total_gpu = agg.get("total_gpu_us", 0) + tp_total

        # GPU util (average across units)
        avg_util = sum(m.gpu_util for m in metrics_list) / max(len(metrics_list), 1)

        # Overlap
        overlap_ratio = agg.get("overlap_ratio", 0)
        serial_eff = agg.get("serial_exec_efficiency", 0)
        idle_ratio = agg.get("idle_ratio", 0)

        # Per-layer comp-comm overlap (averages)
        avg_fwd_ovl = sum(m.fwd_comp_comm_overlap for m in metrics_list) / max(len(metrics_list), 1)
        avg_bwd_ovl = sum(m.bwd_comp_comm_overlap for m in metrics_list) / max(len(metrics_list), 1)

        # Communication ratios
        comm_ratio = agg.get("comm_ratio", 0)
        fsdp_comm_ratio = (
            sum(m.fsdp_comm_ratio for m in metrics_list) / max(len(metrics_list), 1)
        )
        tp_comm_ratio = (
            sum(m.tp_comm_ratio for m in metrics_list) / max(len(metrics_list), 1)
        )

        # Memory
        mem_peak_gib = max((m.memory_peak for m in metrics_list if m.memory_has_data), default=0) / (1024**3)

        # Bottlenecks
        bneck = _format_bottleneck_summary(metrics_list, agg)

        rows.append({
            "Trace": label,
            "Layers": num_units,
            "Step wall": step_wall,
            "AG fwd": ag_fwd,
            "Fwd cmp": fwd_cmp,
            "AG bwd": ag_bwd,
            "Bwd cmp": bwd_cmp,
            "RS": rs,
            "Opt": opt,
            "TP total": tp_total,
            "Total GPU": total_gpu,
            "Util": avg_util,
            "Comm": comm_ratio,
            "FSDP comm": fsdp_comm_ratio,
            "TP comm": tp_comm_ratio,
            "Overlap": overlap_ratio,
            "Serial eff": serial_eff,
            "Idle": idle_ratio,
            "F-Ovl": avg_fwd_ovl,
            "B-Ovl": avg_bwd_ovl,
            "Mem peak": mem_peak_gib,
            "Bottlenecks": bneck,
        })

    # ----- Print text table -----
    col_widths = {h: max(len(h), 8) for h in headers}
    # Adjust trace column to fit all labels
    col_widths["Trace"] = max(len(r["Trace"]) for r in rows)

    def _fmt_val(row, header) -> str:
        v = row[header]
        if isinstance(v, float):
            if header in ("Util", "Comm", "FSDP comm", "TP comm",
                          "Overlap", "Serial eff", "Idle", "F-Ovl", "B-Ovl"):
                return f"{v:.1%}"
            elif header in ("Mem peak",):
                return f"{v:.1f}G" if v > 0 else "N/A"
            elif header in ("Step wall", "AG fwd", "Fwd cmp", "AG bwd", "Bwd cmp",
                            "RS", "Opt", "TP total", "Total GPU"):
                return _format_us(v)
            else:
                return f"{v:.1f}"
        elif isinstance(v, int):
            return str(v)
        return str(v)

    # Table header
    sep = "  "
    hdr_line = sep.join(h.ljust(col_widths[h]) for h in headers)
    print("=" * len(hdr_line))
    print("Benchmark Comparison")
    print("=" * len(hdr_line))
    print(hdr_line)
    print("-" * len(hdr_line))
    for row in rows:
        line = sep.join(_fmt_val(row, h).ljust(col_widths[h]) for h in headers)
        print(line)
    print("=" * len(hdr_line))

    # ----- Write CSV -----
    if output_file:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                flat = {h: _fmt_val(row, h) for h in headers}
                writer.writerow(flat)
        print(f"\nComparison written to {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Analyze a single trace:     python main.py <trace.json> [--output report.txt]")
        print("  Compare multiple traces:    python main.py --compare trace1.json trace2.json [--output comparison.csv]")
        print("  Annotate trace with phases: python main.py --annotate <trace.json> [--output annotated.json]")
        sys.exit(1)

    if sys.argv[1] == "--annotate":
        trace_file = None
        output_file = None
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
                continue
            elif not sys.argv[i].startswith("--"):
                trace_file = sys.argv[i]
            i += 1
        if trace_file is None:
            print("Error: --annotate requires a trace file.")
            sys.exit(1)
        if output_file is None:
            base, ext = os.path.splitext(trace_file)
            output_file = f"{base}_annotated{ext}"
        _annotate_trace(trace_file, output_file)
        return

    if sys.argv[1] == "--compare":
        trace_files = []
        output_file = None
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--output":
                if i + 1 < len(sys.argv):
                    output_file = sys.argv[i + 1]
                    i += 2
                    continue
            elif not sys.argv[i].startswith("--"):
                trace_files.append(sys.argv[i])
            i += 1
        if len(trace_files) < 2:
            print("Error: --compare requires at least 2 trace files.")
            sys.exit(1)
        _compare_traces(trace_files, output_file)
        return

    trace_file = sys.argv[1]
    output_file = None
    if len(sys.argv) >= 3 and sys.argv[2] == "--output":
        output_file = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Loading trace from {trace_file}...")
    parser = TraceParser(trace_file)
    if not parser.load():
        sys.exit(1)
    print(f"Loaded {len(parser.cpu_events)} CPU, {len(parser.gpu_events)} GPU, {len(parser.memory_events)} memory events.")

    roots = parser.build_tree()
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)

    detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
    fsdp = detector.extract_fsdp_phases(roots)

    print(f"Detected {len(fsdp.units)} FSDP units.")
    report = Report(fsdp, roots, output_path=output_file)
    text, markers = report.generate_report()

    print(text)

    if output_file:
        print(f"\nReport written to {output_file}")


if __name__ == "__main__":
    main()
