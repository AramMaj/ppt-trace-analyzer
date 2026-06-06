"""

"""
import sys
import csv
import os
from trace_parser import TraceParser
from fsdp_detector import StandardFSDPDetector
from bottleneck_detector import Report, Metrics, Bottlenecks, _format_us


def _process_trace(trace_file: str):
    """Run the full pipeline on one trace. Returns (aggregated, metrics_list, fsdp, report_text) or None."""
    parser = TraceParser(trace_file)
    if not parser.load():
        return None

    roots = parser.build_tree()
    roots = _select_profiler_step(roots, parser)
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)

    detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
    fsdp = detector.extract_fsdp_phases(roots)
    _sanitize_optimizer(fsdp)

    report = Report(fsdp, roots, output_path=None)
    text, markers = report.generate_report()

    return report.aggregated, report.metrics_list, fsdp, report, text


_profiler_step_cache = None


def _select_profiler_step(roots, parser):
    """Filter GPU/memory events to only the last ProfilerStep's time range.

    Multiple ProfilerStep#N events (e.g. 3 training steps) exist on
    different threads in the tree.  The detector's phase-picking logic
    (latest by start_time) already selects data from the last step, but
    GPU/memory events from all steps would be attributed.  This function
    filters those to the last step's interval and also sanitises the
    optimizer-step list afterwards.
    """
    global _profiler_step_cache
    if _profiler_step_cache is not None:
        return _profiler_step_cache

    from trace_parser import TraceParserHelper

    step_events = []
    for root in roots:
        for node in TraceParserHelper.iter_nodes([root]):
            if node.name.startswith('ProfilerStep#') and node.raw_event:
                pid = node.raw_event.get('pid', 0)
                step_events.append((node.name, pid, node.start_time, node.end_time, node))

    if not step_events:
        _profiler_step_cache = (roots, None, None, None)
        return roots

    from collections import defaultdict
    by_name = defaultdict(list)
    for name, pid, start, end, node in step_events:
        by_name[name].append((pid, start, end, node))

    chosen = []
    for name, variants in by_name.items():
        variants.sort(key=lambda x: -x[0])
        pid, start, end, node = variants[0]
        chosen.append((name, pid, start, end, node))

    chosen.sort(key=lambda x: x[2])
    if len(chosen) <= 1:
        _profiler_step_cache = (roots, None, None, None)
        return roots

    last_name, last_pid, last_start, last_end, last_node = chosen[-1]
    step_labels = [c[0].replace('ProfilerStep#', '#') for c in chosen]
    print(f"  {len(chosen)} ProfilerSteps detected: {{{','.join(step_labels)}}}, using last: #{last_name.split('#')[-1]}")

    # Filter GPU events to this step's time range (with margin for async overlap)
    step_dur = last_end - last_start
    margin = max(step_dur * 0.05, 2000.0)
    parser.gpu_events = [ev for ev in parser.gpu_events
                         if ev.get('ts', 0) >= last_start - margin
                         and ev.get('ts', 0) + ev.get('dur', 0) <= last_end + margin]
    parser.memory_events = [ev for ev in parser.memory_events
                            if last_start - margin <= ev.get('ts', 0) <= last_end + margin]

    _profiler_step_cache = (roots, last_start, last_end, last_name)
    return roots


def _sanitize_optimizer(fsdp):
    """Filter optimizer_step/-zero_grad to only events within the last step's time range."""
    global _profiler_step_cache
    if _profiler_step_cache is None:
        return
    roots, step_start, step_end, step_name = _profiler_step_cache
    if step_start is None or step_end is None:
        return
    fsdp.optimizer_step = [n for n in fsdp.optimizer_step
                           if step_start <= n.start_time <= step_end]
    fsdp.optimizer_zero_grad = [n for n in fsdp.optimizer_zero_grad
                                if step_start <= n.start_time <= step_end]


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


TIMELINE_BAR_CHARS = {'AG fwd': 'A', 'Fwd cmp': 'F', 'AG bwd': 'a', 'Bwd cmp': 'B', 'RS': 'R', 'Optimizer': 'O'}


def _phase_wall_span(nodes):
    if not nodes:
        return None
    start = min(n.start_time for n in nodes)
    end = max(n.end_time for n in nodes)
    return (start, end)


def _get_phase_spans(unit):
    """Return list of (label, start, end) for non-empty phases in a unit."""
    phases = []
    phase_src = [
        ('AG fwd', unit.all_gather_fwd),
        ('Fwd cmp', unit.fwd_compute),
        ('AG bwd', unit.all_gather_bwd),
        ('Bwd cmp', unit.bwd_compute),
        ('RS', unit.reduce_scatter),
    ]
    for label, nodes in phase_src:
        span = _phase_wall_span(nodes)
        if span is not None:
            phases.append((label, span[0], span[1]))
    return phases


def _print_timeline(fsdp, report):
    """Print a compact ASCII Gantt chart showing detected phases per layer over time."""
    n_layers = len(fsdp.units)
    if n_layers == 0:
        print("  No FSDP units to show.")
        return

    # Collect all phase intervals
    intervals = []  # (layer_idx, label, start, end)
    for idx, unit in enumerate(fsdp.units):
        for label, nodes in [
            ('AG fwd', unit.all_gather_fwd),
            ('Fwd cmp', unit.fwd_compute),
            ('AG bwd', unit.all_gather_bwd),
            ('Bwd cmp', unit.bwd_compute),
            ('RS', unit.reduce_scatter),
        ]:
            span = _phase_wall_span(nodes)
            if span:
                intervals.append((idx, label, span[0], span[1]))

    if not intervals:
        print("  No phase intervals found.")
        return

    t_min = min(i[2] for i in intervals)
    t_max = max(i[3] for i in intervals)
    span_total = t_max - t_min
    if span_total <= 0:
        print("  Zero-length timeline.")
        return

    # Terminal width: 100 chars for bars + label
    BAR_WIDTH = 80
    COL_WIDTH = max(len(u.layer_name) for u in fsdp.units) + 1

    # Print time scale header
    print()
    print("─" * (COL_WIDTH + BAR_WIDTH + 4))
    print("Phase Timeline (each character ≈ {:.1f}ms)".format(span_total / (BAR_WIDTH * 1000)))
    print("─" * (COL_WIDTH + BAR_WIDTH + 4))

    scale_chars = BAR_WIDTH
    scale_lines = []
    num_ticks = 5
    for t in range(num_ticks + 1):
        pct = t / num_ticks
        ts = t_min + pct * span_total
        pos = int(pct * scale_chars)
        scale_lines.append((pos, _format_us(ts)))
    scale_str = ""
    prev_end = 0
    for pos, label in scale_lines:
        gap = pos - prev_end
        if gap <= 0:
            continue
        if len(label) <= gap:
            scale_str += " " * (gap - len(label)) + label
        else:
            scale_str += " " * gap
        prev_end = pos
    print(" " * COL_WIDTH + scale_str)

    # Print each layer's timeline
    for idx, unit in enumerate(fsdp.units):
        name = unit.layer_name
        line_chars = [' '] * BAR_WIDTH

        for label, start, end in [(l, s, e) for (li, l, s, e) in intervals if li == idx]:
            col_start = max(0, int((start - t_min) / span_total * BAR_WIDTH))
            col_end = min(BAR_WIDTH - 1, int((end - t_min) / span_total * BAR_WIDTH))
            ch = TIMELINE_BAR_CHARS.get(label, '#')
            for c in range(col_start, min(col_end + 1, BAR_WIDTH)):
                line_chars[c] = ch

        timeline_str = ''.join(line_chars)
        # Add bottleneck marker
        for m in report.metrics_list:
            if m.layer_name == unit.layer_name:
                issues = Bottlenecks.detect(m)
                if issues:
                    phases = _get_phase_spans(unit)
                    if phases:
                        longest = max(phases, key=lambda p: p[2] - p[1])
                        bpos = int((longest[1] - t_min) / span_total * BAR_WIDTH)
                        if 0 <= bpos < BAR_WIDTH:
                            # Mark with '!' if not already occupied
                            if timeline_str[bpos] == ' ':
                                timeline_str = timeline_str[:bpos] + '!' + timeline_str[bpos + 1:]
                break

        print(f"{name:<{COL_WIDTH}} {timeline_str}")

    # Legend
    print("─" * (COL_WIDTH + BAR_WIDTH + 4))
    legend_parts = [f"{ch}={name}" for name, ch in TIMELINE_BAR_CHARS.items()]
    legend_parts.append("!=bottleneck")
    print(" " * COL_WIDTH + "  ".join(legend_parts))

    # Bottleneck details
    bneck_count = 0
    for m in report.metrics_list:
        issues = Bottlenecks.detect(m)
        if issues:
            bneck_count += 1
            if bneck_count <= 5:
                print(f"  {m.layer_name}: {'; '.join(issues)}")
    if bneck_count > 5:
        print(f"  ... and {bneck_count - 5} more layers with bottlenecks")
    if bneck_count == 0:
        print("  No bottlenecks detected")
    print()


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
        print("  Text phase timeline:        python main.py --timeline <trace.json>")
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
        from trace_annotator import annotate_trace
        annotate_trace(trace_file, output_file)
        return

    if sys.argv[1] == "--timeline":
        trace_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
        output_file = None
        if "--output" in sys.argv:
            idx = sys.argv.index("--output")
            if idx + 1 < len(sys.argv):
                output_file = sys.argv[idx + 1]
        if trace_file is None:
            print("Error: --timeline requires a trace file.")
            sys.exit(1)
        parser = TraceParser(trace_file)
        if not parser.load():
            sys.exit(1)
        roots = parser.build_tree()
        roots = _select_profiler_step(roots, parser)
        parser.attribute_gpu_kernel_with_logical_operation(roots)
        parser.attribute_memory(roots)
        detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
        fsdp = detector.extract_fsdp_phases(roots)
        _sanitize_optimizer(fsdp)
        report = Report(fsdp, roots, output_path=None)
        report.generate_report()
        _print_timeline(fsdp, report)
        if output_file:
            print(f"  (use --annotate to write annotated trace to {output_file})")
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
    roots = _select_profiler_step(roots, parser)
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)

    detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
    fsdp = detector.extract_fsdp_phases(roots)
    _sanitize_optimizer(fsdp)

    print(f"Detected {len(fsdp.units)} FSDP units.")
    report = Report(fsdp, roots, output_path=output_file)
    text, markers = report.generate_report()

    print(text)

    if output_file:
        print(f"\nReport written to {output_file}")


if __name__ == "__main__":
    main()
