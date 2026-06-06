"""CLI entry point.  Dispatches to four modes:

  Default     — per-layer GPU time breakdown + bottleneck report
  --timeline  — ASCII Gantt chart of FSDP2 pipeline stagger
  --compare   — side-by-side benchmark table (2+ traces)
  --annotate  — Chrome Trace JSON with phases, flows, counters, bottlenecks

Each mode delegates to dedicated modules (pipeline.py, timeline.py,
comparison.py, trace_annotator.py) — ``main.py`` only parses argv and prints.
"""

import sys
import os

from trace_parser import TraceParser
from fsdp_detector import StandardFSDPDetector
from bottleneck_detector import Report

from pipeline import process_trace, select_profiler_step, sanitize_optimizer
from timeline import print_timeline
from comparison import compare_traces


def main():
    """Route to single-trace report, --timeline Gantt, --compare multi-trace, or --annotate chrome://tracing.
    No argparse dependency — manual argv parsing keeps the import footprint small.
    """
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
        if trace_file is None:
            print("Error: --timeline requires a trace file.")
            sys.exit(1)
        parser = TraceParser(trace_file)
        if not parser.load():
            sys.exit(1)
        roots = parser.build_tree()
        step_start, step_end = select_profiler_step(roots, parser)
        parser.attribute_gpu_kernel_with_logical_operation(roots)
        parser.attribute_memory(roots)
        detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
        fsdp = detector.extract_fsdp_phases(roots)
        sanitize_optimizer(fsdp, step_start, step_end)
        report = Report(fsdp, roots, output_path=None)
        report.generate_report()
        print_timeline(fsdp, report)
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
        compare_traces(trace_files, output_file)
        return

    # Default: single-trace analysis
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
    step_start, step_end = select_profiler_step(roots, parser)
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)

    detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
    fsdp = detector.extract_fsdp_phases(roots)
    sanitize_optimizer(fsdp, step_start, step_end)

    print(f"Detected {len(fsdp.units)} FSDP units.")
    report = Report(fsdp, roots, output_path=output_file)
    text, markers = report.generate_report()

    print(text)

    if output_file:
        print(f"\nReport written to {output_file}")


if __name__ == "__main__":
    main()
