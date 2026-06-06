"""Pipeline orchestration for PPT trace analysis.

Handles ProfilerStep selection, optimizer sanitisation,
and the full analysis pipeline (parse → build → attribute → detect → report).
"""

from trace_parser import TraceParser, TraceParserHelper
from fsdp_detector import StandardFSDPDetector
from bottleneck_detector import Report


def process_trace(trace_file: str):
    """Run the full pipeline on one trace.

    Returns ``(aggregated, metrics_list, fsdp, report, text)`` or ``None``.
    """
    parser = TraceParser(trace_file)
    if not parser.load():
        return None

    roots = parser.build_tree()
    step_bounds = select_profiler_step(roots, parser)
    step_start, step_end = step_bounds
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)

    detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
    fsdp = detector.extract_fsdp_phases(roots)
    sanitize_optimizer(fsdp, step_start, step_end)

    report = Report(fsdp, roots, output_path=None)
    text, markers = report.generate_report()

    return report.aggregated, report.metrics_list, fsdp, report, text


def select_profiler_step(roots, parser):
    """Filter GPU/memory events to only the last ProfilerStep's time range.

    Multiple ProfilerStep#N events (e.g. 3 training steps) exist on
    different threads in the tree.  The detector's phase-picking logic
    (latest by start_time) already selects data from the last step, but
    GPU/memory events from all steps would be attributed.  This function
    filters those to the last step's interval.

    Returns ``(step_start, step_end)`` or ``(None, None)`` if there are
    fewer than two ProfilerSteps.
    """
    step_events = []
    for root in roots:
        for node in TraceParserHelper.iter_nodes([root]):
            if node.name.startswith('ProfilerStep#') and node.raw_event:
                pid = node.raw_event.get('pid', 0)
                step_events.append((node.name, pid, node.start_time, node.end_time, node))

    if not step_events:
        return None, None

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
        return None, None

    last_name, last_pid, last_start, last_end, last_node = chosen[-1]
    step_labels = [c[0].replace('ProfilerStep#', '#') for c in chosen]
    print(f"  {len(chosen)} ProfilerSteps detected: {{{','.join(step_labels)}}}, using last: #{last_name.split('#')[-1]}")

    step_dur = last_end - last_start
    margin = max(step_dur * 0.05, 2000.0)
    parser.gpu_events = [ev for ev in parser.gpu_events
                         if ev.get('ts', 0) >= last_start - margin
                         and ev.get('ts', 0) + ev.get('dur', 0) <= last_end + margin]
    parser.memory_events = [ev for ev in parser.memory_events
                            if last_start - margin <= ev.get('ts', 0) <= last_end + margin]

    return last_start, last_end


def sanitize_optimizer(fsdp, step_start=None, step_end=None):
    """Filter optimizer_step/-zero_grad to only events within the step's time range."""
    if step_start is None or step_end is None:
        return
    fsdp.optimizer_step = [n for n in fsdp.optimizer_step
                           if step_start <= n.start_time <= step_end]
    fsdp.optimizer_zero_grad = [n for n in fsdp.optimizer_zero_grad
                                if step_start <= n.start_time <= step_end]
