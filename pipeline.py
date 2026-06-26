"""Orchestration — composes the FSDP/TP analysis pipeline for reuse across modes.
``main.py`` and ``--timeline`` inline the steps directly, but ``--compare``
needs a reusable ``process_trace`` that returns structured (agg, metrics, fsdp,
report, text).

Handles multi-step traces (8B TP = 3 ProfilerSteps, async TP = 1) by filtering
to the last step and sanitising optimiser nodes that leak from earlier steps.
``process_all_steps`` iterates over every ProfilerStep for multi-step comparison.
"""

from trace_parser import TraceParser, TraceParserHelper
from fsdp_detector import StandardFSDPDetector
from bottleneck_detector import Report


def process_trace(trace_file: str, model_config=None):
    """Full pipeline end-to-end.

    1. ``TraceParser.load`` → classify events, validate JSON structure
    2. ``build_tree`` → time-nested CPU event tree per (pid, tid)
    3. ``select_profiler_step`` → pick the last ProfilerStep, filter GPU events
    4. ``attribute_gpu_kernel_with_logical_operation`` → ext-ID + time-overlap
    5. ``attribute_memory`` → [memory] delta propagation (no-op on reference traces)
    6. ``StandardFSDPDetector.extract_fsdp_phases`` → 7 sub-detectors for FSDP2/TP
    7. ``sanitize_optimizer`` → prune Optimizer.step/zero_grad from earlier steps
    8. ``Report.generate_report`` → per-layer Metrics + text report + JSON markers

    Returns ``(aggregated, metrics_list, fsdp, report, text)`` or ``None`` on load failure.
    """
    parser = TraceParser(trace_file)
    if not parser.load():
        return None

    roots = parser.build_tree()
    # Attribute GPU kernels BEFORE filtering to the active step — GPU kernels for
    # early layers' backward can finish hundreds of ms after the CPU ProfilerStep
    # ends (pipelined execution).  Attributing against the full GPU event set
    # ensures their ext-ID correlation succeeds even after the step-level filter
    # discards the events themselves.
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)

    step_start, step_end, filter_start, filter_end = select_profiler_step(roots, parser)

    # Prune GPU kernels from CPU tree nodes that fall outside the filtered
    # time range.  Attribution ran on all events (ensuring ext-ID matching),
    # but phase GPU time functions (``_phase_gpu_time*``) walk these lists
    # without any time-range awareness — they would count kernels from
    # earlier ProfilerSteps otherwise.
    if filter_start is not None:
        for node in TraceParserHelper.iter_nodes(roots):
            if not node.direct_gpu_kernels:
                continue
            kept = [ev for ev in node.direct_gpu_kernels
                    if ev.get('ts', 0) >= filter_start
                    and ev.get('ts', 0) + ev.get('dur', 0) <= filter_end]
            if len(kept) != len(node.direct_gpu_kernels):
                node.direct_gpu_kernels = kept
                node.gpu_duration = sum(k.get('dur', 0) for k in kept if k.get('dur', 0) > 0)
                node.direct_gpu_duration = node.gpu_duration

    # Compute AG per-layer AFTER step filtering so that the time range
    # filter excludes all-gather kernels from earlier ProfilerSteps.
    # Must still be BEFORE extract_fsdp_phases because
    # _detect_fsdp_gpu_fallback duplicates AG kernels onto unit phase
    # nodes, polluting the ancestry-based classification.
    from bottleneck_detector import _compute_ag_per_layer
    ag_range = (filter_start, filter_end) if filter_start is not None else None
    ag_per_layer = _compute_ag_per_layer(roots, time_range=ag_range)

    detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
    fsdp = detector.extract_fsdp_phases(roots)
    sanitize_optimizer(fsdp, step_start, step_end)

    from trace_annotator import _get_ac2g_bwd_supplement, _find_profiler_steps
    all_steps = _find_profiler_steps(trace_file)
    step_index = len(all_steps) - 1 if len(all_steps) > 1 else 0
    num_steps = len(all_steps) if all_steps else 1
    ac2g_supplement = _get_ac2g_bwd_supplement(
        parser.all_events, step_index, num_steps,
    )
    for unit in fsdp.units:
        unit.ag_bwd_supplement_us = ac2g_supplement.get(unit.layer_name, 0.0)

    report = Report(fsdp, roots, output_path=None, model_config=model_config,
                    ag_per_layer=ag_per_layer)
    text, markers = report.generate_report()

    return report.aggregated, report.metrics_list, fsdp, report, text


def process_all_steps(trace_file: str, model_config=None):
    """Analyse every ProfilerStep in the trace and return one result per step.

    *model_config* (optional ``ModelConfig``) enables MFU/HFU/tokens-per-second
    computation in each step's report.

    Returns a list of ``(step_name, aggregated, metrics_list, fsdp, report, text)``
    tuples, one per ProfilerStep.  Handles single-step traces gracefully.
    """
    from trace_annotator import (_find_profiler_steps, _filter_gpu_events,
                                 _prune_roots_for_step, _get_ac2g_bwd_supplement)

    steps = _find_profiler_steps(trace_file)
    if not steps:
        result = process_trace(trace_file, model_config=model_config)
        if result is None:
            return []
        _, metrics_list, fsdp, report, text = result
        return [("Trace", report.aggregated, metrics_list, fsdp, report, text)]

    num_steps = len(steps)
    results = []
    for step_idx, (step_name, step_start, step_end, _) in enumerate(steps):
        parser = TraceParser(trace_file)
        if not parser.load():
            continue
        _filter_gpu_events(parser, step_start, step_end)
        roots = parser.build_tree()
        roots = _prune_roots_for_step(roots, step_name, step_start, step_end)
        parser.attribute_gpu_kernel_with_logical_operation(roots)
        parser.attribute_memory(roots)

        from bottleneck_detector import _compute_ag_per_layer
        ag_per_layer = _compute_ag_per_layer(roots, time_range=(step_start, step_end))

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

        ac2g_supplement = _get_ac2g_bwd_supplement(
            parser.all_events, step_idx, num_steps,
        )
        for unit in fsdp.units:
            unit.ag_bwd_supplement_us = ac2g_supplement.get(unit.layer_name, 0.0)

        report = Report(fsdp, roots, output_path=None, model_config=model_config,
                        ag_per_layer=ag_per_layer)
        text, markers = report.generate_report()
        sname = step_name.replace("ProfilerStep#", "Step#")
        results.append((sname, report.aggregated, report.metrics_list, fsdp, report, text))

    return results


def select_profiler_step(roots, parser):
    """Filter GPU/memory events to only the last ProfilerStep's time range.

    PyTorch profiler captures multiple ProfilerStep#N markers across training
    iterations (3 in the 8B TP trace: #1, #2, #3).  The detector analyses the
    last completed step only.  Without this filter, GPU events from earlier steps
    would pollute the last step's attribution — especially problematic for ext-ID
    correlation where GPU events from Step#1 carry ``External id`` values that
    collide with Step#3 CPU events.

    Falls back to ``(None, None, None, None)`` when fewer than 2 steps are found
    (single-step traces like async TP's ProfilerStep#9) — caller should still
    proceed with full time range.

    Returns ``(step_start, step_end, filter_start, filter_end)``:
    * ``step_start`` / ``step_end`` — the strict CPU ProfilerStep boundaries.
    * ``filter_start`` / ``filter_end`` — the margin-extended bounds used for
      GPU event filtering.  Pass these to ``_compute_ag_per_layer(time_range=...)``
      so that all GPU-time comparisons use the same event set.
    """
    step_events = []
    for root in roots:
        for node in TraceParserHelper.iter_nodes([root]):
            if node.name.startswith('ProfilerStep#') and node.raw_event:
                pid = node.raw_event.get('pid', 0)
                step_events.append((node.name, pid, node.start_time, node.end_time, node))

    if not step_events:
        print("  No ProfilerStep markers found — analysing full trace time range.")
        return None, None, None, None

    from collections import defaultdict
    by_name = defaultdict(list)
    for name, pid, start, end, node in step_events:
        by_name[name].append((pid, start, end, node))

    latest_per_step = []
    for name, occurrences in by_name.items():
        occurrences.sort(key=lambda x: -x[0])
        pid, start, end, node = occurrences[0]
        latest_per_step.append((name, pid, start, end, node))

    latest_per_step.sort(key=lambda x: x[2])
    if len(latest_per_step) <= 1:
        print("  Single ProfilerStep only — using full trace time range.")
        return None, None, None, None

    last_name, last_pid, last_start, last_end, last_node = latest_per_step[-1]
    step_labels = [c[0].replace('ProfilerStep#', '#') for c in latest_per_step]
    print(f"  {len(latest_per_step)} ProfilerSteps detected: {{{','.join(step_labels)}}}, using last: #{last_name.split('#')[-1]}")

    # 5% margin around the step to catch deferred GPU kernels (e.g. fused ADAMW
    # that executes ~727ms after the CPU ProfilerStep ends in the 8B trace).
    step_dur = last_end - last_start
    margin = max(step_dur * 0.05, 2000.0)
    filter_start = last_start - margin
    filter_end = last_end + margin
    parser.gpu_events = [ev for ev in parser.gpu_events
                         if ev.get('ts', 0) >= filter_start
                         and ev.get('ts', 0) + ev.get('dur', 0) <= filter_end]
    parser.memory_events = [ev for ev in parser.memory_events
                            if filter_start <= ev.get('ts', 0) <= filter_end]

    return last_start, last_end, filter_start, filter_end


def sanitize_optimizer(fsdp, step_start=None, step_end=None):
    """Prune optimizer_step/zero_grad nodes outside the active step's time range.

    The tree has Optimizer.step#N nodes from all ProfilerSteps (3 in the 8B trace).
    Without this filter, the report would sum GPU time from all three optimiser
    steps, over-counting the last step's optimiser cost by 3×.

    NB: Fused ADAMW GPU kernels that execute *after* the CPU ProfilerStep ends
    (the ~727ms deferral in the 8B trace) are still collected if their CPU node
    falls within the margin.  The GPU time margin filter in ``select_profiler_step``
    is the main mechanism; this function only prunes the CPU node list.
    """
    if step_start is None or step_end is None:
        return
    fsdp.optimizer_step = [n for n in fsdp.optimizer_step
                           if step_start <= n.start_time <= step_end]
    fsdp.optimizer_zero_grad = [n for n in fsdp.optimizer_zero_grad
                                if step_start <= n.start_time <= step_end]
