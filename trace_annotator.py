"""Chrome Trace annotation generator — appends FSDP/TP phase spans, flow arrows
between consecutive all-gather→compute→reduce-scatter, GPU activity counters
(number of concurrently-active layers), and per-layer bottleneck markers so the
pipeline stagger, overlap bubbles, and async TP overlap can be inspected in
``chrome://tracing``.

The annotated trace merges all ProfilerSteps into a single PID=9999 with:
  - One TID per FSDP layer (layers 0..33 for 8B, 0..7 for async TP)
  - One TID for Optimizer steps (all steps combined)
  - One TID for TP collectives (mesh_tp kernels, all steps combined)
  - One TID for GPU activity counter (active-layer count via ph: 'C')
  - One TID for bottleneck markers (ph: 'i' instant events)

Tested against both reference traces — the output JSON validates in
``chrome://tracing`` with all event types present and matched flow pairs.
"""

import json
from typing import List, Dict, Optional, Tuple

from trace_parser import TraceParser
from fsdp_detector import StandardFSDPDetector
from bottleneck_detector import Report, Bottlenecks, ModelConfig, _phase_gpu_time, _phase_wall_span

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
    "Bwd AG": "#E91E63",
    "Bwd cmp": "#F44336",
    "RS": "#9C27B0",
    "Pre-fwd AG": "#795548",
    "Prefetch AG": "#00BCD4",
    "Trailing RS": "#607D8B",
    "Optimizer": "#607D8B",
    "TP": "#FFEB3B",
}

PHASE_LABELS = ["AG fwd", "Fwd cmp", "AG bwd", "Bwd AG", "Bwd cmp", "RS",
                "Pre-fwd AG", "Prefetch AG", "Trailing RS"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_ac2g_ag_bwd_spans(raw_events: List[dict]):
    """Parse ac2g flow events + gpu_user_annotation to identify all-gather
    copy-out GPU spans on dedicated CUDA streams (e.g. stream 18 in gpu-server).

    Uses only streams where the ``FSDP::all_gather`` annotations DO NOT overlap
    with GEMM compute kernels (i.e. streams dedicated to all-gather copy-out).
    Returns ``{layer_name: [(gpu_start, gpu_end, kernel_dur_us), …]}`` per layer,
    or an empty dict when no ac2g annotations are present.
    """
    # 1. Find streams that carry ac2g flow-step events
    ac2g_streams = set()
    for ev in raw_events:
        if ev.get('cat') == 'ac2g' and ev.get('ph') == 'f':
            ac2g_streams.add(ev['tid'])

    if not ac2g_streams:
        return {}

    # 2. On ac2g streams, find FSDP::all_gather annotations with a layer name
    import re
    layer_pat = re.compile(r'model\.layers\.(\d+)')
    annotations = []  # (tid, ts, end, layer_name)
    for ev in raw_events:
        tid = ev.get('tid', 0)
        if tid not in ac2g_streams:
            continue
        if ev.get('cat') == 'gpu_user_annotation' and ev.get('ph') == 'X':
            name = ev.get('name', '')
            if 'FSDP::all_gather' not in name:
                continue
            m = layer_pat.search(name)
            if not m:
                continue
            ts = ev.get('ts', 0)
            dur = ev.get('dur', 0)
            annotations.append((tid, ts, ts + dur, f'model.layers.{m.group(1)}'))

    if not annotations:
        return {}

    # 3. Classify streams as "pure" (only copy-out kernels, no GEMM compute)
    ann_streams = {tid for tid, _, _, _ in annotations}
    pure_streams = set()
    for tid in ann_streams:
        kernels_on_stream = [
            ev for ev in raw_events
            if ev.get('tid') == tid and ev.get('cat') == 'kernel'
            and 'nccl' not in ev.get('name', '').lower()
        ]
        if not kernels_on_stream:
            pure_streams.add(tid)
            continue
        has_gemm = any(
            'gemm' in ev.get('name', '').lower()
            or '16816' in ev.get('name', '').lower()
            for ev in kernels_on_stream
        )
        if not has_gemm:
            pure_streams.add(tid)

    if not pure_streams:
        return {}

    # 4. For each annotation on a *pure* stream, record the annotation span
    #    and contained GPU kernel durations.
    result = {}
    for tid, ann_ts, ann_end, layer_name in annotations:
        if tid not in pure_streams:
            continue
        kernel_total = 0
        for ev in raw_events:
            if ev.get('tid') != tid:
                continue
            if ev.get('cat') != 'kernel':
                continue
            if 'nccl' in ev.get('name', '').lower():
                continue
            k_ts = ev.get('ts', 0)
            k_end = k_ts + ev.get('dur', 0)
            if k_ts >= ann_ts and k_end <= ann_end:
                kernel_total += ev.get('dur', 0)
        result.setdefault(layer_name, []).append((ann_ts, ann_end, kernel_total))

    return result


def _filter_ac2g_spans_by_step(ac2g_layer_spans, step_index, num_steps):
    """Pick the AG bwd (backward) copy-out annotation for *step_index*.

    Each layer has ``num_steps × 2`` annotations (fwd copy-out + bwd copy-out
    per step), ordered by timestamp.  For AG bwd we want the SECOND (larger,
    later) annotation — the backward all-gather copy-out.
    Returns a new dict with only the kernels from the bwd annotation.
    """
    if num_steps <= 1:
        return ac2g_layer_spans

    filtered = {}
    for layer_name, spans in ac2g_layer_spans.items():
        sorted_spans = sorted(spans, key=lambda x: x[0])
        # Each layer: num_steps steps × 2 per step = fwd + bwd
        # Pick the second (bwd) annotation for this step
        bwd_idx = step_index * 2 + 1
        if bwd_idx < len(sorted_spans):
            # Return just the kernel spans from this single annotation
            filtered[layer_name] = [sorted_spans[bwd_idx]]
    return filtered


def _collect_ac2g_gpu_spans(layer_name: str,
                            ac2g_layer_spans: Dict[str, list]):
    """Return ac2g-identified GPU span boundaries for *layer_name*.
    Returns ``[(gpu_start, gpu_end), …]`` or ``[]``.
    """
    raw = ac2g_layer_spans.get(layer_name, [])
    return [(s, e) for s, e, _ in raw]


def _total_ac2g_kernel_us(layer_name: str,
                          ac2g_layer_spans: Dict[str, list]) -> float:
    """Sum of GPU kernel execution times within ac2g spans for *layer_name*."""
    raw = ac2g_layer_spans.get(layer_name, [])
    return sum(kdur for _, _, kdur in raw)


def _get_ac2g_bwd_supplement(raw_events, step_index: int = 0,
                              num_steps: int = 1):
    """Compute all-gather backward copy-out GPU kernel supplement from ac2g events.
    Returns ``{layer_name: total_kernel_us}`` — the kernel time on pure copy-out
    streams (e.g. stream 18) that is MISSING from CPU-tree-based ``ag_bwd_gpu``.

    For each layer, selects only the *second* annotation per step (bwd copy-out,
    which is always later and larger than the fwd copy-out).
    """
    all_spans = _find_ac2g_ag_bwd_spans(raw_events)
    if not all_spans:
        return {}
    result = {}
    for layer_name, spans in all_spans.items():
        sorted_spans = sorted(spans, key=lambda x: x[0])
        if num_steps <= 1:
            # Single step: 2 annotations (fwd, bwd). Take the bwd (last).
            pick = sorted_spans[-1:] if len(sorted_spans) >= 2 else sorted_spans
        else:
            # Multi-step: filter by step_index.
            bwd_idx = step_index * 2 + 1
            pick = [sorted_spans[bwd_idx]] if bwd_idx < len(sorted_spans) else []
        total = sum(kernel_us for _, _, kernel_us in pick)
        if total > 0:
            result[layer_name] = total
    return result

def _collect_nccl_gpu_spans(node):
    """Recursively collect only NCCL-kernel GPU spans from *node*.

    Used for AG / RS comm phases so the bar shows just the collective
    (not copy-out/in kernels that precede or follow it on the GPU).
    """
    spans = []
    for gpu in node.direct_gpu_kernels:
        if 'nccl' in gpu.get('name', '').lower():
            spans.append((gpu['ts'], gpu['ts'] + gpu.get('dur', 0)))
    for ch in node.children:
        spans.extend(_collect_nccl_gpu_spans(ch))
    return spans


def _collect_child_gpu_spans(node):
    """Recursively collect all GPU spans (excluding NCCL) from *node*.

    Needed for AG bwd, where the copy-out kernel lives on a child
    (``fsdp::split_with_sizes_copy``) rather than the wrapper node.
    """
    spans = []
    for gpu in node.direct_gpu_kernels:
        if 'nccl' in gpu.get('name', '').lower():
            continue
        spans.append((gpu['ts'], gpu['ts'] + gpu.get('dur', 0)))
    for ch in node.children:
        spans.extend(_collect_child_gpu_spans(ch))
    return spans


def _collect_nccl_from_nodes(node, inside_ag=False):
    """Recursively collect NCCL GPU spans under ``FSDP::all_gather`` subtrees.

    Only collects NCCL kernels that live in the subtree of an
    ``FSDP::all_gather`` node (backward prefetch all-gather), skipping
    TP collectives (``_c10d_functional::*``).  Duplicate GPU events
    (same kernel referenced from multiple CPU nodes) are deduplicated.
    """
    spans = []
    is_tp_collective = node.name.startswith('_c10d_functional::')
    is_fsdp_ag = 'FSDP::all_gather' in node.name and 'copy_out' not in node.name
    if is_tp_collective:
        return spans
    if inside_ag:
        for gpu in node.direct_gpu_kernels:
            if 'nccl' in gpu.get('name', '').lower():
                spans.append((gpu['ts'], gpu['ts'] + gpu.get('dur', 0)))
    new_inside = inside_ag or is_fsdp_ag
    for ch in node.children:
        spans.extend(_collect_nccl_from_nodes(ch, inside_ag=new_inside))
    seen = set()
    deduped = []
    for s, e in spans:
        key = (s, e)
        if key not in seen:
            seen.add(key)
            deduped.append((s, e))
    return deduped


def _gpu_kernel_span(nodes, mode):
    """GPU-domain span for one phase.

    *mode*:
      ``'nccl'`` — only NCCL kernels, recursive (AG fwd, RS).
      ``'child'`` — non-NCCL, recursive (AG bwd).
      ``'compute'`` — non-NCCL, flat on the nodes themselves (Fwd cmp, Bwd cmp).
    Returns ``(gpu_start, gpu_end)`` or ``None``.
    """
    if not nodes:
        return None
    spans = []
    collector = {'nccl': _collect_nccl_gpu_spans,
                 'child': _collect_child_gpu_spans,
                 'nccl_from_nodes': _collect_nccl_from_nodes}
    collect_fn = collector.get(mode)
    if collect_fn is not None:
        for n in nodes:
            spans.extend(collect_fn(n))
    else:
        for n in nodes:
            nn = n.name.lower()
            if 'foreach_copy' in nn or 'all_gather_copy_in' in nn:
                continue
            for gpu in n.direct_gpu_kernels:
                if 'nccl' in gpu.get('name', '').lower():
                    continue
                spans.append((gpu.get('ts', 0), gpu.get('ts', 0) + gpu.get('dur', 0)))
    if not spans:
        return None
    return (min(s[0] for s in spans), max(s[1] for s in spans))


def _get_gpu_kernel_spans(unit, ac2g_layer_spans=None):
    """GPU-domain phase spans for all FSDP phases.

    AG fwd / RS → NCCL-only (``'nccl'``).  AG bwd → primary source is
    ac2g-identified copy-out GPU spans (``ac2g_layer_spans``), falling back
    to ``'child'`` (non-NCCL recursive) when ac2g is unavailable.
    Compute phases → non-NCCL flat (``'compute'``).
    """
    phases = []
    for label, node_list, mode in [
        ("AG fwd", unit.all_gather_fwd, 'nccl'),
        ("Fwd cmp", unit.fwd_compute, 'compute'),
        ("AG bwd", unit.all_gather_bwd, 'child'),
        ("Bwd AG", unit.bwd_compute, 'nccl_from_nodes'),
        ("Bwd cmp", unit.bwd_compute, 'compute'),
        ("RS", unit.reduce_scatter, 'nccl'),
    ]:
        span = _gpu_kernel_span(node_list, mode)
        if label == "AG bwd" and ac2g_layer_spans:
            ac2g_spans = _collect_ac2g_gpu_spans(unit.layer_name, ac2g_layer_spans)
            if ac2g_spans:
                # ac2g spans are the authoritative GPU source for AG bwd
                ac2g_start = min(s for s, _ in ac2g_spans)
                ac2g_end = max(e for _, e in ac2g_spans)
                span = (ac2g_start, ac2g_end)
        if span is not None:
            phases.append((label, span[0], span[1], node_list))
    return phases


def _get_unattributed_gpu_spans(fsdp, roots):
    """GPU-domain spans for NCCL kernels NOT in any unit's phase trees.

    Three categories are recognised:

    * **Pre-fwd AG** — initial all-gather inside ``FSDP::pre_forward``
      (without a layer suffix, runs before any layer's forward).
    * **Prefetch AG** — backward-prefetch all-gather inside
      ``FSDP::backward_prefetch``.
    * **Trailing RS** — reduce-scatter inside
      ``root_post_backward_callback`` / ``FSDP::post_backward_reduce``
      that extends past the step boundary.

    Returns ``[(category, gpu_start, gpu_end), …]``.
    """
    # Collect all ext_ids already covered by unit phase trees (recursive)
    def _collect_attributed_exts(fsdp):
        exts = set()
        def _walk(node):
            exts.update(node.external_ids)
            for g in node.direct_gpu_kernels:
                gext = g.get('args', {}).get('External id')
                if gext:
                    exts.add(gext)
            for ch in node.children:
                _walk(ch)
        for u in fsdp.units:
            for lst in (u.all_gather_fwd, u.all_gather_bwd, u.reduce_scatter,
                        u.fwd_compute, u.bwd_compute):
                for n in lst:
                    _walk(n)
        return exts

    attributed = _collect_attributed_exts(fsdp)

    # Find every NCCL kernel in the roots
    nccl_events = []  # [(ext_id, ts, end, ancestors_str, name)]
    def _scan(node, ancestors):
        for g in node.direct_gpu_kernels:
            if 'nccl' in g.get('name', '').lower():
                ext = g.get('args', {}).get('External id')
                nccl_events.append((ext, g['ts'], g['ts'] + g.get('dur', 0),
                                    ' '.join(ancestors), g.get('name', '')))
        for ch in node.children:
            _scan(ch, ancestors + [node.name])

    for root in roots:
        _scan(root, [root.name])

    # Categorise unattributed kernels by ancestor path
    buckets = {}
    for ext_id, ts, end, path, name in nccl_events:
        if ext_id and ext_id in attributed:
            continue
        path_lower = path.lower()
        if 'FSDP::pre_forward'.lower() in path_lower and \
           '(model' not in path and '(layers' not in path and '(layer' not in path:
            # Future-proofing: some traces use "layers", some "model.layers"
            if '(model' not in path and '(layers' not in path and '(layer' not in path:
                buckets.setdefault('Pre-fwd AG', []).append((ts, end))
            else:
                buckets.setdefault('Other', []).append((ts, end))
        elif 'FSDP::backward_prefetch'.lower() in path_lower or \
             'FSDP::pre_backward'.lower() in path_lower:
            # backward_prefetch may be nested inside pre_backward
            buckets.setdefault('Prefetch AG', []).append((ts, end))
        elif 'root_post_backward_callback'.lower() in path_lower or \
             'FSDP::post_backward_reduce'.lower() in path_lower:
            buckets.setdefault('Trailing RS', []).append((ts, end))
        else:
            buckets.setdefault('Other', []).append((ts, end))

    result = []
    for cat, spans in buckets.items():
        if cat == 'Other':
            for ts, end in spans:
                print(f"  [unattributed] Other NCCL: ts={ts:.1f} dur={end-ts:.1f}")
            continue
        if spans:
            s = min(sp[0] for sp in spans)
            e = max(sp[1] for sp in spans)
            result.append((cat, s, e))
    return result


def _get_phase_spans(unit):
    """Wall-clock (CPU) spans for each FSDP phase — used by the layer TID
    and the ASCII timeline so the phase ordering is always correct.
    The AG bwd span is clamped to ``unit.all_gather_bwd_end`` to prevent
    overlap with backward compute of the same layer.
    """
    phases = []
    phase_src = [
        ("AG fwd", unit.all_gather_fwd, None),
        ("Fwd cmp", unit.fwd_compute, unit.fwd_compute_span),
        ("AG bwd", unit.all_gather_bwd, None),
        ("Bwd cmp", unit.bwd_compute, unit.bwd_compute_span),
        ("RS", unit.reduce_scatter, None),
    ]
    for label, node_list, fallback_span in phase_src:
        span = _phase_wall_span(node_list)
        if span is None:
            span = fallback_span
        elif fallback_span is not None:
            span = (fallback_span[0], max(span[1], fallback_span[1]))
        if span is not None:
            if label == "AG bwd" and unit.all_gather_bwd_end is not None:
                span = (span[0], unit.all_gather_bwd_end)
            phases.append((label, span[0], span[1], node_list))
    return phases


def _find_profiler_steps(trace_file: str) -> List[Tuple[str, int, int, int]]:
    """Scan the raw trace JSON for ProfilerStep#N boundaries.
    Avoids loading the full parser for a simple metadata scan.
    Returns deduplicated ``(name, ts, end, tid)`` sorted by time.

    Validates the JSON structure: rejects files with missing ``traceEvents``
    or non-dict root.  In the 8B TP trace this finds 3 steps at pid=24529;
    the async TP trace has a single ProfilerStep#9.
    """
    try:
        with open(trace_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  Error loading trace for step detection: {e}")
        return []

    if not isinstance(data, dict):
        print(f"  Error: trace root is {type(data).__name__}, expected dict")
        return []

    events = data.get("traceEvents")
    if not isinstance(events, list):
        print("  Warning: traceEvents missing or not a list — cannot detect steps")
        return []

    steps: List[Tuple[str, int, int, int]] = []
    for ev in events:
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

    seen = set()
    unique = []
    for name, ts, end, tid in steps:
        key = (name, ts, end)
        if key not in seen:
            seen.add(key)
            unique.append((name, ts, end, tid))
    return unique


def _filter_gpu_events(parser, step_start: int, step_end: int):
    """Keep only GPU/memory events near the active step, with a 5% margin to catch deferred kernels.
    """
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
    """Prune the parsed root list to only the active ProfilerStep's subtree and
    any time-range-contained roots on other threads.

    NB: The 100 µs tolerance on start-time matching is brittle — if the CPU
    event doesn't align due to scheduling jitter the ProfilerStep root is lost.
    """
    profiler_root = None
    for root in roots:
        raw = root.raw_event or {}
        if raw.get("name", "") == step_name and raw.get("ph") == "X":
            if abs(root.start_time - step_start) < 100:
                profiler_root = root
                break

    pruned_roots = []
    if profiler_root is not None:
        pruned_roots.append(profiler_root)

    for root in roots:
        if root is profiler_root:
            continue
        if root.start_time >= step_start and root.end_time <= step_end:
            pruned_roots.append(root)

    return pruned_roots


# ---------------------------------------------------------------------------
# Step-level analysis
# ---------------------------------------------------------------------------

def _analyze_step(
    trace_file: str,
    step_name: str,
    step_start: int,
    step_end: int,
    model_config: Optional[ModelConfig] = None,
    *,  # keyword-only below
    step_index: int = 0,
    num_steps: int = 1,
):
    """Full pipeline for one ProfilerStep.

    Accepts an optional ``ModelConfig`` for MFU/HFU/tokens-per-second computation.
    Returns ``(fsdp, report, roots)`` or ``(None, None, None)``.
    """
    parser = TraceParser(trace_file)
    if not parser.load():
        return None, None, None

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

    ac2g_supplement = _get_ac2g_bwd_supplement(parser.all_events, step_index, num_steps)
    for unit in fsdp.units:
        unit.ag_bwd_supplement_us = ac2g_supplement.get(unit.layer_name, 0.0)

    report = Report(fsdp, roots, output_path=None, model_config=model_config)
    report.generate_report()

    return fsdp, report, roots


# ---------------------------------------------------------------------------
# Cross-step annotation generation (one TID per layer, steps go right)
# ---------------------------------------------------------------------------

def annotate_trace(trace_file: str, output_file: str, max_steps: int = 0,
                   model_config: Optional[ModelConfig] = None):
    """Analyse each ProfilerStep and append phase/flow/counter/bottleneck events
    to the Chrome Trace JSON.  All steps share PID=9999; each layer gets its
    own TID with phases from all steps flowing rightward on the same row.

    ``model_config`` enables MFU/HFU/tokens-per-second computation and adds
    ``compute_to_comm``, ``exposed_comm``, ``ag_overlap_eff`` to bottleneck
    marker args.  Leave as ``None`` to skip throughput metrics.

    Validated against both reference traces:
    - 8B TP: 3 steps, 34 layers → ~680 phases, ~540 flows, 2 TP, 3 opt, 34 bneck
    - async TP: 1 step, 8 layers → ~40 phases, ~32 flows, 285 TP, 1 opt, 8 bneck
    """
    # 1. Load original trace with validation
    try:
        with open(trace_file) as f:
            trace_json = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  Error loading trace: {e}")
        return

    if not isinstance(trace_json, dict):
        print(f"  Error: trace root is {type(trace_json).__name__}, expected dict")
        return

    if "traceEvents" not in trace_json:
        trace_json["traceEvents"] = []

    # 2. Find ProfilerSteps
    steps = _find_profiler_steps(trace_file)
    if not steps:
        steps = [("Trace", 0, 0, 0)]
        all_ts = [
            ev.get("ts", 0)
            for ev in trace_json.get("traceEvents", [])
            if ev.get("ph") == "X"
        ]
        if all_ts:
            steps[0] = ("Trace", min(all_ts), max(all_ts), 0)
        print("  No ProfilerStep markers found, annotating full trace as one step.")
    else:
        print(f"  Found {len(steps)} ProfilerStep(s)")

    if max_steps and max_steps < len(steps):
        steps = steps[-max_steps:]

    # 3a. Build ac2g-based all-gather copy-out GPU spans (across all steps)
    raw_events = trace_json.get("traceEvents", [])
    ac2g_layer_spans = _find_ac2g_ag_bwd_spans(raw_events)
    if ac2g_layer_spans:
        print(f"  Found {sum(len(v) for v in ac2g_layer_spans.values())} ac2g copy-out GPU "
              f"spans on {len(ac2g_layer_spans)} layers")

    # 3b. Analyse all steps
    num_steps = len(steps)
    step_analyses = []
    for step_idx, (step_name, step_start, step_end, _) in enumerate(steps):
        sname = step_name.replace("ProfilerStep#", "Step#")
        print(f"  Analysing {sname} …")
        fsdp, report, roots = _analyze_step(
            trace_file, step_name, step_start, step_end,
            model_config=model_config,
            step_index=step_idx, num_steps=num_steps,
        )
        if fsdp is None:
            continue
        step_analyses.append((step_name, fsdp, report, roots))

    if not step_analyses:
        print("  No steps could be analysed.")
        return

    max_layers = max(len(fsdp.units) for _, fsdp, _, _ in step_analyses)
    pid = PPT_PID_BASE
    annotations = []

    # -- Layer threads: all steps' phases on the layer's TID --------------
    for layer_idx in range(max_layers):
        tid = layer_idx + 1

        # Pick a step that has this layer for the thread name
        layer_name = f"Layer {layer_idx}"
        for _, fsdp, _, _ in step_analyses:
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

        for step_idx, (step_name, fsdp, _, _) in enumerate(step_analyses):
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
    if any(fsdp.optimizer_step for _, fsdp, _, _ in step_analyses):
        annotations.append({
            "ph": "M", "pid": pid, "tid": opt_tid,
            "name": "thread_name",
            "args": {"name": "Optimizer (all steps)"},
        })
        for step_name, fsdp, _, _ in step_analyses:
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
        for _, fsdp, _, _ in step_analyses
    )
    if has_tp:
        annotations.append({
            "ph": "M", "pid": pid, "tid": tp_tid,
            "name": "thread_name",
            "args": {"name": "TP collectives (all steps)"},
        })
        tp_kernels = []
        for step_name, fsdp, _, _ in step_analyses:
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
    for _, fsdp, _, _ in step_analyses:
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

    # -- GPU phase thread (GPU domain — overlaps GPU kernel events) --------
    gpu_comp_tid = max_layers + 5
    annotations.append({
        "ph": "M", "pid": pid, "tid": gpu_comp_tid,
        "name": "thread_name",
        "args": {"name": "GPU phases (all steps)"},
    })
    for step_idx, (step_name, fsdp, _, roots) in enumerate(step_analyses):
        step_ac2g = _filter_ac2g_spans_by_step(ac2g_layer_spans, step_idx, len(step_analyses))
        # Per-unit phase bars
        for layer_idx, unit in enumerate(fsdp.units):
            for label, start, end, nodes in _get_gpu_kernel_spans(unit, ac2g_layer_spans=step_ac2g):
                gpu_dur = _phase_gpu_time(nodes)
                ac2g_us = _total_ac2g_kernel_us(unit.layer_name, step_ac2g) if label == "AG bwd" else 0
                annotations.append({
                    "ph": "X", "pid": pid, "tid": gpu_comp_tid,
                    "ts": start, "dur": end - start,
                    "cat": PPT_CAT,
                    "name": f"{label} ({step_name}, Layer {layer_idx})",
                    "args": {
                        "phase": label,
                        "layer": unit.layer_name,
                        "gpu_us": round(gpu_dur + ac2g_us, 1),
                        "wall_us": round(end - start, 1),
                        "num_nodes": len(nodes),
                        "step": step_name,
                        "domain": "gpu",
                    },
                    "cname": PHASE_COLORS.get(label, "#888"),
                })

        # Cross-unit / unattributed phase bars
        for label, start, end in _get_unattributed_gpu_spans(fsdp, roots):
            annotations.append({
                "ph": "X", "pid": pid, "tid": gpu_comp_tid,
                "ts": start, "dur": end - start,
                "cat": PPT_CAT,
                "name": f"{label} ({step_name})",
                "args": {
                    "phase": label,
                    "layer": "-",
                    "gpu_us": round(end - start, 1),
                    "wall_us": round(end - start, 1),
                    "num_nodes": 0,
                    "step": step_name,
                    "domain": "gpu",
                },
                "cname": PHASE_COLORS.get(label, "#888"),
            })

    # -- Bottleneck markers (all steps merged) ----------------------------
    bneck_tid = max_layers + 3
    annotations.append({
        "ph": "M", "pid": pid, "tid": bneck_tid,
        "name": "thread_name",
        "args": {"name": "Bottlenecks (all steps)"},
    })
    for step_name, fsdp, report, _ in step_analyses:
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
            ctc = metric.compute_to_comm_ratio
            ctc_str = f"{ctc:.1f}x" if ctc != float('inf') else "inf"
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
                    "compute_to_comm": ctc_str,
                    "exposed_comm": round(metric.exposed_comm_fraction, 3),
                    "ag_overlap_eff": round(metric.ag_fwd_overlap_efficiency, 3),
                    "rs_overlap_eff": round(metric.rs_overlap_efficiency, 3),
                    "step": step_name,
                },
                "s": "t", "cname": "#FF5722",
            })

    # 4. Merge and write
    trace_json["traceEvents"].extend(annotations)
    num_orig = len(trace_json["traceEvents"]) - len(annotations)
    try:
        with open(output_file, "w") as f:
            json.dump(trace_json, f)
    except OSError as e:
        print(f"  Error writing annotated trace to {output_file}: {e}")
        return
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
