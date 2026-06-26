"""HTML report with dashboard cards and interactive charts (Chart.js via CDN).

CSS, JS, and page structure live in ``html_template.html`` — this module only
builds data and injects it into the template via ``{{TITLE}}``, ``{{BODY}}``,
and ``{{CHART_DATA}}`` placeholders.
"""

import os
import json
import sys
from pipeline import process_trace, process_all_steps
from bottleneck_detector import ModelConfig, _format_us, Bottlenecks
from metric_registry import METRIC_REGISTRY, THRESHOLD_REGISTRY
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAGE_TEMPLATE = None
_BODY_TEMPLATES = {}


def _render_page(title: str, body: str, chart_data: str) -> str:
    global _PAGE_TEMPLATE
    if _PAGE_TEMPLATE is None:
        with open(os.path.join(_HERE, "html_template.html")) as f:
            _PAGE_TEMPLATE = f.read()
    data_script = f"<script>const DATA = {chart_data}</script>"
    return (_PAGE_TEMPLATE
            .replace("{{TITLE}}", title)
            .replace("{{BODY}}", body)
            .replace("{{CHART_SCRIPT}}", data_script))


def _load_body(name: str) -> str:
    if name not in _BODY_TEMPLATES:
        with open(os.path.join(_HERE, name)) as f:
            _BODY_TEMPLATES[name] = f.read()
    return _BODY_TEMPLATES[name]


def _fill(template: str, **kwargs) -> str:
    for key, val in kwargs.items():
        template = template.replace("{{" + key + "}}", str(val))
    return template


def _dashboard_cards(aggregated: dict, metrics_list: list, throughput: dict) -> str:
    num_layers = len(metrics_list)
    wall = aggregated.get("step_wall", 0)
    wall_str = _format_us(wall)
    avg_busy = sum(m.gpu_busy for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_ctc = sum(m.compute_to_comm_ratio for m in metrics_list) / len(metrics_list) if metrics_list else 0
    ctc_str = f"{avg_ctc:.2f}x" if avg_ctc != float('inf') else "inf"
    steps_s = f"{1e6 / wall:.1f}" if wall > 0 else "N/A"
    mfu = throughput.get('mfu', 0)
    tps = throughput.get('tokens_per_second_per_gpu', 0)

    cards = f"""
    <div class="kpi-item" title="Number of FSDP units (transformer layers + embeddings + output head) in the model"><div class="kpi-value">{num_layers}</div><div class="kpi-label">Layers</div></div>
    <div class="kpi-item" title="End-to-end wall time of the analysed profiler step (CPU dispatch span, including pipeline bubbles)"><div class="kpi-value">{wall_str}</div><div class="kpi-label">Step wall time</div></div>
    <div class="kpi-item" title="Steps per second — throughput. Higher is better"><div class="kpi-value">{steps_s}/s</div><div class="kpi-label">Steps per second</div></div>
    <div class="kpi-item" title="Fraction of the CPU wall span where at least one GPU kernel was active. Targets: &gt;70% good, &gt;90% excellent"><div class="kpi-value">{avg_busy:.1%}</div><div class="kpi-label">GPU utilization</div></div>
    <div class="kpi-item" title="Ratio of GPU compute time to GPU communication (NCCL collective) time. Higher = more time spent computing vs communicating. &gt;2x is balanced, &lt;1x is communication-bound"><div class="kpi-value">{ctc_str}</div><div class="kpi-label">Compute:communication ratio</div></div>
"""
    if mfu > 0:
        cards += f'<div class="kpi-item" title="Model FLOPs utilization — achieved FLOPs as a fraction of peak theoretical FLOPs. Higher is better"><div class="kpi-value">{mfu:.1%}</div><div class="kpi-label">Model FLOPs utilization (MFU)</div></div>\n'
        hfu = throughput.get('hfu', 0)
        if hfu > 0 and abs(hfu - mfu) > 0.001:
            cards += f'<div class="kpi-item" title="Hardware FLOPs utilization — accounts for activation recomputation overhead. Higher is better"><div class="kpi-value">{hfu:.1%}</div><div class="kpi-label">Hardware FLOPs utilization (HFU)</div></div>\n'
        cards += f'<div class="kpi-item" title="Tokens generated or processed per second, per GPU. Higher is better"><div class="kpi-value">{tps:.1f}</div><div class="kpi-label">Tokens per second per GPU</div></div>\n'
    return f'<div class="kpi-strip">{cards}</div>'


def _phase_pie_chart_data(aggregated: dict) -> str:
    total = aggregated.get("total_gpu_us", 0) + aggregated.get("tp_total_gpu_us", 0)
    if total == 0:
        return "null"
    phases = [
        ("Forward compute", aggregated.get("fwd_cmp_gpu_us", 0), COLORS['fwd_cmp']),
        ("Backward compute", aggregated.get("bwd_cmp_gpu_us", 0), COLORS['bwd_cmp']),
        ("All-gather fwd", aggregated.get("ag_fwd_gpu_us", 0), COLORS['ag']),
        ("All-gather bwd", aggregated.get("ag_bwd_gpu_us", 0), COLORS['ag']),
        ("Reduce-scatter", aggregated.get("rs_gpu_us", 0), COLORS['rs']),
        ("  All-reduce in RS", aggregated.get("ar_in_rs_gpu_us", 0), COLORS['ar_rs']),
        ("TP all-gather", aggregated.get("tp_ag_gpu_us", 0), COLORS['tp_ag']),
        ("TP all-reduce", aggregated.get("tp_ar_gpu_us", 0), COLORS['tp_ar']),
        ("TP reduce-scatter", aggregated.get("tp_rs_gpu_us", 0), COLORS['tp_rs']),
        ("Optimizer", aggregated.get("optimizer_gpu_us", 0), COLORS['opt']),
    ]
    phases = [(l, v, c) for l, v, c in phases if v > 0]
    labels = [p[0] for p in phases]
    values = [round(p[1], 1) for p in phases]
    colors = [p[2] for p in phases]
    return json.dumps({"labels": labels, "values": values, "colors": colors})


COLORS = {
    'fwd_cmp': '#4e79a7', 'bwd_cmp': '#e15759', 'ag': '#76b7b2',
    'rs': '#f28e2b', 'ar_rs': '#c23531', 'opt': '#59a14f', 'tp_ag': '#af7aa1',
    'tp_rs': '#ff9da7', 'tp_ar': '#9c755f',
}


def _busy_chart_data(metrics_list: list) -> str:
    labels = [m.layer_name for m in metrics_list]
    values = [round(m.gpu_busy * 100, 1) for m in metrics_list]
    return json.dumps({"labels": labels, "values": values})


def _comp_comm_chart_data(metrics_list: list, aggregated: dict) -> str:
    labels = [m.layer_name for m in metrics_list]
    datasets = [
        {"label": "Forward compute", "data": [round(m.fwd_cmp_gpu, 1) for m in metrics_list], "backgroundColor": COLORS['fwd_cmp']},
        {"label": "Backward compute", "data": [round(m.bwd_cmp_gpu, 1) for m in metrics_list], "backgroundColor": COLORS['bwd_cmp']},
        {"label": "FSDP comm", "data": [round(m.ag_fwd_gpu + m.ag_bwd_gpu + m.rs_gpu + m.ar_in_rs_gpu, 1) for m in metrics_list], "backgroundColor": COLORS['ag']},
        {"label": "TP comm", "data": [round(m.tp_total_gpu, 1) for m in metrics_list], "backgroundColor": COLORS['tp_ag']},
    ]
    return json.dumps({"labels": labels, "datasets": datasets})


def _overlap_chart_data(metrics_list: list) -> str:
    labels = [m.layer_name for m in metrics_list]
    fwd_vals = [round(m.ag_fwd_gpu, 1) for m in metrics_list]
    bwd_vals = [round(m.ag_bwd_gpu, 1) for m in metrics_list]
    rs_vals = [round(m.rs_gpu, 1) for m in metrics_list]
    ar_vals = [round(m.ar_in_rs_gpu, 1) for m in metrics_list]
    return json.dumps({
        "labels": labels,
        "datasets": [
            {"label": "AG forward", "data": fwd_vals, "backgroundColor": "#76b7b2", "borderColor": "#76b7b2"},
            {"label": "AG backward", "data": bwd_vals, "backgroundColor": "#e15759", "borderColor": "#e15759"},
            {"label": "Reduce scatter", "data": rs_vals, "backgroundColor": "#f28e2b", "borderColor": "#f28e2b"},
            {"label": "All-reduce in RS", "data": ar_vals, "backgroundColor": "#c23531", "borderColor": "#c23531"},
        ]
    })


def _ctc_chart_data(metrics_list: list) -> str:
    pairs = [(m.layer_name, m.compute_to_comm_ratio) for m in metrics_list]
    pairs = [(name, v) for name, v in pairs if v != float('inf')]
    if not pairs:
        return json.dumps({"labels": [], "values": []})
    labels, values = zip(*pairs)
    return json.dumps({"labels": list(labels), "values": [round(v, 2) for v in values]})


def _phase_metrics_table(aggregated: dict, num_layers: int) -> str:
    total_gpu = aggregated.get("total_gpu_us", 0)
    tp_total = aggregated.get("tp_total_gpu_us", 0)
    total = total_gpu + tp_total

    phases = [
        ("All-gather forward", "ag_fwd_gpu_us", COLORS['ag']),
        ("Forward compute", "fwd_cmp_gpu_us", COLORS['fwd_cmp']),
        ("  Tensor-parallel all-gather", "tp_ag_gpu_us", COLORS['tp_ag']),
        ("  Tensor-parallel all-reduce", "tp_ar_gpu_us", COLORS['tp_ar']),
        ("All-gather backward", "ag_bwd_gpu_us", COLORS['ag']),
        ("Backward compute", "bwd_cmp_gpu_us", COLORS['bwd_cmp']),
        ("Reduce scatter", "rs_gpu_us", COLORS['rs']),
        ("  All-reduce in RS", "ar_in_rs_gpu_us", COLORS['ar_rs']),
        ("  Tensor-parallel reduce-scatter", "tp_rs_gpu_us", COLORS['tp_rs']),
        ("Optimizer step", "optimizer_gpu_us", COLORS['opt']),
    ]

    rows = ""
    for label, key, color in phases:
        avg = aggregated.get(key, 0)
        pct = avg / total * 100 if total > 0 else 0
        rows += f"<tr><td><span class='legend-dot' style='background:{color}'></span> {label}</td><td>{_format_us(avg)}</td><td>{pct:.1f}%</td></tr>\n"

    rows += f"<tr style='border-top:1px solid #ccc'><td><strong>Fully sharded data parallelism (FSDP) total</strong></td><td>{_format_us(total_gpu)}</td><td></td></tr>\n"
    rows += f"<tr><td><strong>Tensor parallelism (TP) total</strong></td><td>{_format_us(tp_total)}</td><td>{tp_total / total * 100:.1f}%</td></tr>\n"
    rows += f"<tr><td>Total CPU dispatch time</td><td>{_format_us(aggregated.get('total_cpu_us', 0))}</td><td></td></tr>\n"

    return f"""<p style="font-size:11px;color:#888;margin-bottom:8px">Average GPU time per layer per phase, and its share of all GPU cycles. Lower is better for communication phases; compute phases should dominate.</p>
<table>
<tr><th>Phase</th><th>Average GPU time per layer</th><th>Share of GPU cycles</th></tr>
{rows}
</table>"""


def _efficiency_table(aggregated: dict, metrics_list: list) -> str:
    total = aggregated.get("total_gpu_us", 0) + aggregated.get("tp_total_gpu_us", 0)
    tp_total = aggregated.get("tp_total_gpu_us", 0)
    fsdp_comm = (aggregated.get("ag_fwd_gpu_us", 0) + aggregated.get("ag_bwd_gpu_us", 0)
                 + aggregated.get("rs_gpu_us", 0) + aggregated.get("ar_in_rs_gpu_us", 0))
    comp = aggregated.get("fwd_cmp_gpu_us", 0) + aggregated.get("bwd_cmp_gpu_us", 0)
    true_comp = comp
    avg_util = sum(m.gpu_busy for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_ctc = sum(m.compute_to_comm_ratio for m in metrics_list) / len(metrics_list) if metrics_list else 0
    max_span = max(m.layer_span for m in metrics_list) if metrics_list else 0
    min_span = min(m.layer_span for m in metrics_list if m.layer_span > 0) or max_span
    imbalance = max_span / min_span if min_span > 0 else 1
    avg_fwd_ov = sum(m.fwd_comp_comm_overlap for m in metrics_list) / max(len(metrics_list), 1)
    avg_bwd_ov = sum(m.bwd_comp_comm_overlap for m in metrics_list) / max(len(metrics_list), 1)
    bwd_to_fwd = (aggregated.get("bwd_cmp_gpu_us", 0) / aggregated.get("fwd_cmp_gpu_us", 1)
                  if aggregated.get("fwd_cmp_gpu_us", 0) > 0 else 0)
    ov = aggregated.get("overlap_ratio", 0)
    idle = aggregated.get("idle_ratio", 0)
    serial = aggregated.get("serial_ratio", 0)

    return f"""<p style="font-size:11px;color:#888;margin-bottom:8px">Breakdown of GPU cycles and pipeline efficiency. Compute should dominate; a high communication share (&gt;40%) suggests communication-bound performance.</p>
<table>
<tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
<tr><td>Compute time (attention, MLP)</td><td>{true_comp / total:.1%}</td><td>Share of GPU cycles on actual arithmetic (attention, MLP). Higher is better</td></tr>
<tr><td>Fully sharded data parallelism (FSDP) communication</td><td>{fsdp_comm / total:.1%}</td><td>Share on NCCL all-gather / reduce-scatter collectives. Lower is better</td></tr>
<tr><td>Tensor parallelism (TP) communication</td><td>{tp_total / total:.1%}</td><td>Share on TP all-gather / all-reduce / reduce-scatter. Lower is better</td></tr>
<tr><td>Optimizer step</td><td>{aggregated.get('optimizer_ratio', 0):.1%}</td><td>Share on ADAMW parameter update. Should be &lt;10% with fused optimizer</td></tr>
<tr><td>Average GPU utilization</td><td>{avg_util:.1%}</td><td>Fraction of wall time with active GPU kernels. Target: &gt;70%</td></tr>
<tr><td>Compute-to-communication ratio</td><td>{avg_ctc:.2f}x</td><td>Compute time per unit of communication. &gt;2x balanced, &lt;1x communication-bound</td></tr>
<tr><td>Backward-to-forward compute ratio</td><td>{bwd_to_fwd:.2f}x</td><td>Ratio of backward to forward compute time. ~2x with activation checkpointing; &gt;3x may indicate imbalance or recomputation overhead</td></tr>
<tr><td>Layer span imbalance</td><td>{imbalance:.1f}x</td><td>Ratio of longest to shortest layer wall span. High imbalance = pipeline serialization</td></tr>
<tr style='border-top:1px solid #ccc'><td>Pipeline concurrent execution</td><td>{ov:.1%}</td><td>Share of step wall where multiple layers overlapped on GPU. Higher = better overlap</td></tr>
<tr><td>Serial execution ratio</td><td>{serial:.1%}</td><td>Share where only one layer was active. Lower = better pipeline utilization</td></tr>
<tr><td>Idle / gap ratio</td><td>{idle:.1%}</td><td>Share with no GPU activity at all. May indicate data-loading stalls</td></tr>
<tr><td>Forward TP overlap ratio</td><td>{avg_fwd_ov:.1%}</td><td>How well TP communication hides behind forward compute. Higher = better overlap</td></tr>
<tr><td>Backward TP overlap ratio</td><td>{avg_bwd_ov:.1%}</td><td>How well TP communication hides behind backward compute. Higher = better overlap</td></tr>
</table>"""


def _per_unit_table(metrics_list: list) -> str:
    mem_avail = any(m.memory_has_data for m in metrics_list)
    cols = ["Layer", "All-gather\nforward", "Forward\ncompute", "All-gather\nbackward", "Backward\ncompute",
            "Reduce\nscatter", "Optimizer\n(avg/layer)",
            "Total GPU", "GPU\nutil", "Comp:\nComm", "Exposed\ncomm", "Wall\nspan", "Kernel\ncount", "Avg kernel\nduration"]
    if mem_avail:
        cols.append("Peak\nmemory")
    cols.append("Bottlenecks")
    thead = "".join(f"<th>{c}</th>" if c != "Issues" else '<th style="text-align:left">Issues</th>' for c in cols)

    # Collect raw numeric values per column for outlier detection
    numeric_keys = [
        "ag_fwd_gpu_us", "fwd_cmp_gpu_us", "ag_bwd_gpu_us", "bwd_cmp_gpu_us",
        "rs_gpu_us", "optimizer_gpu_us", "total_gpu_us", "layer_span_us",
        "gpu_busy", "compute_to_comm_ratio", "avg_exposed_ratio",
        "kernel_count", "avg_kernel_dur_us",
    ]
    col_values = {k: [] for k in numeric_keys}
    dicts = [m.to_dict() for m in metrics_list]
    for d in dicts:
        for k in numeric_keys:
            v = d.get(k, 0)
            if isinstance(v, (int, float)) and v > 0:
                col_values[k].append(v)

    def _median(vals):
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0
        return s[n // 2]

    medians = {k: _median(col_values[k]) for k in numeric_keys}

    def _is_outlier(val, med):
        if not isinstance(val, (int, float)) or val == float('inf') or med <= 0:
            return False
        return val > 2 * med or val < 0.5 * med

    rows = ""
    for mi, m in enumerate(metrics_list):
        d = dicts[mi]
        issues = Bottlenecks.detect(m)
        ctc = d.get('compute_to_comm_ratio', 0)
        ctc_str = f"{ctc:.2f}x" if ctc != float('inf') else "inf"
        exp = d.get('avg_exposed_ratio', 0)
        mem_str = ""
        if mem_avail and d.get('memory_peak', 0) > 0:
            mem_str = f"{d['memory_peak']/(1024**3):.1f}G"
        elif mem_avail:
            mem_str = "N/A"
        tags = ""
        if issues:
            short_names = [iss.split("(")[0].strip() for iss in issues]
            detail = ", ".join(short_names)
            count = len(short_names)
            severity = "tag-high" if count >= 5 else ("tag-med" if count >= 3 else "tag-low")
            tags = f'<span class="tag {severity}" title="{detail}">({count})</span>'
        else:
            tags = '<span class="tag tag-ok">OK</span>'

        def _cell(val, fmt, key=None):
            cls = ""
            title = ""
            if key is not None:
                raw = d.get(key, 0)
                med = medians[key]
                if _is_outlier(raw, med):
                    cls = ' class="ol"'
                    ratio = raw / med if med > 0 else 0
                    title = f' title="{ratio:.1f}× column median"'
            return f"<td{cls}{title}>{fmt}</td>"

        rows += f"""<tr>
<td>{m.layer_name}</td>
{_cell(d['ag_fwd_gpu_us'], _format_us(d['ag_fwd_gpu_us']), 'ag_fwd_gpu_us')}
{_cell(d['fwd_cmp_gpu_us'], _format_us(d['fwd_cmp_gpu_us']), 'fwd_cmp_gpu_us')}
{_cell(d['ag_bwd_gpu_us'], _format_us(d['ag_bwd_gpu_us']), 'ag_bwd_gpu_us')}
{_cell(d['bwd_cmp_gpu_us'], _format_us(d['bwd_cmp_gpu_us']), 'bwd_cmp_gpu_us')}
{_cell(d['rs_gpu_us'], _format_us(d['rs_gpu_us']), 'rs_gpu_us')}
{_cell(d['optimizer_gpu_us'], _format_us(d['optimizer_gpu_us']), 'optimizer_gpu_us')}
{_cell(d['total_gpu_us'], _format_us(d['total_gpu_us']), 'total_gpu_us')}
{_cell(d['gpu_busy'], f"{d['gpu_busy']:.1%}", 'gpu_busy')}
{_cell(ctc, ctc_str, 'compute_to_comm_ratio')}
{_cell(exp, f"{exp:.1%}", 'avg_exposed_ratio')}
{_cell(d['layer_span_us'], _format_us(d['layer_span_us']), 'layer_span_us')}
{_cell(d['kernel_count'], str(d['kernel_count']), 'kernel_count')}
{_cell(d['avg_kernel_dur_us'], f"{d['avg_kernel_dur_us']:.1f}us", 'avg_kernel_dur_us')}
"""
        if mem_avail:
            rows += f"<td>{mem_str}</td>\n"
        rows += f"<td class='tag-cell'>{tags}</td></tr>\n"

    info = '<p style="font-size:10px;color:#999;margin-top:6px">GPU times in microseconds. Yellow-highlighted cells are outliers (&gt;2&times; or &lt;0.5&times; column median) — inspect for load imbalance.</p>'
    return f"""<div class="table-wrap">
<table><tr>{thead}</tr>
{rows}
</table>{info}</div>"""


# ---------------------------------------------------------------------------
# Trace diagnostics — step timing, kernel stats, and consistency warnings
# surfaced in both single-trace and comparison HTML reports.
# ---------------------------------------------------------------------------

_GPU_ARCH_KEYWORDS = [
    ("ampere", "NVIDIA Ampere (A100/A30/A10/RTX 3090)"),
    ("hopper", "NVIDIA Hopper (H100/H200)"),
    ("blackwell", "NVIDIA Blackwell (B100/B200)"),
    ("turing", "NVIDIA Turing (T4/RTX 2080)"),
    ("volta", "NVIDIA Volta (V100)"),
]


def _detect_gpu_architecture(trace_file: str) -> str:
    """Determine GPU architecture from the trace file's ``deviceProperties``.

    Falls back to scanning kernel names for architecture keywords if
    ``deviceProperties`` is not present.
    """
    try:
        with open(trace_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return "unknown"

    if isinstance(data, dict):
        props = data.get("deviceProperties")
        if isinstance(props, list) and len(props) > 0:
            name = props[0].get("name", "")
            if name:
                return name

    events = data.get("traceEvents") if isinstance(data, dict) else None
    if not isinstance(events, list):
        return "unknown"

    checked = 0
    for ev in events:
        if ev.get("cat") not in ("kernel",):
            continue
        name = ev.get("name", "").lower()
        for kw, label in _GPU_ARCH_KEYWORDS:
            if kw in name:
                return label
        checked += 1
        if checked >= 2000:
            break
    return "unknown"


def _trace_step_diagnostics(trace_file: str) -> dict:
    """Read the trace JSON directly to find CPU and GPU ProfilerStep boundaries.

    Returns a dict with:
      ``cpu_dur_us`` — duration of the last CPU ProfilerStep
      ``gpu_dur_us`` — duration of the last GPU ProfilerStep (gpu_user_annotation)
      ``gap_us`` — how much GPU extends beyond CPU (positive = GPU tail)
      ``num_cpu_steps`` — total CPU ProfilerSteps found
      ``warnings`` — list of human-readable warning strings

    Uses the same lightweight ``json.load`` pattern as ``_find_profiler_steps`` in
    ``trace_annotator.py`` — no full ``TraceParser.load`` needed.
    """
    try:
        with open(trace_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

    if not isinstance(data, dict):
        return {}
    events = data.get("traceEvents")
    if not isinstance(events, list):
        return {}

    cpu_steps: list[tuple[int, int]] = []
    gpu_steps: list[tuple[int, int]] = []
    kernel_dur_buckets = {"<10us": 0, "10-100us": 0, "100us-1ms": 0, "1-10ms": 0, ">10ms": 0}

    for ev in events:
        name = ev.get("name", "")
        ph = ev.get("ph", "")
        cat = ev.get("cat", "")
        dur = ev.get("dur", 0)

        # Kernel duration bucketing (skip ProfilerStep markers)
        if dur > 0 and cat and (cat.startswith("gpu") or cat == "kernel") and ph == "X" and not name.startswith("ProfilerStep#"):
            if dur < 10:
                kernel_dur_buckets["<10us"] += 1
            elif dur < 100:
                kernel_dur_buckets["10-100us"] += 1
            elif dur < 1000:
                kernel_dur_buckets["100us-1ms"] += 1
            elif dur < 10000:
                kernel_dur_buckets["1-10ms"] += 1
            else:
                kernel_dur_buckets[">10ms"] += 1
            continue

        if not name.startswith("ProfilerStep#"):
            continue
        if ph != "X":
            continue
        ts = ev.get("ts", 0)
        if cat in ("cpu_op", "user_annotation"):
            cpu_steps.append((ts, ts + dur))
        elif cat == "gpu_user_annotation":
            gpu_steps.append((ts, ts + dur))

    if not cpu_steps:
        return {"num_cpu_steps": 0, "warnings": ["No ProfilerStep markers found."]}

    cpu_steps.sort(key=lambda x: x[0])
    gpu_steps.sort(key=lambda x: x[0])

    last_cpu = cpu_steps[-1]
    cpu_start, cpu_end = last_cpu
    cpu_dur = cpu_end - cpu_start

    gpu_dur = None
    gap = None
    warnings = []

    if gpu_steps:
        last_gpu = gpu_steps[-1]
        gpu_dur = last_gpu[1] - last_gpu[0]
        gap = last_gpu[1] - cpu_end
        if gap > cpu_dur * 0.1:
            warnings.append(
                f"GPU ProfilerStep extends {_format_us(gap)} beyond CPU step "
                f"({gap * 100 // cpu_dur}% of CPU duration) — "
                "some GPU events may be misattributed or clipped."
            )
    else:
        warnings.append(
            "No GPU ProfilerStep (gpu_user_annotation) found — "
            "cannot verify GPU event window completeness."
        )

    if len(cpu_steps) > 1:
        warnings.append(
            f"Multi-step trace: {len(cpu_steps)} ProfilerSteps found, "
            "analysing the last step only."
        )

    gpu_arch = _detect_gpu_architecture(trace_file)

    total_bucketed = sum(kernel_dur_buckets.values())
    result = {
        "num_cpu_steps": len(cpu_steps),
        "cpu_dur_us": cpu_dur,
        "gpu_dur_us": gpu_dur,
        "gap_us": gap,
        "gpu_architecture": gpu_arch,
        "kernel_dur_buckets": kernel_dur_buckets if total_bucketed > 0 else {},
        "warnings": warnings,
    }
    return result


def _kernel_stats_diagnostics(aggregated: dict, metrics_list: list) -> dict:
    """Compute kernel-level stats from already-aggregated phase data.

    Returns a dict with:
      ``total_kernels`` — sum of kernel_count across all layers
      ``avg_kernel_dur_us`` — duration-weighted average
      ``nccl_total_us`` — NCCL phase total (AG fwd + AG bwd + RS + TP phases)
      ``compute_total_us`` — compute phase total (fwd + bwd)
      ``opt_total_us`` — optimiser phase total
      ``granular_total_us`` — sum of the three categories above
    """
    total_kernels = sum(m.kernel_count for m in metrics_list)
    weighted_dur = sum(m.avg_kernel_dur_us * m.kernel_count for m in metrics_list)
    avg_dur = weighted_dur / max(total_kernels, 1)

    nccl = (aggregated.get("ag_fwd_gpu_us", 0) + aggregated.get("ag_bwd_gpu_us", 0)
            + aggregated.get("rs_gpu_us", 0) + aggregated.get("ar_in_rs_gpu_us", 0)
            + aggregated.get("tp_ag_gpu_us", 0) + aggregated.get("tp_rs_gpu_us", 0)
            + aggregated.get("tp_ar_gpu_us", 0))
    comp = aggregated.get("fwd_cmp_gpu_us", 0) + aggregated.get("bwd_cmp_gpu_us", 0)
    opt = aggregated.get("optimizer_gpu_us", 0)

    return {
        "total_kernels": total_kernels,
        "avg_kernel_dur_us": avg_dur,
        "nccl_total_us": nccl,
        "compute_total_us": comp,
        "opt_total_us": opt,
        "granular_total_us": nccl + comp + opt,
    }


def _render_diagnostics_section(diag: dict, kstats: dict) -> str:
    """Render a compact diagnostics card for single-trace HTML."""
    parts = []

    # Step timing block
    if diag.get("cpu_dur_us"):
        cpu_str = _format_us(diag["cpu_dur_us"])
        gpu_str = _format_us(diag["gpu_dur_us"]) if diag.get("gpu_dur_us") else "N/A"
        gap_str = _format_us(diag["gap_us"]) if diag.get("gap_us") else "N/A"
        gap_flag = ""
        if diag.get("gap_us") and diag.get("cpu_dur_us") and diag["gap_us"] > diag["cpu_dur_us"] * 0.1:
            gap_flag = ' <span class="tag tag-high">large GPU tail</span>'
        arch_label = diag.get("gpu_architecture", "")
        arch_row = f"<tr><td>GPU architecture</td><td>{arch_label}</td></tr>" if arch_label and arch_label != "unknown" else ""
        parts.append(
            f"<tr><td>CPU ProfilerStep</td><td>{cpu_str}</td></tr>"
            f"<tr><td>GPU ProfilerStep</td><td>{gpu_str}</td></tr>"
            f"<tr><td>GPU step gap</td><td>{gap_str}{gap_flag}</td></tr>"
            f"{arch_row}"
        )
        if diag.get("num_cpu_steps", 0) > 1:
            parts.append(f"<tr><td>ProfilerSteps</td><td>{diag['num_cpu_steps']} (last analysed)</td></tr>")

    # Kernel stats block
    if kstats:
        kc = kstats["total_kernels"]
        nccl_s = _format_us(kstats["nccl_total_us"])
        cmp_s = _format_us(kstats["compute_total_us"])
        opt_s = _format_us(kstats["opt_total_us"])
        avg_s = f"{kstats['avg_kernel_dur_us']:.1f}µs"
        bucket_rows = ""
        step_total_k = 0
        if diag.get("kernel_dur_buckets"):
            buckets = diag["kernel_dur_buckets"]
            step_total_k = sum(buckets.values())
            total_k = step_total_k
            items = ", ".join(
                f"{label}: <b>{buckets[label]:,}</b> ({buckets[label] / total_k:.1%})"
                for label in ["<10us", "10-100us", "100us-1ms", "1-10ms", ">10ms"]
                if buckets.get(label, 0) > 0
            )
            bucket_rows = f'<tr><td colspan="2" style="font-size:11px;padding-left:20px;color:#666">Kernel duration: {items}</td></tr>'
        layer_label = f"GPU kernels (attributed to FSDP layers)</td><td>{kc:,}</td></tr>"
        step_label = f"<tr><td>GPU kernels in profiler step</td><td>{step_total_k:,}</td></tr>" if step_total_k else ""
        parts.append(
            f"<tr><td>{layer_label}"
            f"{step_label}"
            f"<tr><td>NCCL / compute / optimizer</td><td>{nccl_s} / {cmp_s} / {opt_s}</td></tr>"
            f"<tr><td>Avg kernel duration</td><td>{avg_s}</td></tr>"
            f"{bucket_rows}"
        )

    rows = "".join(parts)
    if not rows:
        return ''

    # Warnings
    warns = diag.get("warnings", [])
    warn_html = ""
    if warns:
        items = "".join(f'<li style="margin:4px 0;font-size:12px">{w}</li>' for w in warns)
        warn_html = f'<div style="background:#fef3cd;border:1px solid #ffc107;padding:8px 12px;margin-top:8px"><strong style="font-size:13px">Diagnostics</strong><ul style="margin:4px 0 0 16px;padding:0">{items}</ul></div>'

    return f"""<div class="section">
<h2>Trace Diagnostics</h2>
<div style="display:flex;gap:24px;flex-wrap:wrap">
<table style="min-width:auto"><tr><th colspan="2" style="text-align:left">Step Timing</th></tr>
{rows}
</table>
</div>
{warn_html}
</div>"""


def _render_consistency_warnings(all_results: list, trace_files: list = None) -> str:
    """Check per-trace kernel stats for consistency and emit warnings for the comparison page."""
    if len(all_results) < 2:
        return ""

    n_traces = len(all_results)

    # Collect nccl_total_us per trace
    nccl_totals = []
    comp_totals = []
    labels = []
    for label, agg, metrics, steps, tp in all_results:
        kstats = _kernel_stats_diagnostics(agg, metrics)
        nccl_totals.append(kstats["nccl_total_us"])
        comp_totals.append(kstats["compute_total_us"])
        labels.append(label)

    warnings = []
    if nccl_totals:
        mx = max(nccl_totals)
        mn = min(nccl_totals)
        if mn > 0 and mx / mn > 1.5:
            slow = labels[nccl_totals.index(mx)]
            fast = labels[nccl_totals.index(mn)]
            warnings.append(
                f"NCCL kernel time varies {mx/mn:.1f}x across traces "
                f"({slow}: {_format_us(mx)} vs {fast}: {_format_us(mn)}). "
                "If compute kernel times are similar, the NCCL variance is likely a "
                "profiler recording artifact (dropped CUPTI activity records), "
                "not genuine performance variation."
            )

    if comp_totals:
        mx = max(comp_totals)
        mn = min(comp_totals)
        if mn > 0 and mx / mn > 1.2:
            slow = labels[comp_totals.index(mx)]
            fast = labels[comp_totals.index(mn)]
            warnings.append(
                f"Compute kernel time varies {mx/mn:.1f}x across traces "
                f"({slow}: {_format_us(mx)} vs {fast}: {_format_us(mn)}). "
                "Identical models should have near-identical compute times — "
                "investigate if model configs differ."
            )

    # Info header: trace count + GPU arch
    info_parts = [f"{n_traces} trace files"]
    if trace_files:
        arch = _detect_gpu_architecture(trace_files[0])
        if arch != "unknown":
            info_parts.append(arch)
    info_line = " &middot; ".join(info_parts)

    if not warnings:
        return f"""<div class="section">
<h2>Trace Consistency</h2>
<div style="background:#e8f5e9;border:1px solid #4caf50;padding:10px 16px">
<strong style="font-size:14px">Traces consistent</strong>
<div style="margin-top:6px;font-size:13px;color:#555">{info_line}</div>
</div>
</div>"""

    items = "".join(
        f'<li style="margin:6px 0;font-size:13px">{w}</li>' for w in warnings
    )
    return f"""<div class="section">
<h2>Trace Consistency</h2>
<div style="background:#fef3cd;border:1px solid #ffc107;padding:10px 16px">
<strong style="font-size:14px">Inconsistencies detected</strong>
<div style="margin-top:6px;font-size:13px;color:#555">{info_line}</div>
<ul style="margin:8px 0 0 16px;padding:0">{items}</ul>
</div>
</div>"""


BOTTLENECK_DESCRIPTIONS = {
    "I/O or pipeline bubble":
        "Wall time with no active FSDP work. May indicate a data-loading stall, Python GIL contention, "
        "or a synchronization barrier.",

    "comm-bound":
        "Communication consumes most of the GPU budget. Check interconnect topology (NVLink/NIC), "
        "reduce all-gather frequency, or increase sharding degree (HSDP).",

    "all-gather-heavy":
        "All-gather dominates FSDP communication. Consider a higher sharding degree (HSDP), "
        "async all-gather, or overlapping AG with compute.",

    "reduce-scatter-heavy":
        "Reduce-scatter dominates FSDP communication. Try gradient compression, a higher sharding "
        "degree, or fusing RS with backward compute.",

    "TP-heavy":
        "Tensor-parallel collectives dominate GPU time. Reduce TP degree if the model fits in "
        "host memory, or fuse TP communication kernels.",

    "optimizer-heavy":
        "ADAMW parameter update dominates GPU. Enable fused optimizer (apex FusedAdam) or reduce "
        "precision for optimizer states.",

    "low GPU utilization":
        "GPU is idle more than half the wall span. The pipeline stagger may not be overlapping "
        "this layer's work with neighbours. Check for blocking CPU operations or excessive synchronization.",

    "small-kernel-bound":
        "Hundreds of tiny CUDA kernels — mostly launch latency, not GPU execution. Use torch.compile "
        "or manually fuse the split/empty kernels in the all-gather copy-in path.",

    "serial pipeline":
        "Layers execute mostly sequentially on GPU — little overlap between consecutive shard groups. "
        "The FSDP2 pipeline stagger may be operating at too coarse a granularity.",

    "low async TP overlap":
        "Async TP collectives are not hiding behind compute. TP communication sits on the critical "
        "path instead of overlapping with GEMM kernels. Check TP communication scheduling.",

    "async TP asymmetry":
        "Forward and backward TP overlap differ significantly. The pipeline stagger is uneven "
        "across phases — investigate if activation recomputation or gradient scaling is asymmetric.",

    "host-bound":
        "CPU wall span is much wider than GPU execution. Likely pipeline serialization — the CPU "
        "launches a layer, starts the next, and only returns later. Check for blocking CPU "
        "operations or Python-level serialization.",

    "copy-heavy all-gather":
        "All-gather dominated by GPU memcpy (split_with_sizes_copy) rather than NCCL. Common with "
        "small buffer sizes — try fusing the copy kernels or increasing the shard granularity.",

    "fwd-bwd imbalance":
        "Forward and backward phases use noticeably different GPU time. This may be natural with "
        "activation checkpointing; if extreme, check for gradient accumulation asymmetry.",

    "HBM bandwidth-bound":
        "Compute kernels average <8 µs with low GPU utilisation — memory-bandwidth-limited. "
        "Check for suboptimal tensor shapes or use memory-bound-optimized kernel implementations.",

    "exposed all-gather":
        "All-gather GPU time approaches or exceeds forward compute — the gather is fully exposed. "
        "Upgrade inter-node bandwidth (IB/RoCE), overlap AG with compute, or use async AG.",

    "synchronous TP on critical path":
        "TP collectives overlap poorly with compute and sit on the critical path. This defeats "
        "the purpose of async TP. Consider fusing TP communication or reducing TP degree.",

    "NVLink saturation":
        "Many small TP kernel launches fragment NVLink bandwidth. Try fusing TP communication "
        "messages or increasing the message size per collective.",

    "no comm/compute overlap":
        "The all-gather overlaps poorly with forward compute. The pipeline stagger may be "
        "insufficient for this layer — check for excessive synchronization or narrow the pipeline.",

    "exposed communication":
        "GPU is idle for much of the wall time — communication is fully exposed. Consider "
        "overlapping communication with compute from other layers via pipeline stagger.",

    "low cross-layer GPU overlap":
        "Adjacent layers' GPU spans overlap by <20%. The pipeline stagger is not keeping the GPU "
        "busy — try increasing the number of in-flight layers or narrowing the pipeline window.",

    "AG fwd dominates":
        "All-gather forward consumes a disproportionate share of GPU time. Likely a pipeline "
        "bottleneck or insufficient overlap with compute.",

    "AG bwd dominates":
        "All-gather backward consumes a disproportionate share of GPU time. Likely a pipeline "
        "bottleneck or insufficient overlap with backward compute.",

    "RS dominates":
        "Reduce-scatter consumes a disproportionate share of GPU time. Check gradient accumulation "
        "or injection pressure.",

    "Optimizer dominates":
        "The optimizer step consumes a disproportionate share of GPU time. Consider a fused optimizer "
        "or reduce precision for optimizer states.",

    "TP dominates":
        "Tensor-parallel collectives dominate GPU time. Consider reducing TP degree or fusing "
        "TP communication kernels.",

    "RS exceeds bwd compute":
        "Reduce-scatter GPU time exceeds backward compute GPU time — strong injection pressure. "
        "Consider gradient accumulation or increasing sharding degree.",
}

# Key supplementary metrics per bottleneck type for layer-level evidence.
BOTTLENECK_EVIDENCE_METRICS = {
    "I/O or pipeline bubble": ["idle_ratio", "gpu_busy"],
    "comm-bound": ["comm_ratio", "gpu_busy"],
    "all-gather-heavy": ["ag_fwd_gpu_us", "ag_bwd_gpu_us", "gpu_busy"],
    "reduce-scatter-heavy": ["rs_gpu_us", "gpu_busy"],
    "TP-heavy": ["tp_total_gpu_us", "total_gpu_us"],
    "optimizer-heavy": ["optimizer_ratio", "optimizer_gpu_us"],
    "low GPU utilization": ["gpu_busy", "fwd_busy", "bwd_busy"],
    "small-kernel-bound": ["kernel_count", "avg_kernel_dur_us"],
    "serial pipeline": ["serial_ratio", "overlap_ratio", "gpu_busy"],
    "low async TP overlap": ["fwd_comp_comm_overlap", "bwd_comp_comm_overlap"],
    "async TP asymmetry": ["fwd_comp_comm_overlap", "bwd_comp_comm_overlap"],
    "host-bound": ["cpu_wall_to_gpu_ratio", "cpu_wall_us", "total_gpu_us"],
    "copy-heavy all-gather": ["copy_data_movement_gpu_us", "nccl_comm_gpu_us"],
    "fwd-bwd imbalance": ["fwd_cmp_gpu_us", "bwd_cmp_gpu_us", "ag_fwd_gpu_us", "ag_bwd_gpu_us", "rs_gpu_us"],
    "exposed all-gather": ["ag_fwd_gpu_us", "fwd_cmp_gpu_us", "gpu_busy"],
    "HBM bandwidth-bound": ["comp_kernel_avg_dur_us", "comp_ratio", "gpu_busy"],
    "RS exceeds bwd compute": ["rs_gpu_us", "bwd_cmp_gpu_us"],
    "synchronous TP on critical path": ["tp_total_gpu_us", "fwd_cmp_gpu_us", "fwd_comp_comm_overlap"],
    "NVLink saturation": ["comp_kernel_avg_dur_us", "nccl_in_comp_count"],
    "no comm/compute overlap": ["gpu_busy", "ag_fwd_gpu_us", "fwd_cmp_gpu_us"],
    "exposed communication": ["gpu_busy", "fwd_cmp_gpu_us", "bwd_cmp_gpu_us"],
    "low cross-layer GPU overlap": ["pipeline_overlap_ratio", "serial_ratio"],
    "AG fwd dominates": ["ag_fwd_gpu_us", "total_gpu_us"],
    "AG bwd dominates": ["ag_bwd_gpu_us", "total_gpu_us"],
    "RS dominates": ["rs_gpu_us", "total_gpu_us"],
    "Optimizer dominates": ["optimizer_gpu_us", "total_gpu_us"],
    "TP dominates": ["tp_total_gpu_us", "total_gpu_us"],
}


def _evidence_for_layer(metrics, short_name):
    """Extract key metric evidence for a given bottleneck type from a Metrics object."""
    evidence_metrics = BOTTLENECK_EVIDENCE_METRICS.get(short_name, [])
    d = metrics.to_dict()
    parts = []
    # These are unit-ratio metrics (0..1) formatted as percentages
    unit_ratio_keys = {"gpu_busy", "fwd_busy", "bwd_busy",
                       "fwd_comp_comm_overlap", "bwd_comp_comm_overlap",
                       "pipeline_overlap_ratio", "serial_ratio",
                       "idle_ratio", "comp_ratio", "comm_ratio",
                       "optimizer_ratio", "fsdp_comm_ratio",
                       "tp_comm_ratio", "avg_exposed_ratio",
                       "ag_fwd_exposed_ratio", "rs_exposed_ratio",
                       "ag_bwd_exposed_ratio", "overlap_ratio",
                       "compute_to_comm_ratio"}
    for key in evidence_metrics:
        v = d.get(key)
        if v is None:
            continue
        if key in unit_ratio_keys:
            parts.append(f"{key}={v:.2%}")
        elif key == "cpu_wall_to_gpu_ratio":
            parts.append(f"{key}={v:.1f}x")
        elif key.endswith("_us") or key in ("cpu_wall_us", "total_gpu_us"):
            parts.append(f"{key}={v:.0f}us")
        elif key in ("kernel_count", "nccl_in_comp_count"):
            parts.append(f"{key}={v:.0f}")
        elif key in ("avg_kernel_dur_us", "comp_kernel_avg_dur_us"):
            parts.append(f"{key}={v:.1f}us")
        else:
            parts.append(f"{key}={v}")
    return ", ".join(parts)


def _bottleneck_tags(metrics_list: list) -> str:
    # Group by short name (before parenthesis) to collapse variants like
    # "comm-bound (comm=58.3%)" and "comm-bound (comm=85.0%)".
    # Also capture per-layer evidence.
    all_issues = defaultdict(list)  # short_name -> [(layer_name, issue_text, Metrics)]
    for m in metrics_list:
        issues = Bottlenecks.detect(m)
        for iss in issues:
            short = iss.split("(")[0].strip()
            all_issues[short].append((m.layer_name, iss, m))

    if not all_issues:
        return '<p class="tag-ok" style="font-size:12px">No bottlenecks detected.</p>'

    legend = ('<div style="font-size:11px;color:#888;margin-bottom:10px">'
              '<span style="color:#c0392b">&#9679;</span> widespread (&#8805;50% of layers)'
              ' &nbsp; '
              '<span style="color:#b8860b">&#9679;</span> moderate (&#8805;3 layers)'
              ' &nbsp; '
              '<span style="color:#1a7a2e">&#9679;</span> few layers'
              '</div>')
    parts = legend
    for short, entries in sorted(all_issues.items(), key=lambda x: -len(x[1])):
        count = len(set(layer_name for layer_name, _, _ in entries))
        if count >= len(metrics_list) * 0.5:
            severity = "tag-high"
        elif count >= 3:
            severity = "tag-med"
        else:
            severity = "tag-low"
        # Count layers for display
        layer_names = [l for l, _, _ in entries]
        layers_str = ", ".join(layer_names[:4])
        if len(layer_names) > 4:
            layers_str += f" (+{len(layer_names) - 4})"
        desc = BOTTLENECK_DESCRIPTIONS.get(short, "")
        desc_suffix = f"<br><span style='font-size:11px;color:#666;font-style:italic;margin-left:8px'>{desc}</span>" if desc else ""

        # Build collapsible content: per-layer evidence lines
        evidence_lines = ""
        for layer_name, issue_text, m in entries:
            extra = _evidence_for_layer(m, short)
            evidence_str = issue_text
            if extra:
                evidence_str += f"  [{extra}]"
            evidence_lines += (
                f'<div style="font-size:11px;color:#444;padding:2px 0 2px 16px">'
                f'<span style="font-weight:500">{layer_name}:</span> {evidence_str}'
                f'</div>\n'
            )

        details_id = f"bneck-{short.replace(' ', '-').replace('/', '-')}"
        collapsible = (
            f'<details id="{details_id}" style="margin:2px 0">'
            f'<summary style="cursor:pointer;font-size:11px;color:#888;padding:2px 0">'
            f'<span class="tag {severity}">{short}</span> '
            f'<span style="font-size:12px;font-weight:500">{count}/{len(metrics_list)} layers</span>'
            f' — {layers_str}'
            f'{desc_suffix}'
            f'</summary>'
            f'{evidence_lines}'
            f'</details>'
        )
        parts += collapsible + '\n'
    return parts


def _model_config_card(cfg: ModelConfig) -> str:
    if not cfg or not cfg.is_configured:
        return ""
    parts = [f"<b>{cfg.hidden_dim}</b> hidden, <b>{cfg.num_layers}</b> layers, <b>{cfg.num_heads}</b> heads"]
    if cfg.num_kv_heads != cfg.num_heads:
        parts.append(f"<b>{cfg.num_kv_heads}</b> KV heads")
    if cfg.intermediate_dim and cfg.intermediate_dim != 4 * cfg.hidden_dim:
        parts.append(f"<b>{cfg.intermediate_dim}</b> intermediate")
    parts += [f"seq <b>{cfg.seq_len}</b>", f"vocab <b>{cfg.vocab_size}</b>",
              f"batch <b>{cfg.batch_size}</b>", f"<b>{cfg.num_gpus}</b> GPUs"]
    if cfg.activation_checkpointing is not None:
        parts.append(f"act ckpt <b>{cfg.activation_checkpointing:.0%}</b>")
    return f'<div style="margin:6px 0 14px;padding:8px 14px;border:1px solid #ddd;font-size:13px;color:#555;line-height:1.7"><b>Model config:</b> {", ".join(parts)}</div>\n'


def _render_metric_registry() -> str:
    """Render the METRIC_REGISTRY as a table showing description, calculation, unit, and which bottlenecks use each metric."""
    rows = ""
    used = {k: v for k, v in METRIC_REGISTRY.items() if v.get("used_by")}
    unused = {k: v for k, v in METRIC_REGISTRY.items() if not v.get("used_by")}

    for label, group in [("Used by bottleneck detection", used), ("Informational only (not used by detection)", unused)]:
        if not group:
            continue
        rows += f'<tr style="background:#f5f5f5"><td colspan="5" style="font-weight:600;font-size:11px;padding:6px 7px;text-align:left">{label}</td></tr>\n'
        for key, info in sorted(group.items()):
            desc = info.get("description", "")
            calc = info.get("calculation", "")
            unit = info.get("unit", "")
            used_by = info.get("used_by", [])
            used_str = ", ".join(used_by) if used_by else '<span style="color:#999">—</span>'
            calc_cell = f'<span style="font-size:10px;color:#666;font-style:italic">{calc}</span>' if calc else '<span style="color:#999;font-size:10px">—</span>'
            rows += (
                f"<tr>"
                f"<td style='font-family:monospace;font-size:10px;white-space:nowrap'>{key}</td>"
                f"<td style='font-size:10px;color:#555'>{desc}</td>"
                f"<td>{calc_cell}</td>"
                f"<td style='font-size:10px'>{unit}</td>"
                f"<td style='font-size:10px'>{used_str}</td>"
                f"</tr>\n"
            )

    return f"""<div class="section">
<h2>Metric Registry</h2>
<details>
<summary style="cursor:pointer;font-size:12px;color:#555;padding:4px 0;margin-bottom:4px">
All {len(METRIC_REGISTRY)} metrics — click to expand
</summary>
<p style="font-size:11px;color:#888;margin-bottom:8px">
The "used_by" column shows which bottlenecks consume each metric — metrics with no consumers are informational only.
</p>
<div class="table-wrap">
<table>
<thead><tr>
<th style="text-align:left">Metric</th>
<th style="text-align:left">Description</th>
<th style="text-align:left">Calculation</th>
<th>Unit</th>
<th style="text-align:left">Used by</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>
</details>
</div>"""


def _render_threshold_registry() -> str:
    """Render the THRESHOLD_REGISTRY as a table showing value, rationale, physical justification, and which bottlenecks use each threshold."""
    import re as _re
    rows = ""
    sections = {
        "A": ("Classical bottleneck thresholds", []),
        "B": ("FSDP2 / TP / async TP specific thresholds", []),
        "C": ("Communication-hiding / BW bottleneck thresholds", []),
    }
    for key, info in THRESHOLD_REGISTRY.items():
        val = info.get("value", "")
        val_s = f"{val:.0%}" if isinstance(val, float) and 0 < val < 1 else str(val)
        rat = info.get("rationale", "")
        phys = info.get("physical_justification", "")
        used_by = info.get("used_by", [])
        used_str = ", ".join(used_by) if used_by else '<span style="color:#999">—</span>'
        key_lower = key.lower()
        if any(w in key_lower for w in ["comp_heavy", "comm_heavy", "io_heavy", "ag_heavy", "rs_heavy", "tp_heavy", "optimizer_heavy", "util_low"]):
            sections["A"][1].append((key, val_s, rat, phys, used_str))
        elif any(w in key_lower for w in ["small_kernel", "async_tp", "overlap_asymmetry", "host_bound", "copy_heavy", "fwd_bwd", "serial_ratio"]):
            sections["B"][1].append((key, val_s, rat, phys, used_str))
        else:
            sections["C"][1].append((key, val_s, rat, phys, used_str))

    for sec_id, (sec_title, items) in sections.items():
        if not items:
            continue
        sec_id_attr = _re.sub(r"[^a-zA-Z0-9_-]", "", sec_title.replace(" ", "-").lower())
        rows += (
            f'<tr style="background:#f5f5f5">'
            f'<td colspan="5" style="font-weight:600;font-size:11px;padding:8px 7px;text-align:left">'
            f'Sec {sec_id} — {sec_title} ({len(items)} thresholds)</td></tr>\n'
        )
        for key, val_s, rat, phys, used_str in items:
            rows += (
                f"<tr>"
                f"<td style='font-family:monospace;font-size:10px;white-space:nowrap'>{key}</td>"
                f"<td style='font-size:10px;color:#555'>{rat}</td>"
                f"<td style='font-size:10px;font-weight:500'>{val_s}</td>"
                f"<td style='font-size:10px;color:#444;font-style:italic;max-width:360px'>{phys}</td>"
                f"<td style='font-size:10px'>{used_str}</td>"
                f"</tr>\n"
            )

    return f"""<div class="section">
<h2>Threshold Registry</h2>
<details>
<summary style="cursor:pointer;font-size:12px;color:#555;padding:4px 0;margin-bottom:4px">
All {len(THRESHOLD_REGISTRY)} thresholds with physical justifications — click to expand
</summary>
<p style="font-size:11px;color:#888;margin-bottom:8px">
Each threshold is documented with its numeric value, detection rationale, and the physical or architectural model it derives from (roofline model, NCCL bandwidth model, Amdahl's law, GPU architecture specs for H100-SXM). Thresholds are grouped by their section in the detection logic.
</p>
<div class="table-wrap">
<table>
<thead><tr>
<th style="text-align:left">Threshold</th>
<th style="text-align:left">Rationale</th>
<th>Value</th>
<th style="text-align:left">Physical Justification</th>
<th style="text-align:left">Used by</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>
</details>
</div>"""


def generate_html_report(trace_file: str, output_path: str = None, model_config: ModelConfig = None):
    """Run the full pipeline and write an enhanced HTML report with charts."""
    result = process_trace(trace_file, model_config=model_config)
    if result is None:
        print(f"Failed to load {trace_file}.")
        return

    aggregated, metrics_list, fsdp, report, text = result

    if output_path is None:
        base, _ = os.path.splitext(trace_file)
        output_path = f"{base}.html"

    title = f"Trace Analysis — {os.path.basename(trace_file)}"
    num_layers = len(metrics_list)

    # Trace diagnostics
    step_diag = _trace_step_diagnostics(trace_file)
    kstats = _kernel_stats_diagnostics(aggregated, metrics_list)
    diag_section = _render_diagnostics_section(step_diag, kstats)

    # Serialise chart data to JSON
    chart_data = {
        "phasePie": json.loads(_phase_pie_chart_data(aggregated)),
        "busyChart": json.loads(_busy_chart_data(metrics_list)),
        "compCommChart": json.loads(_comp_comm_chart_data(metrics_list, aggregated)),
        "overlapChart": json.loads(_overlap_chart_data(metrics_list)),
        "ctcChart": json.loads(_ctc_chart_data(metrics_list)),
    }

    body = _fill(
        _load_body("single_body_template.html"),
        TITLE=title,
        SUBTITLE=f"{os.path.basename(trace_file)} — {num_layers} layers",
        MODEL_CONFIG_CARD=_model_config_card(model_config),
        DASHBOARD_CARDS=_dashboard_cards(aggregated, metrics_list, report.throughput_metrics),
        DIAGNOSTICS_SECTION=diag_section,
        PHASE_METRICS_TABLE=_phase_metrics_table(aggregated, num_layers),
        EFFICIENCY_TABLE=_efficiency_table(aggregated, metrics_list),
        BOTTLENECK_TAGS=_bottleneck_tags(metrics_list),
        PER_UNIT_TABLE=_per_unit_table(metrics_list),
        METRIC_REGISTRY_SECTION=_render_metric_registry(),
        THRESHOLD_REGISTRY_SECTION=_render_threshold_registry(),
    )
    html = _render_page(title, body, json.dumps(chart_data))
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"HTML report written to {output_path}")


def generate_compare_html(trace_files, output_path=None, model_config=None):
    """Compare multiple traces side by side in an HTML report with charts."""
    all_results = []
    trace_labels = []
    for tf in trace_files:
        label = os.path.basename(tf)
        print(f"Processing {label}...", file=sys.stderr)
        steps = process_all_steps(tf, model_config=model_config)
        if not steps:
            print(f"  FAILED to load {tf}", file=sys.stderr)
            continue
        n = len(steps)
        avg_agg = {}
        all_metrics = []
        reports = []
        for step_name, agg, metrics_list, fsdp, report, text in steps:
            for k, v in agg.items():
                if isinstance(v, (int, float)):
                    avg_agg[k] = avg_agg.get(k, 0.0) + v
            all_metrics.extend(metrics_list)
            reports.append(report)
        for k in avg_agg:
            avg_agg[k] /= n

        throughput = reports[-1].throughput_metrics if reports else {}
        trace_labels.append(label)
        all_results.append((label, avg_agg, all_metrics, steps, throughput))

    if not all_results:
        print("No traces processed successfully.")
        return

    if output_path is None:
        base = os.path.commonprefix(trace_files).rstrip('_-. ')
        if not base:
            base = "comparison"
        output_path = f"{base}_comparison.html"

    title = f"Trace Comparison — {', '.join(trace_labels)}"
    m = len(trace_labels)
    palettes = ["#4e79a7", "#e15759", "#76b7b2", "#f28e2b", "#59a14f", "#af7aa1", "#ff9da7", "#9c755f"]
    colors = palettes[:m]

    # Summary cards per trace
    summary_cards = ""
    for i, (label, agg, metrics, steps, tp) in enumerate(all_results):
        wall = agg.get("step_wall", 0)
        busy = sum(m.gpu_busy for m in metrics) / max(len(metrics), 1)
        ctc = sum(m.compute_to_comm_ratio for m in metrics) / max(len(metrics), 1)
        ctc_s = f"{ctc:.2f}x" if ctc != float('inf') else "inf"
        mfu = tp.get("mfu", 0)
        tps = tp.get("tokens_per_second_per_gpu", 0)
        steps_s = f"{1e6 / wall:.1f}" if wall > 0 else "N/A"
        kstats = _kernel_stats_diagnostics(agg, metrics)
        total_kernels = kstats["total_kernels"]

        extras = ""
        if mfu > 0:
            extras += f"<tr><td>MFU</td><td>{mfu:.1%}</td></tr>"
            extras += f"<tr><td>Tok/s/GPU</td><td>{tps:.1f}</td></tr>"
        arch = _detect_gpu_architecture(trace_files[i])
        arch_row = f"<tr><td>GPU</td><td>{arch}</td></tr>" if arch != "unknown" else ""
        extras += (
            f"<tr><td>GPU kernels</td><td>{total_kernels:,}</td></tr>"
            f"<tr><td>NCCL total</td><td>{_format_us(kstats['nccl_total_us'])}</td></tr>"
            f"<tr><td>Compute total</td><td>{_format_us(kstats['compute_total_us'])}</td></tr>"
            f"<tr><td>Avg kernel</td><td>{kstats['avg_kernel_dur_us']:.1f}µs</td></tr>"
            f"{arch_row}"
        )

        summary_cards += f"""<div class="cmp-card" style="border-top:3px solid {colors[i]}">
<h3>{label}</h3>
<table>
<tr><td>Step wall</td><td>{_format_us(wall)}</td></tr>
<tr><td>Layers</td><td>{len(metrics)}</td></tr>
<tr><td>Steps/s</td><td>{steps_s}</td></tr>
<tr><td>GPU busy</td><td>{busy:.1%}</td></tr>
<tr><td>Comp:Comm</td><td>{ctc_s}</td></tr>
{extras}
</table></div>"""

    # Build comparison chart data
    def _avg(vals):
        return sum(vals) / max(len(vals), 1)

    def _get_phases(agg):
        return {
            "ag_fwd": agg.get("ag_fwd_gpu_us", 0),
            "fwd_cmp": agg.get("fwd_cmp_gpu_us", 0),
            "ag_bwd": agg.get("ag_bwd_gpu_us", 0),
            "bwd_cmp": agg.get("bwd_cmp_gpu_us", 0),
            "rs": agg.get("rs_gpu_us", 0),
            "ar_rs": agg.get("ar_in_rs_gpu_us", 0),
            "opt": agg.get("optimizer_gpu_us", 0),
            "tp_ag": agg.get("tp_ag_gpu_us", 0),
            "tp_rs": agg.get("tp_rs_gpu_us", 0),
            "tp_ar": agg.get("tp_ar_gpu_us", 0),
        }

    # Key metrics chart
    km_labels = ["GPU utilization", "Pipeline concurrent\nexecution", "Serial execution\nratio", "Communication\nratio", "Exposed\ncommunication"]
    km_datasets = []
    for i, (label, agg, metrics, steps, tp) in enumerate(all_results):
        vals = [
            _avg([m.gpu_busy for m in metrics]),
            agg.get("overlap_ratio", 0),
            agg.get("serial_ratio", 0),
            agg.get("comm_ratio", 0),
            _avg([m.avg_exposed_ratio for m in metrics]),
        ]
        km_datasets.append({"label": label, "data": [round(v * 100, 1) for v in vals], "backgroundColor": colors[i]})

    # Phase times charts (compute and comm split)
    pt_fwd_bwd_labels = ["Forward\ncompute", "Backward\ncompute"]
    pt_fwd_bwd_datasets = []
    pt_comm_labels = ["All-gather\nforward", "All-gather\nbackward", "Reduce\nscatter", "All-reduce\nin RS"]
    pt_comm_datasets = []
    for i, (label, agg, metrics, steps, tp) in enumerate(all_results):
        ph = _get_phases(agg)
        fwd_bwd_vals = [ph["fwd_cmp"] / 1000, ph["bwd_cmp"] / 1000]
        pt_fwd_bwd_datasets.append({"label": label, "data": [round(v, 2) for v in fwd_bwd_vals], "backgroundColor": colors[i]})
        comm_vals = [ph["ag_fwd"] / 1000, ph["ag_bwd"] / 1000, ph["rs"] / 1000, ph["ar_rs"] / 1000]
        pt_comm_datasets.append({"label": label, "data": [round(v, 2) for v in comm_vals], "backgroundColor": colors[i]})

    # MFU chart
    mfu_datasets = None
    mfu_labels = []
    for _, agg, metrics, steps, tp in all_results:
        if tp.get("mfu", 0) > 0:
            mfu_labels = ["MFU", "HFU"] if tp.get("hfu", 0) > 0 and abs(tp["hfu"] - tp["mfu"]) > 0.001 else ["MFU"]
            break
    if mfu_labels:
        mfu_datasets = []
        for i, (label, agg, metrics, steps, tp) in enumerate(all_results):
            vals = [tp.get("mfu", 0) * 100]
            if len(mfu_labels) > 1:
                vals.append(tp.get("hfu", 0) * 100)
            mfu_datasets.append({"label": label, "data": [round(v, 1) for v in vals], "backgroundColor": colors[i]})

    # Comm breakdown chart — use total (sum) ratios, not per-layer averages,
    # because outlier layers (e.g. tok_embeddings, lm_head) inflate averages.
    comm_labels = ["Communication\nratio", "FSDP\ncommunication", "Tensor-parallel\ncommunication"]
    comm_datasets = []
    for i, (label, agg, metrics, steps, tp) in enumerate(all_results):
        total_fsdp_comm = sum(m.ag_fwd_gpu + m.ag_bwd_gpu + m.rs_gpu for m in metrics)
        total_tp_comm = sum(m.tp_total_gpu for m in metrics)
        total_gpu = sum(m.total_gpu + m.tp_total_gpu for m in metrics)
        cr = (total_fsdp_comm + total_tp_comm) / total_gpu if total_gpu > 0 else 0.0
        fr = total_fsdp_comm / total_gpu if total_gpu > 0 else 0.0
        tr = total_tp_comm / total_gpu if total_gpu > 0 else 0.0
        vals = [cr * 100, fr * 100, tr * 100]
        comm_datasets.append({"label": label, "data": [round(v, 1) for v in vals], "backgroundColor": colors[i]})

    compare_data = {
        "keyMetrics": {"labels": km_labels, "datasets": km_datasets, "unitY": "%"},
        "phaseTimes": {"labels": pt_fwd_bwd_labels, "datasets": pt_fwd_bwd_datasets, "unitY": "ms"},
        "phaseTimesComm": {"labels": pt_comm_labels, "datasets": pt_comm_datasets, "unitY": "ms"},
        "commRatios": {"labels": comm_labels, "datasets": comm_datasets, "unitY": "%"},
    }
    if mfu_datasets:
        compare_data["mfu"] = {"labels": mfu_labels, "datasets": mfu_datasets, "unitY": "%"}

    # ------------------------------------------------------------------
    # Comprehensive comparison table with color coding
    # ------------------------------------------------------------------

    def _better(val, baseline, direction):
        if direction == "higher":
            return val > baseline
        return val < baseline

    def _worse(val, baseline, direction):
        if direction == "higher":
            return val < baseline
        return val > baseline

    def _fmt_metric(name, raw_val):
        if raw_val is None:
            return "N/A"
        # Percentage metrics
        pct_names = ("GPU utilization", "Communication ratio", "FSDP communication ratio",
                     "Tensor-parallel communication ratio",
                     "Pipeline concurrent execution", "Serial execution ratio",
                     "Pipeline idle ratio", "Average exposed communication ratio",
                     "All-gather forward exposed ratio",
                     "Forward TP overlap ratio", "Backward TP overlap ratio",
                     "Model FLOPs utilization (MFU)", "Hardware FLOPs utilization (HFU)")
        if name in pct_names:
            return f"{raw_val:.1%}" if isinstance(raw_val, float) else str(raw_val)
        if name == "Compute-to-communication ratio":
            return "inf" if raw_val == float('inf') else f"{raw_val:.2f}x"
        # Speedup metrics (ratio vs baseline, higher = better)
        if name.endswith("speedup"):
            return f"{raw_val:.2f}x" if isinstance(raw_val, (int, float)) else str(raw_val)
        if name == "Peak memory":
            return f"{raw_val:.1f}G" if raw_val and raw_val > 0 else "N/A"
        # Microsecond metrics
        us_names = ("Step wall time",
                    "All-gather forward (avg per layer)", "Forward compute (avg per layer)",
                    "All-gather backward (avg per layer)", "Backward compute (avg per layer)",
                    "Reduce scatter (avg per layer)", "All-reduce in RS (avg per layer)", "Optimizer (avg per layer)",
                    "TP total (avg per layer)", "Total GPU (avg per layer)",
                    "CPU dispatch (avg per layer)",
                    "Tensor-parallel all-gather (avg per layer)",
                    "Tensor-parallel all-reduce (avg per layer)",
                    "Tensor-parallel reduce-scatter (avg per layer)")
        if name in us_names:
            return _format_us(raw_val) if isinstance(raw_val, (int, float)) else str(raw_val)
        if name == "Steps per second":
            return f"{raw_val:.1f}"
        if name == "Tokens per second per GPU":
            return f"{raw_val:.1f}"
        return str(raw_val)

    # Define all metrics to show in the comparison table
    COMPARE_METRICS = [
        ("Layers", lambda r, m: int(len(m)), None, False),
        ("Step wall time", lambda r, m: r.get("step_wall", 0), "lower", True),
        ("Steps per second", lambda r, m: 1e6 / r.get("step_wall", 1) if r.get("step_wall", 0) > 0 else 0, "higher", True),
    ]

    # Phase times (all per-layer averages)
    for label, key in [("All-gather forward (avg per layer)", "ag_fwd_gpu_us"),
                        ("Forward compute (avg per layer)", "fwd_cmp_gpu_us"),
                        ("All-gather backward (avg per layer)", "ag_bwd_gpu_us"),
                        ("Backward compute (avg per layer)", "bwd_cmp_gpu_us"),
                        ("Reduce scatter (avg per layer)", "rs_gpu_us"),
                        ("All-reduce in RS (avg per layer)", "ar_in_rs_gpu_us"),
                        ("Optimizer (avg per layer)", "optimizer_gpu_us"),
                        ("Tensor-parallel all-gather (avg per layer)", "tp_ag_gpu_us"),
                        ("Tensor-parallel all-reduce (avg per layer)", "tp_ar_gpu_us"),
                        ("Tensor-parallel reduce-scatter (avg per layer)", "tp_rs_gpu_us")]:
        COMPARE_METRICS.append((label, lambda r, m, k=key: r.get(k, 0), "lower", True))

    COMPARE_METRICS.append(("TP total (avg per layer)", lambda r, m: r.get("tp_total_gpu_us", 0), "lower", True))
    COMPARE_METRICS.append(("Total GPU (avg per layer)", lambda r, m: r.get("total_gpu_us", 0) + r.get("tp_total_gpu_us", 0), "lower", True))
    COMPARE_METRICS.append(("CPU dispatch (avg per layer)", lambda r, m: r.get("total_cpu_us", 0), "lower", True))

    # Utilization & ratios
    COMPARE_METRICS.append(("GPU utilization", lambda r, m: sum(x.gpu_busy for x in m) / max(len(m), 1), "higher", True))
    COMPARE_METRICS.append(("Compute-to-communication ratio (avg per layer)", lambda r, m: sum(x.compute_to_comm_ratio for x in m) / max(len(m), 1), "higher", True))

    # MFU/HFU
    COMPARE_METRICS.append(("Model FLOPs utilization (MFU)", lambda r, m, tp=None: tp.get("mfu", 0) if tp else 0, "higher", True))
    COMPARE_METRICS.append(("Hardware FLOPs utilization (HFU)", lambda r, m, tp=None: tp.get("hfu", 0) if tp else 0, "higher", True))
    COMPARE_METRICS.append(("Tokens per second per GPU", lambda r, m, tp=None: tp.get("tokens_per_second_per_gpu", 0) if tp else 0, "higher", True))

    # Comm breakdown (total ratios — not per-layer averages)
    COMPARE_METRICS.append(("Communication ratio", lambda r, m: (sum(x.ag_fwd_gpu + x.ag_bwd_gpu + x.rs_gpu + x.ar_in_rs_gpu + x.tp_total_gpu for x in m)) / max(sum(x.total_gpu + x.tp_total_gpu for x in m), 1), "lower", True))
    COMPARE_METRICS.append(("FSDP communication ratio", lambda r, m: (sum(x.ag_fwd_gpu + x.ag_bwd_gpu + x.rs_gpu + x.ar_in_rs_gpu for x in m)) / max(sum(x.total_gpu + x.tp_total_gpu for x in m), 1), "lower", True))
    COMPARE_METRICS.append(("Tensor-parallel communication ratio", lambda r, m: (sum(x.tp_total_gpu for x in m)) / max(sum(x.total_gpu + x.tp_total_gpu for x in m), 1), "lower", True))

    # Overlap & pipeline
    COMPARE_METRICS.append(("Pipeline concurrent execution", lambda r, m: r.get("overlap_ratio", 0), "higher", True))
    COMPARE_METRICS.append(("Serial execution ratio", lambda r, m: r.get("serial_ratio", 0), "lower", True))
    COMPARE_METRICS.append(("Pipeline idle ratio", lambda r, m: r.get("idle_ratio", 0), "lower", True))

    # Efficiency (all per-layer averages)
    COMPARE_METRICS.append(("Average exposed communication ratio", lambda r, m: sum(x.avg_exposed_ratio for x in m) / max(len(m), 1), "lower", True))
    COMPARE_METRICS.append(("All-gather forward exposed ratio (avg per layer)", lambda r, m: sum(x.ag_fwd_exposed_ratio for x in m) / max(len(m), 1), "lower", True))
    COMPARE_METRICS.append(("Forward TP overlap ratio (avg per layer)", lambda r, m: sum(x.fwd_comp_comm_overlap for x in m) / max(len(m), 1), "higher", True))
    COMPARE_METRICS.append(("Backward TP overlap ratio (avg per layer)", lambda r, m: sum(x.bwd_comp_comm_overlap for x in m) / max(len(m), 1), "higher", True))

    # Memory
    COMPARE_METRICS.append(("Peak memory", lambda r, m: max((x.memory_peak for x in m if x.memory_has_data), default=0) / (1024**3), "lower", True))

    # Bottlenecks (text only, no color)
    COMPARE_METRICS.append(("Bottlenecks", lambda r, m: _format_bottleneck_summary(m), None, False))

    # Compute numeric values for each trace
    trace_values = []
    for ri, (label, agg, metrics, steps, tp) in enumerate(all_results):
        vals = {}
        for name, fn, direction, do_color in COMPARE_METRICS:
            try:
                if any(x in name for x in ("MFU", "HFU", "Tokens")):
                    _tp = all_results[ri][4]
                    if _tp.get("mfu", 0) > 0:
                        vals[name] = fn(agg, metrics, tp=_tp)
                    else:
                        vals[name] = None
                else:
                    vals[name] = fn(agg, metrics)
            except:
                vals[name] = None
        trace_values.append(vals)

    # Speedup rows: ratio vs baseline (first trace).  >1.00x = faster.
    speedup_pairs = [("Step wall speedup", "Step wall time"),
                     ("Total GPU speedup", "Total GPU (avg per layer)")]
    for sname, src_name in speedup_pairs:
        base_val = trace_values[0].get(src_name, 1)
        for ri in range(len(trace_values)):
            cur = trace_values[ri].get(src_name)
            if cur and base_val and cur > 0:
                trace_values[ri][sname] = base_val / cur
            else:
                trace_values[ri][sname] = None
        COMPARE_METRICS.append((sname, lambda r, m: None, "higher", True))

    # Build the HTML table with color coding
    thead_parts = ["<tr><th>Metric</th>"]
    for l in trace_labels:
        thead_parts.append(f'<th>{l}</th>')
    thead_parts.append('</tr>')

    # Format a delta string for a cell value vs baseline
    _pct_names = {"GPU utilization", "Communication ratio", "FSDP communication ratio",
                  "Tensor-parallel communication ratio",
                  "Pipeline concurrent execution", "Serial execution ratio",
                  "Pipeline idle ratio", "Average exposed communication ratio",
                  "All-gather forward exposed ratio (avg per layer)",
                  "Forward TP overlap ratio (avg per layer)", "Backward TP overlap ratio (avg per layer)",
                  "Model FLOPs utilization (MFU)", "Hardware FLOPs utilization (HFU)"}

    def _diff_magnitude(name, vals):
        max_v = max(vals)
        min_v = min(vals)
        if max_v == min_v:
            return 0.0
        if name in _pct_names:
            return (max_v - min_v) * 100  # percentage-point span
        if min_v > 0:
            return (max_v - min_v) / min_v * 100  # relative % span
        return 0.0

    def _format_delta(val, bval, name):
        if bval is None or val is None or bval == 0 or val == bval:
            return ""
        # Percentage metrics: absolute percentage-point change
        if name in _pct_names:
            diff = (val - bval) * 100
            if abs(diff) < 0.05:
                return ""
            sign = "+" if diff > 0 else ""
            return f" ({sign}{diff:.1f}pp)"
        # Everything else: relative percent change
        rel = (val - bval) / bval * 100
        if abs(rel) < 0.5:
            return ""
        sign = "+" if rel > 0 else ""
        return f" ({sign}{rel:.0f}%)"

    # Color coding: compare traces[1..n] against traces[0]
    baseline_vals = trace_values[0] if trace_values else {}

    table_rows = ""
    for name, fn, direction, do_color in COMPARE_METRICS:
        cells = f"<td><strong>{name}</strong></td>"
        for ri in range(len(trace_values)):
            val = trace_values[ri].get(name)
            formatted = _fmt_metric(name, val)
            cell_class = ""
            if do_color and ri > 0 and direction is not None and val is not None and baseline_vals.get(name) is not None:
                bval = baseline_vals[name]
                if bval is not None and val != bval:
                    delta = _format_delta(val, bval, name)
                    if delta:  # Only color when the difference passes the threshold (same threshold as delta text)
                        if _better(val, bval, direction):
                            cell_class = ' class="cmp-better"'
                        elif _worse(val, bval, direction):
                            cell_class = ' class="cmp-worse"'
                        formatted += delta
            cells += f"<td{cell_class}>{formatted}</td>"
        table_rows += f"<tr>{cells}</tr>\n"

    # Key differences: find metrics with the largest span across traces
    diffs = []
    for name, fn, direction, do_color in COMPARE_METRICS:
        if not do_color or name in ("Bottlenecks",) or name.endswith("speedup"):
            continue
        vals = [(ri, trace_values[ri].get(name)) for ri in range(len(trace_values))]
        vals = [(ri, v) for ri, v in vals if isinstance(v, (int, float))]
        if len(vals) < 2:
            continue
        unique_vals = set(round(v, 4) for _, v in vals)
        if len(unique_vals) < 2:
            continue
        mag = _diff_magnitude(name, [v for _, v in vals])
        if mag < 1.0:
            continue
        min_ri = min(vals, key=lambda x: x[1])[0]
        max_ri = max(vals, key=lambda x: x[1])[0]
        min_lab = trace_labels[min_ri]
        max_lab = trace_labels[max_ri]
        min_s = _fmt_metric(name, min(v for _, v in vals))
        max_s = _fmt_metric(name, max(v for _, v in vals))
        pct_s = f"{mag:.0f}%" if name not in _pct_names else f"{mag:.1f}pp"
        direction_word = "higher" if direction == "higher" else "lower"
        diffs.append((mag, name, min_lab, max_lab, min_s, max_s, pct_s, direction_word))

    diffs.sort(key=lambda x: -x[0])
    top_diffs = diffs[:12]

    key_diff_items = ""
    if top_diffs:
        rows = ""
        for mag, name, min_lab, max_lab, min_s, max_s, pct_s, dw in top_diffs:
            rows += (
                f"<tr>"
                f"<td style='font-weight:600;white-space:nowrap'>{name}</td>"
                f"<td>{max_s} <span style='color:#888;font-size:11px'>({max_lab})</span></td>"
                f"<td>{min_s} <span style='color:#888;font-size:11px'>({min_lab})</span></td>"
                f"<td style='font-weight:600'>{pct_s}</td>"
                f"<td style='color:#666;font-size:12px'>{'Lower is better' if dw == 'lower' else 'Higher is better'}</td>"
                f"</tr>\n"
            )
        key_diff_html = f"""
<div style="margin-top:20px">
<h3>Largest Differences</h3>
<p style="font-size:11px;color:#888;margin-bottom:8px">Metrics with the widest span across traces, sorted largest first.</p>
<div class="table-wrap">
<table class="cmp-table">
<thead><tr>
<th>Metric</th>
<th>Highest</th>
<th>Lowest</th>
<th>Range</th>
<th>Direction</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>
</div>"""
    else:
        key_diff_html = ""

    # Bottleneck summary per trace
    bneck_legend = ('<div style="font-size:10px;color:#888;margin-bottom:8px">'
                    '<span style="color:#c0392b">&#9679;</span> &#8805;50% of layers'
                    ' &nbsp; '
                    '<span style="color:#b8860b">&#9679;</span> &#8805;20% of layers'
                    ' &nbsp; '
                    '<span style="color:#1a7a2e">&#9679;</span> &lt;20% of layers'
                    '</div>')
    bneck_sections = ""
    for i, (label, agg, metrics, steps, tp) in enumerate(all_results):
        issues = defaultdict(list)
        for m in metrics:
            for iss in Bottlenecks.detect(m):
                issues[iss.split("(")[0].strip()].append(m.layer_name)
        if not issues:
            bneck_sections += f'<div class="comp-col"><h3>{label}</h3><p class="tag-ok" style="font-size:12px">No bottlenecks detected.</p></div>'
            continue
        seen = set()
        rows = ""
        for short, layers in sorted(issues.items(), key=lambda x: -len(x[1])):
            if short in seen:
                continue
            seen.add(short)
            count = len(layers)
            pct = count / max(len(metrics), 1)
            cls = "tag-high" if pct >= 0.5 else ("tag-med" if pct >= 0.2 else "tag-low")
            desc = BOTTLENECK_DESCRIPTIONS.get(short, "")
            desc_suffix = f"<br><span style='font-size:10px;color:#666;font-style:italic'>{desc}</span>" if desc else ""
            rows += f'<div style="margin:3px 0"><span class="tag {cls}">{short}</span> <span style="font-size:11px;color:#888">{count}/{len(metrics)}</span>{desc_suffix}</div>'
        bneck_sections += f'<div class="comp-col"><h3>{label}</h3>{rows}</div>'

    trace_tabs = "".join(
        f'<span class="trace-tab active" style="background:{colors[i]}">{l}</span>'
        for i, l in enumerate(trace_labels)
    )
    mfu_chart = ""
    if mfu_datasets:
        mfu_chart = f'''<div class="comp-col">
<h3>MFU / HFU</h3>
<div class="chart-box"><canvas id="cmpMfu"></canvas></div>
</div>'''

    consistency_warnings = _render_consistency_warnings(all_results, trace_files=trace_files)

    # Verdict card: pick the most informative speedup headline
    wall_candidates = [(ri, trace_values[ri].get("Step wall speedup", 0) or 0) for ri in range(len(trace_values))]
    gpu_candidates = [(ri, trace_values[ri].get("Total GPU speedup", 0) or 0) for ri in range(len(trace_values))]
    wall_vals = [v for _, v in wall_candidates if v > 0]
    gpu_vals = [v for _, v in gpu_candidates if v > 0]
    wall_range = max(wall_vals) - min(wall_vals) if len(wall_vals) > 1 else 0
    gpu_range = max(gpu_vals) - min(gpu_vals) if len(gpu_vals) > 1 else 0
    if wall_range > 0.02:
        best_ri, best_val = max(wall_candidates, key=lambda x: x[1])
        worst_ri, _ = min(wall_candidates, key=lambda x: x[1])
        label = trace_labels[best_ri]
        ratio = best_val / min(wall_vals)
        detail = f"({ratio:.2f}x step wall vs {trace_labels[worst_ri]})"
        detail_color = "#666" if best_ri != 0 else "#888"
    elif gpu_range > 0.01:
        best_ri, best_val = max(gpu_candidates, key=lambda x: x[1])
        worst_ri, _ = min(gpu_candidates, key=lambda x: x[1])
        label = trace_labels[best_ri]
        ratio = best_val / min(gpu_vals)
        detail = f"({ratio:.2f}x GPU time vs {trace_labels[worst_ri]})"
        detail_color = "#666" if best_ri != 0 else "#888"
    else:
        label = trace_labels[0]
        detail = "(performance is comparable across traces)"
        detail_color = "#888"
    verdict_card = (
        f'<div style="margin:14px 0 12px;padding:12px 16px;'
        f'border:1px solid #ccc;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap">'
        f'<span style="font-size:20px;font-weight:700;color:#111">Trace with highest speedup:</span>'
        f'<span style="font-size:18px;font-weight:600;color:#111">{label}</span>'
        f'<span style="font-size:14px;color:{detail_color}">{detail}</span>'
        f'</div>'
    )

    body = _fill(
        _load_body("compare_body_template.html"),
        TITLE=title,
        FILES=", ".join(trace_files),
        BASELINE=trace_labels[0],
        VERDICT_CARD=verdict_card,
        TRACE_TABS=trace_tabs,
        CONSISTENCY_WARNINGS=consistency_warnings,
        KEY_DIFFERENCES=key_diff_html,
        SUMMARY_CARDS=summary_cards,
        MFU_CHART=mfu_chart,
        TABLE_HEAD="".join(thead_parts),
        TABLE_ROWS=table_rows,
        BOTTLENECK_LEGEND=bneck_legend,
        BOTTLENECK_SECTIONS=bneck_sections,
        METRIC_REGISTRY_SECTION=_render_metric_registry(),
        THRESHOLD_REGISTRY_SECTION=_render_threshold_registry(),
    )
    chart_data = json.dumps({"compare": compare_data})
    html = _render_page(title, body, chart_data)
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Comparison HTML written to {output_path}")


def _format_bottleneck_summary(metrics_list) -> str:
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
