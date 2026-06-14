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
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAGE_TEMPLATE = None
_BODY_TEMPLATES = {}


def _render_page(title: str, body: str, chart_data: str) -> str:
    global _PAGE_TEMPLATE
    if _PAGE_TEMPLATE is None:
        with open(os.path.join(_HERE, "html_template.html")) as f:
            _PAGE_TEMPLATE = f.read()
    return (_PAGE_TEMPLATE
            .replace("{{TITLE}}", title)
            .replace("{{BODY}}", body)
            .replace("{{CHART_DATA}}", chart_data))


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
    avg_util = sum(m.gpu_util for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_ctc = sum(m.compute_to_comm_ratio for m in metrics_list) / len(metrics_list) if metrics_list else 0
    ctc_str = f"{avg_ctc:.2f}x" if avg_ctc != float('inf') else "inf"
    steps_s = f"{1e6 / wall:.1f}" if wall > 0 else "N/A"
    mfu = throughput.get('mfu', 0)
    tps = throughput.get('tokens_per_second_per_gpu', 0)

    cards = f"""
    <div class="card"><div class="value">{num_layers}</div><div class="label">Layers</div></div>
    <div class="card"><div class="value">{wall_str}</div><div class="label">Step Wall</div></div>
    <div class="card"><div class="value">{steps_s}/s</div><div class="label">Steps/sec</div></div>
    <div class="card"><div class="value">{avg_util:.1%}</div><div class="label">GPU Util</div></div>
    <div class="card"><div class="value">{ctc_str}</div><div class="label">Comp:Comm</div></div>
"""
    if mfu > 0:
        cards += f'<div class="card"><div class="value">{mfu:.1%}</div><div class="label">MFU</div></div>\n'
        hfu = throughput.get('hfu', 0)
        if hfu > 0 and abs(hfu - mfu) > 0.001:
            cards += f'<div class="card"><div class="value">{hfu:.1%}</div><div class="label">HFU</div></div>\n'
        cards += f'<div class="card"><div class="value">{tps:.1f}</div><div class="label">Tok/s/GPU</div></div>\n'
    return f'<div class="dashboard">{cards}</div>'


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
    'rs': '#f28e2b', 'opt': '#59a14f', 'tp_ag': '#af7aa1',
    'tp_rs': '#ff9da7', 'tp_ar': '#9c755f',
}


def _util_chart_data(metrics_list: list) -> str:
    labels = [m.layer_name for m in metrics_list]
    values = [round(m.gpu_util * 100, 1) for m in metrics_list]
    return json.dumps({"labels": labels, "values": values})


def _comp_comm_chart_data(metrics_list: list, aggregated: dict) -> str:
    labels = [m.layer_name for m in metrics_list]
    tp_total = aggregated.get("tp_total_gpu_us", 0) / max(len(metrics_list), 1)
    datasets = [
        {"label": "Forward compute", "data": [round(m.fwd_cmp_gpu, 1) for m in metrics_list], "backgroundColor": COLORS['fwd_cmp']},
        {"label": "Backward compute", "data": [round(m.bwd_cmp_gpu, 1) for m in metrics_list], "backgroundColor": COLORS['bwd_cmp']},
        {"label": "FSDP comm", "data": [round(m.ag_fwd_gpu + m.ag_bwd_gpu + m.rs_gpu, 1) for m in metrics_list], "backgroundColor": COLORS['ag']},
        {"label": "TP comm", "data": [round(tp_total, 1) for _ in metrics_list], "backgroundColor": COLORS['tp_ag']},
    ]
    return json.dumps({"labels": labels, "datasets": datasets})


def _overlap_chart_data(metrics_list: list) -> str:
    labels = ["AG forward", "AG backward", "Reduce scatter"]
    vals = []
    for attr in ["ag_fwd_overlap_efficiency", "ag_bwd_overlap_efficiency", "rs_overlap_efficiency"]:
        vs = [getattr(m, attr) for m in metrics_list]
        vals.append(round(sum(vs) / len(vs), 3) if vs else 0)
    return json.dumps({"labels": labels, "values": vals})


def _ctc_chart_data(metrics_list: list) -> str:
    labels = [m.layer_name for m in metrics_list]
    values = []
    for m in metrics_list:
        v = m.compute_to_comm_ratio
        values.append(round(v, 2) if v != float('inf') else None)
    return json.dumps({"labels": labels, "values": values})


def _phase_metrics_table(aggregated: dict, num_layers: int) -> str:
    total_gpu = aggregated.get("total_gpu_us", 0)
    tp_total = aggregated.get("tp_total_gpu_us", 0)
    total = total_gpu + tp_total

    phases = [
        ("All-gather fwd", "ag_fwd_gpu_us", COLORS['ag']),
        ("Forward compute", "fwd_cmp_gpu_us", COLORS['fwd_cmp']),
        ("  TP all-gather", "tp_ag_gpu_us", COLORS['tp_ag']),
        ("  TP all-reduce", "tp_ar_gpu_us", COLORS['tp_ar']),
        ("All-gather bwd", "ag_bwd_gpu_us", COLORS['ag']),
        ("Backward compute", "bwd_cmp_gpu_us", COLORS['bwd_cmp']),
        ("Reduce-scatter", "rs_gpu_us", COLORS['rs']),
        ("  TP reduce-scatter", "tp_rs_gpu_us", COLORS['tp_rs']),
        ("Optimizer step", "optimizer_gpu_us", COLORS['opt']),
    ]

    rows = ""
    for label, key, color in phases:
        per = aggregated.get(key, 0)
        tot = per * num_layers
        pct = per / total * 100 if total > 0 else 0
        rows += f"<tr><td><span class='legend-dot' style='background:{color}'></span> {label}</td><td>{_format_us(per)}</td><td>{_format_us(tot)}</td><td>{pct:.1f}%</td></tr>\n"

    rows += f"<tr style='border-top:2px solid #ccc'><td><strong>FSDP total</strong></td><td>{_format_us(total_gpu)}</td><td>{_format_us(total_gpu * num_layers)}</td><td></td></tr>\n"
    rows += f"<tr><td><strong>TP total</strong></td><td>{_format_us(tp_total)}</td><td>{_format_us(tp_total * num_layers)}</td><td></td></tr>\n"
    rows += f"<tr><td><strong>Total (incl. TP)</strong></td><td>{_format_us(total_gpu + tp_total)}</td><td>{_format_us((total_gpu + tp_total) * num_layers)}</td><td>100.0%</td></tr>\n"
    rows += f"<tr><td>Total CPU</td><td>{_format_us(aggregated.get('total_cpu_us', 0))}</td><td></td><td></td></tr>\n"

    return f"""<table>
<tr><th>Phase</th><th>Per unit</th><th>Total</th><th>% GPU</th></tr>
{rows}
</table>"""


def _efficiency_table(aggregated: dict, metrics_list: list) -> str:
    total = aggregated.get("total_gpu_us", 0) + aggregated.get("tp_total_gpu_us", 0)
    tp_total = aggregated.get("tp_total_gpu_us", 0)
    fsdp_comm = aggregated.get("ag_fwd_gpu_us", 0) + aggregated.get("ag_bwd_gpu_us", 0) + aggregated.get("rs_gpu_us", 0)
    comp = aggregated.get("fwd_cmp_gpu_us", 0) + aggregated.get("bwd_cmp_gpu_us", 0)
    true_comp = comp - tp_total
    avg_util = sum(m.gpu_util for m in metrics_list) / len(metrics_list) if metrics_list else 0
    avg_ctc = sum(m.compute_to_comm_ratio for m in metrics_list) / len(metrics_list) if metrics_list else 0
    max_span = max(m.layer_span for m in metrics_list) if metrics_list else 0
    min_span = min(m.layer_span for m in metrics_list if m.layer_span > 0) or max_span
    imbalance = max_span / min_span if min_span > 0 else 1
    avg_fwd_ov = sum(m.fwd_comp_comm_overlap for m in metrics_list) / max(len(metrics_list), 1)
    avg_bwd_ov = sum(m.bwd_comp_comm_overlap for m in metrics_list) / max(len(metrics_list), 1)
    ov = aggregated.get("overlap_ratio", 0)
    idle = aggregated.get("idle_ratio", 0)
    serial = aggregated.get("serial_exec_efficiency", 0)

    return f"""<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>True compute (ex-TP)</td><td>{true_comp / total:.1%}</td></tr>
<tr><td>FSDP communication</td><td>{fsdp_comm / total:.1%}</td></tr>
<tr><td>TP communication</td><td>{tp_total / total:.1%}</td></tr>
<tr><td>Optimizer</td><td>{aggregated.get('optimizer_ratio', 0):.1%}</td></tr>
<tr><td>Avg GPU utilization</td><td>{avg_util:.1%}</td></tr>
<tr><td>Compute-to-comm ratio</td><td>{avg_ctc:.2f}x</td></tr>
<tr><td>Layer span imbalance</td><td>{imbalance:.1f}x</td></tr>
<tr style='border-top:2px solid #ccc'><td>Pipeline overlap</td><td>{ov:.1%}</td></tr>
<tr><td>Serial execution</td><td>{serial:.1%}</td></tr>
<tr><td>Idle / gap time</td><td>{idle:.1%}</td></tr>
<tr><td>Fwd TP overlap</td><td>{avg_fwd_ov:.1%}</td></tr>
<tr><td>Bwd TP overlap</td><td>{avg_bwd_ov:.1%}</td></tr>
</table>"""


def _per_unit_table(metrics_list: list) -> str:
    mem_avail = any(m.memory_has_data for m in metrics_list)
    cols = ["Layer", "AG fwd", "Fwd cmp", "AG bwd", "Bwd cmp", "RS", "Opt",
            "Total", "Util", "CtC", "ExpC", "Span", "K#", "AvgK"]
    if mem_avail:
        cols.append("Mem")
    cols.append("Issues")
    thead = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for m in metrics_list:
        d = m.to_dict()
        issues = Bottlenecks.detect(m)
        ctc = d.get('compute_to_comm_ratio', 0)
        ctc_str = f"{ctc:.2f}x" if ctc != float('inf') else "inf"
        exp = d.get('exposed_comm_fraction', 0)
        mem_str = ""
        if mem_avail and d.get('memory_peak', 0) > 0:
            mem_str = f"{d['memory_peak']/(1024**3):.1f}G"
        elif mem_avail:
            mem_str = "N/A"
        tags = ""
        if issues:
            for iss in issues[:3]:
                tag_cls = "tag-high" if any(w in iss for w in ["bound", "saturation", "BW", "heavy"]) else "tag-med"
                tags += f'<span class="tag {tag_cls}">{iss.split("(")[0].strip()}</span> '
        else:
            tags = '<span class="tag tag-ok">OK</span>'

        rows += f"""<tr>
<td>{m.layer_name}</td>
<td>{_format_us(d['ag_fwd_gpu_us'])}</td>
<td>{_format_us(d['fwd_cmp_gpu_us'])}</td>
<td>{_format_us(d['ag_bwd_gpu_us'])}</td>
<td>{_format_us(d['bwd_cmp_gpu_us'])}</td>
<td>{_format_us(d['rs_gpu_us'])}</td>
<td>{_format_us(d['optimizer_gpu_us'])}</td>
<td>{_format_us(d['total_gpu_us'])}</td>
<td>{d['gpu_util']:.1%}</td>
<td>{ctc_str}</td>
<td>{exp:.1%}</td>
<td>{_format_us(d['layer_span_us'])}</td>
<td>{d['kernel_count']}</td>
<td>{d['avg_kernel_dur_us']:.1f}us</td>
"""
        if mem_avail:
            rows += f"<td>{mem_str}</td>\n"
        rows += f"<td style='text-align:left'>{tags}</td></tr>\n"

    return f"""<div class="table-wrap">
<table><tr>{thead}</tr>
{rows}
</table></div>"""


def _bottleneck_tags(metrics_list: list) -> str:
    all_issues = defaultdict(list)
    for m in metrics_list:
        issues = Bottlenecks.detect(m)
        for iss in issues:
            all_issues[iss].append(m.layer_name)

    if not all_issues:
        return '<p style="color:#2e7d32">No bottlenecks detected.</p>'

    parts = ""
    for iss, layers in sorted(all_issues.items(), key=lambda x: -len(x[1])):
        count = len(layers)
        if count >= len(metrics_list) * 0.5:
            severity = "tag-high"
        elif count >= 3:
            severity = "tag-med"
        else:
            severity = "tag-low"
        short = iss.split("(")[0].strip()
        layers_str = ", ".join(layers[:4])
        if len(layers) > 4:
            layers_str += f" (+{len(layers) - 4})"
        parts += f'<div style="margin:4px 0"><span class="tag {severity}">{short}</span> <span style="font-size:.8em;color:#666">{count} units — {layers_str}</span></div>\n'
    return parts


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

    # Serialise chart data to JSON
    chart_data = {
        "phasePie": json.loads(_phase_pie_chart_data(aggregated)),
        "utilChart": json.loads(_util_chart_data(metrics_list)),
        "compCommChart": json.loads(_comp_comm_chart_data(metrics_list, aggregated)),
        "overlapChart": json.loads(_overlap_chart_data(metrics_list)),
        "ctcChart": json.loads(_ctc_chart_data(metrics_list)),
    }

    body = _fill(
        _load_body("single_body_template.html"),
        TITLE=title,
        SUBTITLE=f"{os.path.basename(trace_file)} — {num_layers} layers",
        DASHBOARD_CARDS=_dashboard_cards(aggregated, metrics_list, report.throughput_metrics),
        PHASE_METRICS_TABLE=_phase_metrics_table(aggregated, num_layers),
        EFFICIENCY_TABLE=_efficiency_table(aggregated, metrics_list),
        BOTTLENECK_TAGS=_bottleneck_tags(metrics_list),
        PER_UNIT_TABLE=_per_unit_table(metrics_list),
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
        util = sum(m.gpu_util for m in metrics) / max(len(metrics), 1)
        ctc = sum(m.compute_to_comm_ratio for m in metrics) / max(len(metrics), 1)
        ctc_s = f"{ctc:.2f}x" if ctc != float('inf') else "inf"
        mfu = tp.get("mfu", 0)
        tps = tp.get("tokens_per_second_per_gpu", 0)
        steps_s = f"{1e6 / wall:.1f}" if wall > 0 else "N/A"

        extras = ""
        if mfu > 0:
            extras += f"<tr><td>MFU</td><td>{mfu:.1%}</td></tr>"
            extras += f"<tr><td>Tok/s/GPU</td><td>{tps:.1f}</td></tr>"

        summary_cards += f"""<div class="trace-summary" style="border-top:4px solid {colors[i]}">
<h3>{label}</h3>
<table>
<tr><td>Step wall</td><td>{_format_us(wall)}</td></tr>
<tr><td>Layers</td><td>{len(metrics)}</td></tr>
<tr><td>Steps/s</td><td>{steps_s}</td></tr>
<tr><td>GPU util</td><td>{util:.1%}</td></tr>
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
            "opt": agg.get("optimizer_gpu_us", 0),
            "tp_ag": agg.get("tp_ag_gpu_us", 0),
            "tp_rs": agg.get("tp_rs_gpu_us", 0),
            "tp_ar": agg.get("tp_ar_gpu_us", 0),
        }

    # Key metrics chart
    km_labels = ["GPU Util", "Pipeline Overlap", "Serial Exec", "Comm Ratio", "ExpC"]
    km_datasets = []
    for i, (label, agg, metrics, steps, tp) in enumerate(all_results):
        vals = [
            _avg([m.gpu_util for m in metrics]),
            agg.get("overlap_ratio", 0),
            agg.get("serial_exec_efficiency", 0),
            agg.get("comm_ratio", 0),
            _avg([m.exposed_comm_fraction for m in metrics]),
        ]
        km_datasets.append({"label": label, "data": [round(v * 100, 1) for v in vals], "backgroundColor": colors[i]})

    # Phase times chart
    pt_labels = ["AG fwd", "Fwd cmp", "AG bwd", "Bwd cmp", "RS", "Opt"]
    pt_datasets = []
    for i, (label, agg, metrics, steps, tp) in enumerate(all_results):
        ph = _get_phases(agg)
        vals = [ph[k] / 1000 for k in ["ag_fwd", "fwd_cmp", "ag_bwd", "bwd_cmp", "rs", "opt"]]
        pt_datasets.append({"label": label, "data": [round(v, 2) for v in vals], "backgroundColor": colors[i]})

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

    # Comm breakdown chart
    comm_labels = ["Comm ratio", "FSDP comm", "TP comm"]
    comm_datasets = []
    for i, (label, agg, metrics, steps, tp) in enumerate(all_results):
        vals = [
            agg.get("comm_ratio", 0) * 100,
            _avg([m.fsdp_comm_ratio for m in metrics]) * 100,
            _avg([m.tp_comm_ratio for m in metrics]) * 100,
        ]
        comm_datasets.append({"label": label, "data": [round(v, 1) for v in vals], "backgroundColor": colors[i]})

    compare_data = {
        "keyMetrics": {"labels": km_labels, "datasets": km_datasets},
        "phaseTimes": {"labels": pt_labels, "datasets": pt_datasets},
        "commRatios": {"labels": comm_labels, "datasets": comm_datasets},
    }
    if mfu_datasets:
        compare_data["mfu"] = {"labels": mfu_labels, "datasets": mfu_datasets}

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
        if name in ("GPU util", "Comm ratio", "FSDP comm ratio", "TP comm ratio",
                     "Pipeline overlap", "Serial efficiency", "Pipeline idle",
                     "Exposed comm fraction", "AG overlap efficiency",
                     "Fwd TP overlap", "Bwd TP overlap", "MFU", "HFU"):
            return f"{raw_val:.1%}" if isinstance(raw_val, float) else str(raw_val)
        if name == "Compute-to-comm ratio":
            return "inf" if raw_val == float('inf') else f"{raw_val:.2f}x"
        if name == "Peak memory":
            return f"{raw_val:.1f}G" if raw_val and raw_val > 0 else "N/A"
        if name in ("Step wall", "AG forward", "Forward compute", "AG backward",
                     "Backward compute", "Reduce scatter", "Optimizer",
                     "TP total", "Total GPU", "Total CPU",
                     "TP all-gather", "TP all-reduce", "TP reduce-scatter"):
            return _format_us(raw_val) if isinstance(raw_val, (int, float)) else str(raw_val)
        if name == "Steps/second":
            return f"{raw_val:.1f}"
        if name == "Tokens/sec/GPU":
            return f"{raw_val:.1f}"
        return str(raw_val)

    # Define all metrics to show in the comparison table
    COMPARE_METRICS = [
        ("Layers", lambda r, m: int(len(m)), None, False),
        ("Step wall", lambda r, m: r.get("step_wall", 0), "lower", True),
        ("Steps/second", lambda r, m: 1e6 / r.get("step_wall", 1) if r.get("step_wall", 0) > 0 else 0, "higher", True),
    ]

    # Phase times
    for label, key in [("AG forward", "ag_fwd_gpu_us"), ("Forward compute", "fwd_cmp_gpu_us"),
                        ("AG backward", "ag_bwd_gpu_us"), ("Backward compute", "bwd_cmp_gpu_us"),
                        ("Reduce scatter", "rs_gpu_us"), ("Optimizer", "optimizer_gpu_us"),
                        ("TP all-gather", "tp_ag_gpu_us"), ("TP all-reduce", "tp_ar_gpu_us"),
                        ("TP reduce-scatter", "tp_rs_gpu_us")]:
        COMPARE_METRICS.append((label, lambda r, m, k=key: r.get(k, 0), "lower", True))

    COMPARE_METRICS.append(("TP total", lambda r, m: r.get("tp_total_gpu_us", 0), "lower", True))
    COMPARE_METRICS.append(("Total GPU", lambda r, m: r.get("total_gpu_us", 0) + r.get("tp_total_gpu_us", 0), "lower", True))
    COMPARE_METRICS.append(("Total CPU", lambda r, m: r.get("total_cpu_us", 0), "lower", True))

    # Utilization & ratios
    COMPARE_METRICS.append(("GPU util", lambda r, m: sum(x.gpu_util for x in m) / max(len(m), 1), "higher", True))
    COMPARE_METRICS.append(("Compute-to-comm ratio", lambda r, m: sum(x.compute_to_comm_ratio for x in m) / max(len(m), 1), "higher", True))

    # MFU/HFU
    COMPARE_METRICS.append(("MFU", lambda r, m, tp=None: tp.get("mfu", 0) if tp else 0, "higher", True))
    COMPARE_METRICS.append(("HFU", lambda r, m, tp=None: tp.get("hfu", 0) if tp else 0, "higher", True))
    COMPARE_METRICS.append(("Tokens/sec/GPU", lambda r, m, tp=None: tp.get("tokens_per_second_per_gpu", 0) if tp else 0, "higher", True))

    # Comm breakdown
    COMPARE_METRICS.append(("Comm ratio", lambda r, m: r.get("comm_ratio", 0), "lower", True))
    COMPARE_METRICS.append(("FSDP comm ratio", lambda r, m: sum(x.fsdp_comm_ratio for x in m) / max(len(m), 1), "lower", True))
    COMPARE_METRICS.append(("TP comm ratio", lambda r, m: sum(x.tp_comm_ratio for x in m) / max(len(m), 1), "lower", True))

    # Overlap & pipeline
    COMPARE_METRICS.append(("Pipeline overlap", lambda r, m: r.get("overlap_ratio", 0), "higher", True))
    COMPARE_METRICS.append(("Serial efficiency", lambda r, m: r.get("serial_exec_efficiency", 0), "higher", True))
    COMPARE_METRICS.append(("Pipeline idle", lambda r, m: r.get("idle_ratio", 0), "lower", True))

    # Efficiency
    COMPARE_METRICS.append(("Exposed comm fraction", lambda r, m: sum(x.exposed_comm_fraction for x in m) / max(len(m), 1), "lower", True))
    COMPARE_METRICS.append(("AG overlap efficiency", lambda r, m: sum(x.ag_fwd_overlap_efficiency for x in m) / max(len(m), 1), "lower", True))
    COMPARE_METRICS.append(("Fwd TP overlap", lambda r, m: sum(x.fwd_comp_comm_overlap for x in m) / max(len(m), 1), "higher", True))
    COMPARE_METRICS.append(("Bwd TP overlap", lambda r, m: sum(x.bwd_comp_comm_overlap for x in m) / max(len(m), 1), "higher", True))

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
                if name in ("MFU", "HFU", "Tokens/sec/GPU"):
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

    # Build the HTML table with color coding
    thead_parts = ["<tr><th>Metric</th>"]
    for l in trace_labels:
        thead_parts.append(f'<th>{l}</th>')
    thead_parts.append('</tr>')

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
                    if _better(val, bval, direction):
                        cell_class = ' class="cmp-better"'
                    elif _worse(val, bval, direction):
                        cell_class = ' class="cmp-worse"'
            cells += f"<td{cell_class}>{formatted}</td>"
        table_rows += f"<tr>{cells}</tr>\n"

    # Bottleneck summary per trace
    bneck_sections = ""
    for i, (label, agg, metrics, steps, tp) in enumerate(all_results):
        issues = defaultdict(int)
        for m in metrics:
            for iss in Bottlenecks.detect(m):
                issues[iss] += 1
        if not issues:
            bneck_sections += f'<div class="comp-col"><h3>{label}</h3><p style="color:#2e7d32">No bottlenecks detected.</p></div>'
            continue
        rows = ""
        for iss, count in sorted(issues.items(), key=lambda x: -x[1]):
            pct = count / max(len(metrics), 1)
            short = iss.split("(")[0].strip()
            cls = "tag-high" if pct >= 0.5 else ("tag-med" if pct >= 0.2 else "tag-low")
            rows += f'<div style="margin:3px 0"><span class="tag {cls}">{short}</span> <span style="font-size:.75em;color:#888">{count}/{len(metrics)}</span></div>'
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

    body = _fill(
        _load_body("compare_body_template.html"),
        TITLE=title,
        FILES=", ".join(trace_files),
        BASELINE=trace_labels[0],
        TRACE_TABS=trace_tabs,
        SUMMARY_CARDS=summary_cards,
        MFU_CHART=mfu_chart,
        TABLE_HEAD="".join(thead_parts),
        TABLE_ROWS=table_rows,
        BOTTLENECK_SECTIONS=bneck_sections,
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
