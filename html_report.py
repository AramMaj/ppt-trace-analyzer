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

_HERE = os.path.dirname(__file__)
_PAGE_TEMPLATE = None

COLORS = {
    'fwd_cmp': '#4e79a7', 'bwd_cmp': '#e15759', 'ag': '#76b7b2',
    'rs': '#f28e2b', 'opt': '#59a14f', 'tp_ag': '#af7aa1',
    'tp_rs': '#ff9da7', 'tp_ar': '#9c755f',
}



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
    html = body
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"HTML report written to {output_path}")
