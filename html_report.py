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



def generate_html_report(trace_file, output_path=None, model_config=None):
    result = process_trace(trace_file, model_config=model_config)
    if result is None: return
    aggregated, metrics_list, fsdp, report, text = result
    if output_path is None:
        base, _ = os.path.splitext(trace_file)
        output_path = f"{base}.html"
    title = f"Trace Analysis — {os.path.basename(trace_file)}"
    body = f"""<h1>{title}</h1><p class="subtitle">{os.path.basename(trace_file)} — {len(metrics_list)} layers</p>"""
    with open(output_path, 'w') as f: f.write(body)
    print(f"HTML report written to {output_path}")
