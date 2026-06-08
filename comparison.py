"""Multi-trace benchmark comparison — side-by-side text table and CSV output.
Lets you compare compute-vs-communication ratios, pipeline overlap, async TP
efficiency, and bottleneck profiles across different model configurations
(TP degree, FSDP sharding, batch size, GPU count) in a single table.

Each row is one (trace, ProfilerStep) pair; columns include per-phase GPU time,
utilisation, compute-to-comm ratio, exposed comm, FSDP/TP communication breakdown,
overlap metrics, and the top-3 bottlenecks.  Also writes CSV for spreadsheet import.
"""

import csv
import os
import sys
from collections import defaultdict

from bottleneck_detector import Bottlenecks, _format_us


def _format_bottleneck_summary(metrics_list, aggregated) -> str:
    """One-line bottleneck summary (top 3 issues by prevalence) for the comparison table.
    ``; ``-separated for compact display — avoids multi-line wraps in the terminal table.
    """
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


def compare_traces(trace_files, output_file=None, model_config=None):
    """Process multiple traces and print a side-by-side comparison table.

    Analyses every ProfilerStep in each trace so the table shows one row per
    (trace, step) — useful for comparing steady-state across steps.

    Parameters
    ----------
    trace_files : list of str
        Paths to Chrome Trace JSON files.
    output_file : str or None
        If given, writes a CSV copy of the comparison table.
    model_config : ModelConfig or None
        Enables MFU/HFU/tokens-per-second columns in the comparison table.
    """
    from pipeline import process_all_steps

    trace_results = []
    for tf in trace_files:
        label = os.path.basename(tf)
        print(f"Processing {label}...", file=sys.stderr)
        step_results = process_all_steps(tf, model_config=model_config)
        if not step_results:
            print(f"  FAILED to load {tf}", file=sys.stderr)
            continue
        for step_name, agg, metrics_list, fsdp, report, text in step_results:
            trace_results.append((f"{label} ({step_name})", agg, metrics_list, fsdp, report, text))

    if not trace_results:
        print("No traces processed successfully.")
        sys.exit(1)

    # Build the comparison table row by row
    rows = []
    has_tp = model_config is not None
    headers = [
        "Trace", "Layers",
        "Step wall", "AG fwd", "Fwd cmp", "AG bwd", "Bwd cmp",
        "RS", "Opt", "TP total", "Total GPU",
        "Util", "CtC", "MFU", "HFU",
        "Comm", "FSDP comm", "TP comm",
        "Overlap", "Serial eff", "Idle",
        "ExpC", "AG Ovl",
        "F-Ovl", "B-Ovl",
        "Mem peak",
        "Bottlenecks",
    ]

    for label, agg, metrics_list, fsdp, report, text in trace_results:
        num_units = len(metrics_list)
        step_wall = agg.get("step_wall", 0)

        ag_fwd = agg.get("ag_fwd_gpu_us", 0)
        fwd_cmp = agg.get("fwd_cmp_gpu_us", 0)
        ag_bwd = agg.get("ag_bwd_gpu_us", 0)
        bwd_cmp = agg.get("bwd_cmp_gpu_us", 0)
        rs = agg.get("rs_gpu_us", 0)
        opt = agg.get("optimizer_gpu_us", 0)
        tp_total = agg.get("tp_total_gpu_us", 0)
        total_gpu = agg.get("total_gpu_us", 0) + tp_total

        avg_util = sum(m.gpu_util for m in metrics_list) / max(len(metrics_list), 1)
        mfu = report.throughput_metrics.get('mfu', 0) if hasattr(report, 'throughput_metrics') else 0
        hfu = report.throughput_metrics.get('hfu', 0) if hasattr(report, 'throughput_metrics') else 0
        # New universal metrics (compute-to-comm, exposed comm, AG overlap eff)
        ctc_vals = [m.compute_to_comm_ratio for m in metrics_list]
        non_inf = [v for v in ctc_vals if v != float('inf')]
        avg_ctc = sum(non_inf) / max(len(non_inf), 1) if non_inf else float('inf')
        avg_expc = sum(m.exposed_comm_fraction for m in metrics_list) / max(len(metrics_list), 1)
        avg_ag_ovl = sum(m.ag_fwd_overlap_efficiency for m in metrics_list) / max(len(metrics_list), 1)

        overlap_ratio = agg.get("overlap_ratio", 0)
        serial_eff = agg.get("serial_exec_efficiency", 0)
        idle_ratio = agg.get("idle_ratio", 0)

        avg_fwd_ovl = sum(m.fwd_comp_comm_overlap for m in metrics_list) / max(len(metrics_list), 1)
        avg_bwd_ovl = sum(m.bwd_comp_comm_overlap for m in metrics_list) / max(len(metrics_list), 1)

        comm_ratio = agg.get("comm_ratio", 0)
        fsdp_comm_ratio = sum(m.fsdp_comm_ratio for m in metrics_list) / max(len(metrics_list), 1)
        tp_comm_ratio = sum(m.tp_comm_ratio for m in metrics_list) / max(len(metrics_list), 1)

        mem_peak_gib = max((m.memory_peak for m in metrics_list if m.memory_has_data), default=0) / (1024**3)
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
            "CtC": avg_ctc,
            "MFU": mfu,
            "HFU": hfu if hfu else 0,
            "Comm": comm_ratio,
            "FSDP comm": fsdp_comm_ratio,
            "TP comm": tp_comm_ratio,
            "Overlap": overlap_ratio,
            "Serial eff": serial_eff,
            "Idle": idle_ratio,
            "ExpC": avg_expc,
            "AG Ovl": avg_ag_ovl,
            "F-Ovl": avg_fwd_ovl,
            "B-Ovl": avg_bwd_ovl,
            "Mem peak": mem_peak_gib,
            "Bottlenecks": bneck,
        })

    col_widths = {h: max(len(h), 8) for h in headers}
    col_widths["Trace"] = max(len(r["Trace"]) for r in rows)

    def _fmt_val(row, header) -> str:
        v = row[header]
        if isinstance(v, float):
            if header in ("Util", "Comm", "FSDP comm", "TP comm",
                          "Overlap", "Serial eff", "Idle",
                          "ExpC", "AG Ovl", "F-Ovl", "B-Ovl"):
                return f"{v:.1%}"
            elif header == "CtC":
                return "inf" if v == float('inf') else f"{v:.2f}x"
            elif header in ("MFU", "HFU"):
                return f"{v:.1%}" if v and v > 0 else "N/A"
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

    if output_file:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                flat = {h: _fmt_val(row, h) for h in headers}
                writer.writerow(flat)
        print(f"\nComparison written to {output_file}")
