""" 
CLI entry point for initiating trace anaylsis. 

Usage 
    python main.py [mode] [traceFilePath] [info] [--output filename]

    [mode] - Specifiaction of analysis type 
        " "          - Bottleneck report + Per-layer GPU time breakdown 
        "--timeline" - ASCII Gantt chart of FSDP2 pipeline stagger
        "--compare"" - Side-by-Side benchmark table of two or more traces
        "--annotate" - Annotated Trace File in Chrome Trace Format 

    [info] - Specification of model architecture and training details  
        "--hidden-dim", "--num-layers", "--num-heads", "--seq-len", 
        "--vocab-size", "--batch-size", "--batch-size", 
        "--activation-checkpointing", "--precision-bits", "--gpu-peak-tflops", 
        "--gpu-hbm-bw", "--intermediate-dim", "--num-kv-heads"
"""

import sys
import os

from trace_parser import TraceParser
from fsdp_detector import StandardFSDPDetector
from bottleneck_detector import Report

from pipeline import select_profiler_step, sanitize_optimizer
from timeline import print_timeline
from comparison import compare_traces


MODEL_FLAGS = {
    "--hidden-dim": "hidden_dim",
    "--num-layers": "num_layers",
    "--num-heads": "num_heads",
    "--seq-len": "seq_len",
    "--vocab-size": "vocab_size",
    "--batch-size": "batch_size",
    "--num-gpus": "num_gpus",
    "--activation-checkpointing": "activation_checkpointing",
    "--precision-bits": "precision_bits",
    "--gpu-peak-tflops": "gpu_peak_tflops",
    "--gpu-hbm-bw": "gpu_hbm_bw_gbps",
    "--intermediate-dim": "intermediate_dim",
    "--num-kv-heads": "num_kv_heads",
}


def _parse_model_config(argv_slice):
    """Extract MODEL_FlAGS from input arguments"""
    kwargs = {}
    i = 0
    while i < len(argv_slice):
        flag = argv_slice[i]
        if flag in MODEL_FLAGS and i + 1 < len(argv_slice):
            key = MODEL_FLAGS[flag]
            val = argv_slice[i + 1]
            if key == "activation_checkpointing":
                kwargs[key] = float(val)
            elif key == "precision_bits":
                kwargs[key] = int(val)
            elif key in ("gpu_peak_tflops", "gpu_hbm_bw_gbps"):
                kwargs[key] = float(val)
            else:
                kwargs[key] = int(val)
            i += 2
            continue
        break
    return i, kwargs


def main():
    """CLI entry point to the program"""

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Analyze a single trace:     python main.py <trace.json> [--output report.txt]")
        print("                               [--hidden-dim N] [--num-layers N] [--num-heads N] [--seq-len N]")
        print("                               [--vocab-size N] [--batch-size N] [--num-gpus N] [--activation-checkpointing F]")
        print("  Compare multiple traces:    python main.py --compare trace1.json trace2.json ... [--output comparison.csv]")
        print("                               [--hidden-dim N] [--num-layers N] [--num-heads N] [--seq-len N]")
        print("                               [--vocab-size N] [--batch-size N] [--num-gpus N] [--activation-checkpointing F]")
        print("                               [--html] (generate HTML comparison page instead of text table)")
        print("  Annotate trace with phases: python main.py --annotate <trace.json> [--output annotated.json]")
        print("                               [--hidden-dim N] [--num-layers N] [--num-heads N] [--seq-len N]")
        print("                               [--vocab-size N] [--batch-size N] [--num-gpus N] [--activation-checkpointing F]")
        print("  Text phase timeline:        python main.py --timeline <trace.json>")
        print("  HTML report:                python main.py --html <trace.json> [--output report.html]")
        print("                               [--hidden-dim N] [--num-layers N] etc.")
        sys.exit(1)

    if sys.argv[1] == "--html":
        trace_file = None
        output_file = None
        mc_kwargs = {}
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
                continue
            if sys.argv[i] in MODEL_FLAGS:
                consumed, kw = _parse_model_config(sys.argv[i:])
                mc_kwargs.update(kw)
                i += consumed
                continue
            elif not sys.argv[i].startswith("--"):
                trace_file = sys.argv[i]
            i += 1
        if trace_file is None:
            print("Error: --html requires a trace file.")
            sys.exit(1)
        from html_report import generate_html_report
        from bottleneck_detector import ModelConfig
        cfg = ModelConfig(**mc_kwargs) if mc_kwargs else None
        generate_html_report(trace_file, output_path=output_file, model_config=cfg)
        return

    if sys.argv[1] == "--annotate":
        trace_file = None
        output_file = None
        mc_kwargs = {}
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
                continue
            if sys.argv[i] in MODEL_FLAGS:
                consumed, kw = _parse_model_config(sys.argv[i:])
                mc_kwargs.update(kw)
                i += consumed
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
        from bottleneck_detector import ModelConfig
        cfg = ModelConfig(**mc_kwargs) if mc_kwargs else None
        if cfg and not cfg.is_configured:
            print("Warning: --hidden-dim, --num-layers, and --batch-size are required for MFU/HFU.")
        annotate_trace(trace_file, output_file, model_config=cfg)
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
        parser.attribute_gpu_kernel_with_logical_operation(roots)
        parser.attribute_memory(roots)
        step_start, step_end, filter_start, filter_end = select_profiler_step(roots, parser)
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
        ac2g_supplement = _get_ac2g_bwd_supplement(parser.all_events, step_index, num_steps)
        for unit in fsdp.units:
            unit.ag_bwd_supplement_us = ac2g_supplement.get(unit.layer_name, 0.0)
        report = Report(fsdp, roots, output_path=None, ag_per_layer=ag_per_layer)
        report.generate_report()
        print_timeline(fsdp, report)
        return

    if sys.argv[1] == "--compare":
        trace_files = []
        output_file = None
        mc_kwargs = {}
        html_mode = False
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
                continue
            if sys.argv[i] == "--html":
                html_mode = True
                i += 1
                continue
            if sys.argv[i] in MODEL_FLAGS:
                consumed, kw = _parse_model_config(sys.argv[i:])
                mc_kwargs.update(kw)
                i += consumed
                continue
            elif not sys.argv[i].startswith("--"):
                trace_files.append(sys.argv[i])
            i += 1
        if len(trace_files) < 2:
            print("Error: --compare requires at least 2 trace files.")
            sys.exit(1)
        from bottleneck_detector import ModelConfig
        cfg = ModelConfig(**mc_kwargs) if mc_kwargs else None
        if html_mode:
            from html_report import generate_compare_html
            generate_compare_html(trace_files, output_path=output_file, model_config=cfg)
        else:
            compare_traces(trace_files, output_file, model_config=cfg)
        return

    # Default: single-trace analysis
    trace_file = sys.argv[1]
    output_file = None
    mc_kwargs = {}
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
            continue
        if sys.argv[i] in MODEL_FLAGS:
            consumed, kw = _parse_model_config(sys.argv[i:])
            mc_kwargs.update(kw)
            i += consumed
            continue
        i += 1

    from bottleneck_detector import ModelConfig
    cfg = ModelConfig(**mc_kwargs) if mc_kwargs else None
    if cfg and not cfg.is_configured:
        print("Warning: --hidden-dim, --num-layers, and --batch-size are required for MFU/HFU.")

    print(f"Loading trace from {trace_file}...")
    parser = TraceParser(trace_file)
    if not parser.load():
        sys.exit(1)
    print(f"Loaded {len(parser.cpu_events)} CPU, {len(parser.gpu_events)} GPU, {len(parser.memory_events)} memory events.")

    roots = parser.build_tree()
    # Attribute GPU kernels BEFORE step filtering — GPU kernels for early layers'
    # backward can finish hundreds of ms after the CPU ProfilerStep ends (pipelined
    # execution). Attributing against the full GPU event set ensures their ext-ID
    # correlation succeeds even after step-level filtering discards the events.
    parser.attribute_gpu_kernel_with_logical_operation(roots)
    parser.attribute_memory(roots)
    step_start, step_end, filter_start, filter_end = select_profiler_step(roots, parser)
    from bottleneck_detector import _compute_ag_per_layer
    ag_range = (filter_start, filter_end) if filter_start is not None else None
    ag_per_layer = _compute_ag_per_layer(roots, time_range=ag_range)

    detector = StandardFSDPDetector(gpu_events=parser.gpu_events)
    fsdp = detector.extract_fsdp_phases(roots)
    sanitize_optimizer(fsdp, step_start, step_end)
    print(f"Detected {len(fsdp.units)} FSDP units.")

    from trace_annotator import _get_ac2g_bwd_supplement, _find_profiler_steps
    all_steps = _find_profiler_steps(trace_file)
    step_index = len(all_steps) - 1 if len(all_steps) > 1 else 0
    num_steps = len(all_steps) if all_steps else 1
    ac2g_supplement = _get_ac2g_bwd_supplement(parser.all_events, step_index, num_steps)
    for unit in fsdp.units:
        unit.ag_bwd_supplement_us = ac2g_supplement.get(unit.layer_name, 0.0)

    report = Report(fsdp, roots, output_path=output_file, model_config=cfg,
                    ag_per_layer=ag_per_layer)
    text, markers = report.generate_report()
    print(text)

    if output_file:
        print(f"\nReport written to {output_file}")


if __name__ == "__main__":
    main()
