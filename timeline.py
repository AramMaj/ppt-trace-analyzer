"""ASCII Gantt-chart timeline — shows FSDP2 pipeline stagger and serial bubbles
at a glance.  The text report gives precise per-phase GPU times, but the Gantt
reveals which layers are serialised, where backward-phase starts, and where
pipeline idle gaps / bottlenecks occur, without needing chrome://tracing.

One row per layer (34 for 8B TP, 8 for async TP).  Phase characters:
  A = all-gather forward (NCCL unshard)
  F = forward compute (attn/MLP/norm)
  a = all-gather backward (re-shard)
  B = backward compute (autograd)
  R = reduce-scatter (grad sync)
  O = optimizer
  ! = bottleneck marker at the longest phase midpoint
"""

from bottleneck_detector import Bottlenecks, _format_us, _phase_wall_span
from trace_annotator import _get_phase_spans

TIMELINE_BAR_CHARS = {
    'AG fwd': 'A', 'Fwd cmp': 'F', 'AG bwd': 'a',
    'Bwd cmp': 'B', 'RS': 'R', 'Optimizer': 'O',
}


def print_timeline(fsdp, report):
    """Compact ASCII Gantt chart — one row per layer, 80-char time-scaled bar, ``!`` marks bottlenecks.
    """
    n_layers = len(fsdp.units)
    if n_layers == 0:
        print("  No FSDP units to show.")
        return

    # Collect all phase intervals with their layer index, label, start, end
    intervals = []
    for idx, unit in enumerate(fsdp.units):
        for label, nodes, stored_span in [
            ('AG fwd', unit.all_gather_fwd, None),
            ('Fwd cmp', unit.fwd_compute, unit.fwd_compute_span),
            ('AG bwd', unit.all_gather_bwd, None),
            ('Bwd cmp', unit.bwd_compute, unit.bwd_compute_span),
            ('RS', unit.reduce_scatter, None),
        ]:
            if stored_span is not None:
                intervals.append((idx, label, stored_span[0], stored_span[1]))
            else:
                span = _phase_wall_span(nodes)
                if span:
                    intervals.append((idx, label, span[0], span[1]))

    if not intervals:
        print("  No phase intervals found.")
        return

    # Global time range for normalising the 80-col bar
    t_min = min(i[2] for i in intervals)
    t_max = max(i[3] for i in intervals)
    span_total = t_max - t_min
    if span_total <= 0:
        print("  Zero-length timeline.")
        return

    BAR_WIDTH = 80
    COL_WIDTH = max(len(u.layer_name) for u in fsdp.units) + 1

    print()
    print("─" * (COL_WIDTH + BAR_WIDTH + 4))
    print("Phase Timeline (each character ≈ {:.1f}ms)".format(span_total / (BAR_WIDTH * 1000)))
    print("─" * (COL_WIDTH + BAR_WIDTH + 4))

    # Scale tick labels at 5 evenly-spaced positions
    scale_lines = []
    num_ticks = 5
    for t in range(num_ticks + 1):
        pct = t / num_ticks
        ts = t_min + pct * span_total
        pos = int(pct * BAR_WIDTH)
        scale_lines.append((pos, _format_us(ts)))
    scale_str = ""
    prev_end = 0
    for pos, label in scale_lines:
        gap = pos - prev_end
        if gap <= 0:
            continue
        if len(label) <= gap:
            scale_str += " " * (gap - len(label)) + label
        else:
            scale_str += " " * gap
        prev_end = pos
    print(" " * COL_WIDTH + scale_str)

    # One row per layer — draw phase characters, then overlay bottleneck marker
    for idx, unit in enumerate(fsdp.units):
        name = unit.layer_name
        line_chars = [' '] * BAR_WIDTH

        for label, start, end in [(l, s, e) for (li, l, s, e) in intervals if li == idx]:
            col_start = max(0, int((start - t_min) / span_total * BAR_WIDTH))
            col_end = min(BAR_WIDTH - 1, int((end - t_min) / span_total * BAR_WIDTH))
            ch = TIMELINE_BAR_CHARS.get(label, '#')
            for c in range(col_start, min(col_end + 1, BAR_WIDTH)):
                line_chars[c] = ch

        # Overlay '!' at the midpoint of the longest phase if bottlenecked
        timeline_str = ''.join(line_chars)
        for m in report.metrics_list:
            if m.layer_name == unit.layer_name:
                issues = Bottlenecks.detect(m)
                if issues:
                    phases = _get_phase_spans(unit)
                    if phases:
                        longest = max(phases, key=lambda p: p[2] - p[1])
                        bpos = int((longest[1] - t_min) / span_total * BAR_WIDTH)
                        if 0 <= bpos < BAR_WIDTH:
                            if timeline_str[bpos] == ' ':
                                timeline_str = timeline_str[:bpos] + '!' + timeline_str[bpos + 1:]
                break

        print(f"{name:<{COL_WIDTH}} {timeline_str}")

    print("─" * (COL_WIDTH + BAR_WIDTH + 4))
    legend_parts = [f"{ch}={name}" for name, ch in TIMELINE_BAR_CHARS.items()]
    legend_parts.append("!=bottleneck")
    print(" " * COL_WIDTH + "  ".join(legend_parts))

    bneck_count = 0
    for m in report.metrics_list:
        issues = Bottlenecks.detect(m)
        if issues:
            bneck_count += 1
            if bneck_count <= 5:
                print(f"  {m.layer_name}: {'; '.join(issues)}")
    if bneck_count > 5:
        print(f"  ... and {bneck_count - 5} more layers with bottlenecks")
    if bneck_count == 0:
        print("  No bottlenecks detected")
    print()
