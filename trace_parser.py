#!/usr/bin/env python3
"""
Trace parser for PyTorch Profiler JSON traces. (Focus: FSDP)

usage:
    Single trace analysis:
        python final_trace_analyzer.py <trace.json> [--output report.txt]
    
    Benchmark (CSV comparison):
        python final_trace_analyzer.py --compare trace1.json trace2.json --output comparison.csv [--op OPERATION_NAME]
"""

import json
import sys
import heapq
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Iterator


# ----------------------------------------------------------------------
# Helper function: iterate through the tree of logical operations
def iter_nodes(roots: List['OperationWrapper']) -> Iterator['OperationWrapper']:
    """Iterate over all nodes in the tree (depth-first, iterative)."""
    stack = list(roots)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


# -----------------------------------------------------------------------
# Data structure for a logical operation / node in the call tree
@dataclass
class OperationWrapper:
    name: str
    start_time: float
    end_time: float
    cpu_duration: float
    gpu_duration: float = 0.0
    memory_delta: int = 0
    children: List['OperationWrapper'] = field(default_factory=list)
    external_ids: Set[int] = field(default_factory=set)
    direct_gpu_duration: float = 0.0
    raw_event: Optional[Dict] = None

    @property
    def total_time(self) -> float:
        return max(self.cpu_duration, self.gpu_duration)

    @property
    def exclusive_cpu(self) -> float:
        return self.cpu_duration - sum(c.cpu_duration for c in self.children)

    @property
    def exclusive_gpu(self) -> float:
        return self.gpu_duration - sum(c.gpu_duration for c in self.children)


# ------------------------------------------------------------------------
# Async dependency resolver (CUDA events, stream wait, async intervals)
class AsyncDependencyResolver:
    def __init__(self):
        self.event_record_times: Dict[Tuple[int, int], float] = {}
        self.pending_intervals: Dict[str, Tuple[float, Dict]] = {}
        self.dependencies: List[Tuple[Dict, Dict]] = []

    def process_event(self, ev: Dict):
        ph = ev.get('ph', '')
        cat = ev.get('cat', '')
        name = ev.get('name', '')
        args = ev.get('args', {})
        ev_id = args.get('id') or ev.get('id')
        ts = ev.get('ts', 0)

        if 'cudaEventRecord' in name or (cat == 'cuda_runtime' and 'record' in name.lower()):
            event_ptr = args.get('event')
            stream_ptr = args.get('stream')
            if event_ptr is not None:
                self.event_record_times[(event_ptr, stream_ptr)] = ts
            return

        if 'cudaStreamWaitEvent' in name or (cat == 'cuda_runtime' and 'wait' in name.lower()):
            event_ptr = args.get('event')
            for (ev_ptr, _), rec_ts in self.event_record_times.items():
                if ev_ptr == event_ptr:
                    self.pending_intervals[f"wait_{ts}_{event_ptr}"] = (rec_ts, ev)
                    break
            return

        if ph in ('b', 's') and ev_id is not None:
            self.pending_intervals[ev_id] = (ts, ev)
        elif ph in ('e', 'f') and ev_id is not None:
            if ev_id in self.pending_intervals:
                start_ts, start_ev = self.pending_intervals.pop(ev_id)
                self.dependencies.append((start_ev, ev))
                ev['_async_start'] = start_ts
                ev['_async_end'] = ts
                start_ev['_async_interval'] = (start_ts, ts)

        if ph == 'f' and 'bp' in args and ev_id is not None:
            if args.get('bp') == 's':
                self.pending_intervals[f"flow_{ev_id}"] = (ts, ev)
            elif args.get('bp') == 'e':
                key = f"flow_{ev_id}"
                if key in self.pending_intervals:
                    start_ts, start_ev = self.pending_intervals.pop(key)
                    self.dependencies.append((start_ev, ev))
                    start_ev['_async_interval'] = (start_ts, ts)

    def resolve_dependencies(self, all_events: List[Dict]):
        for ev in all_events:
            self.process_event(ev)

    def get_parent_from_async_only(self, gpu_event: Dict) -> Optional[Dict]:
        ts = gpu_event.get('ts', 0)
        for start_ev, _ in self.dependencies:
            interval = start_ev.get('_async_interval')
            if interval and interval[0] <= ts <= interval[1]:
                return start_ev
        return None


# ------------------------------------------------------------------------
# Trace parser class
class TraceParser:
    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.data = None
        self.events_by_pid_tid: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
        self.cpu_events: List[Dict] = []
        self.gpu_events: List[Dict] = []
        self.all_events: List[Dict] = []
        self.memory_events: List[Dict] = []
        self.distributed_info: Dict = {}
        self.async_resolver = AsyncDependencyResolver()

    def load(self):
        with open(self.trace_file, 'r') as f:
            self.data = json.load(f)
        self.distributed_info = self.data.get('distributedInfo', {})
        for ev in self.data.get('traceEvents', []):
            self.all_events.append(ev)
            pid, tid = ev.get('pid', 0), ev.get('tid', 0)
            cat = ev.get('cat', '')
            ph = ev.get('ph', '')
            name = ev.get('name', '')

            if ph == 'X' and cat in ('cpu_op', 'user_annotation'):
                self.cpu_events.append(ev)
                self.events_by_pid_tid[(pid, tid)].append(ev)
            elif ph == 'X' and self._is_gpu_event(cat):
                self.gpu_events.append(ev)
            elif ph in ('i', 'C') and ('memory' in cat.lower() or name == "[memory]"):
                self.memory_events.append(ev)

        self.async_resolver.resolve_dependencies(self.all_events)

    @staticmethod
    def _is_gpu_event(cat: str) -> bool:
        GPU_CATEGORIES = {
            'kernel',
            'gpu_memcpy',
            'gpu_memset',
            'cuda_runtime',
            'cuda_driver',
            'gpu_user_annotation',
        }
        return cat in GPU_CATEGORIES or 'gpu' in cat.lower() or 'cuda' in cat.lower()

    def build_tree(self) -> List[OperationWrapper]:
        all_roots = []
        for (pid, tid), events in self.events_by_pid_tid.items():
            events_sorted = sorted(events, key=lambda e: e['ts'])
            stack = []
            for ev in events_sorted:
                start = ev['ts']
                dur = ev.get('dur', 0)
                node = OperationWrapper(
                    name=ev['name'],
                    start_time=start,
                    end_time=start + dur,
                    cpu_duration=dur,
                    raw_event=ev
                )
                ext_id = ev.get('args', {}).get('External id')
                if ext_id is not None:
                    node.external_ids.add(ext_id)

                # Pop events that ended before this one starts
                while stack and stack[-1][0]['ts'] + stack[-1][0].get('dur', 0) <= start:
                    stack.pop()
                if stack:
                    stack[-1][1].children.append(node)
                else:
                    all_roots.append(node)
                stack.append((ev, node))
        return all_roots

    def attribute_gpu_time_with_dependencies(self, roots: List[OperationWrapper]):
        # Step 1: direct External id mapping
        ext_to_gpu = defaultdict(float)
        for gpu_ev in self.gpu_events:
            ext_id = gpu_ev.get('args', {}).get('External id')
            if ext_id is not None:
                ext_to_gpu[ext_id] += gpu_ev.get('dur', 0)

        # Step 2: collect all logical nodes using iterative traversal
        all_nodes = list(iter_nodes(roots))
        event_to_node = {id(node.raw_event): node for node in all_nodes if node.raw_event is not None}

        # Step 3: find parent CPU event for each GPU event
        gpu_parent_pairs = []

        # GPU events with External id are handled exclusively via ext_to_gpu (step 5)
        remaining_gpu = []
        for gpu_ev in self.gpu_events:
            ext_id = gpu_ev.get('args', {}).get('External id')
            if ext_id is not None:
                continue  # skip, handled in step 5

            parent = self.async_resolver.get_parent_from_async_only(gpu_ev)
            if parent is not None:
                gpu_parent_pairs.append((gpu_ev, parent))
            else:
                remaining_gpu.append(gpu_ev)

        # 3b: sweep for the rest
        if remaining_gpu:
            global count
            count += 1 
            all_cpu = []
            for evlist in self.events_by_pid_tid.values():
                all_cpu.extend(evlist)
            all_cpu.sort(key=lambda e: e['ts'])
            remaining_gpu.sort(key=lambda e: e['ts'])

            heap = []
            cpu_idx = 0
            n_cpu = len(all_cpu)

            for gpu_ev in remaining_gpu:
                ts = gpu_ev['ts']
                while cpu_idx < n_cpu and all_cpu[cpu_idx]['ts'] <= ts:
                    ce = all_cpu[cpu_idx]
                    dur = ce.get('dur', 0)
                    end = ce['ts'] + dur
                    # Use negative start time to get the innermost (most recently started) active interval
                    heapq.heappush(heap, (-ce['ts'], end, ce))
                    cpu_idx += 1
                while heap and heap[0][1] < ts:
                    heapq.heappop(heap)
                if heap:
                    gpu_parent_pairs.append((gpu_ev, heap[0][2]))

        # Step 4: attribute GPU time to nodes
        for gpu_ev, parent_cpu in gpu_parent_pairs:
            node = event_to_node.get(id(parent_cpu))
            if node is not None:
                node.direct_gpu_duration += gpu_ev.get('dur', 0)

        # Step 5: add External id contributions (no double counting)
        for node in all_nodes:
            for ext_id in node.external_ids:
                node.direct_gpu_duration += ext_to_gpu.get(ext_id, 0)

        # Step 6: propagate GPU time upwards
        def compute(node):
            child_gpu = sum(compute(ch) for ch in node.children)
            node.gpu_duration = node.direct_gpu_duration + child_gpu
            return node.gpu_duration
        for root in roots:
            compute(root)

    def attribute_memory(self, roots: List[OperationWrapper]):
        if not self.memory_events:
            print("Warning: No memory events found. Memory profiling may be disabled in the trace.")
            return

        all_nodes = list(iter_nodes(roots))
        all_nodes.sort(key=lambda n: n.start_time)

        mem_events_sorted = sorted(self.memory_events, key=lambda e: e['ts'])
        heap = []
        node_idx = 0
        n_nodes = len(all_nodes)

        for mem_ev in mem_events_sorted:
            ts = mem_ev['ts']
            while node_idx < n_nodes and all_nodes[node_idx].start_time <= ts:
                node = all_nodes[node_idx]
                # Use negative start time to get the innermost active node
                heapq.heappush(heap, (-node.start_time, node.end_time, node))
                node_idx += 1
            while heap and heap[0][1] < ts:
                heapq.heappop(heap)
            if heap:
                deepest = heap[0][2]
                args = mem_ev.get('args', {})
                size = args.get('Bytes', args.get('size', args.get('bytes', 0)))
                device_type = args.get('Device Type', -1)
                if device_type == 1:   # GPU memory only
                    deepest.memory_delta += size

        # Propagate memory deltas upward
        def propagate(node):
            total = node.memory_delta
            for ch in node.children:
                total += propagate(ch)
            node.memory_delta = total
            return total
        for root in roots:
            propagate(root)

    def get_aggregated_metrics(self, roots: List[OperationWrapper], op_filter: Optional[str] = None) -> Dict[str, Dict]:
        all_nodes = list(iter_nodes(roots))
        agg = defaultdict(lambda: {'count': 0, 'total_cpu_us': 0.0, 'total_gpu_us': 0.0, 'total_mem_bytes': 0})

        for node in all_nodes:
            if op_filter and op_filter not in node.name:
                continue
            agg[node.name]['count'] += 1
            agg[node.name]['total_cpu_us'] += node.cpu_duration
            agg[node.name]['total_gpu_us'] += node.gpu_duration
            agg[node.name]['total_mem_bytes'] += node.memory_delta

        for name, data in agg.items():
            c = data['count']
            data['avg_cpu_us'] = data['total_cpu_us'] / c if c else 0
            data['avg_gpu_us'] = data['total_gpu_us'] / c if c else 0
            data['avg_mem_bytes'] = data['total_mem_bytes'] / c if c else 0

        return dict(agg)

# ------------------------------------------------------------------------
# FSDP specific analysis
def extract_fsdp_phases_aggregated(roots: List[OperationWrapper]) -> Dict[str, Dict[str, float]]:
    """
    Returns a nested dict: unit -> { phase: total_gpu_us }
    Propagates unit (layer) from parent to children.
    Phases: "All-Gather (AG)", "Reduce-Scatter (RS)","Forward Comp (FWD)", "Backward Comp (BWD)"
    """
    agg = defaultdict(lambda: defaultdict(float))

    def collect(node, current_unit=None):
        # Determine unit from this node (if any)
        match = re.search(r"model\.layers\.(\d+)", node.name)
        if match:
            current_unit = f"Layer {match.group(1)}"

        phase = None
        name_lower = node.name.lower()

        # All-Gather
        if "all_gather" in name_lower:
            phase = "All-Gather (AG)"
        # Reduce-Scatter
        elif "reduce_scatter" in name_lower:
            phase = "Reduce-Scatter (RS)"
        elif node.raw_event and node.raw_event.get('args', {}).get('Collective name') == "_reduce_scatter_base":
            phase = "Reduce-Scatter (RS)"
        # Backward
        elif "fsdp::pre_backward" in name_lower or "fsdp::backward_prefetch" in name_lower:
            phase = "Backward Comp (BWD)"
        # Forward: detect FSDP::pre_forward and compute forward compute time
        elif "fsdp::pre_forward" in name_lower and current_unit:
            # Forward compute time: sum GPU durations of children that are not collectives
            forward_gpu = 0.0
            for child in node.children:
                child_name = child.name.lower()
                if "all_gather" not in child_name and "reduce_scatter" not in child_name:
                    forward_gpu += child.gpu_duration
            if forward_gpu > 0:
                agg[current_unit]["Forward Comp (FWD)"] += forward_gpu
            else:
                agg[current_unit]["Forward Comp (FWD)"] += node.gpu_duration
            phase = None  # already handled
        # Note: ProfilerStep is NOT treated as forward phase because it spans the whole step and it would massively inflate times.

        if phase and current_unit:
            agg[current_unit][phase] += node.gpu_duration

        for child in node.children:
            collect(child, current_unit)

    for root in roots:
        collect(root)
    return agg

def get_fsdp_timeline_aggregated_string(agg: Dict[str, Dict[str, float]]) -> str:
    if not agg:
        return "No FSDP phases found."
    lines = []
    lines.append("\n" + "="*90)
    lines.append("FSDP Phase Summary (aggregated GPU time per layer)")
    lines.append("="*90)
    lines.append(f"{'Layer':<12} {'All-Gather (ms)':>18} {'Reduce-Scatter (ms)':>20} "
                 f"{'Forward (ms)':>15} {'Backward (ms)':>15}")
    lines.append("-"*90)

    def layer_key(unit):
        try:
            return int(unit.split()[1])
        except:
            return 0

    for unit in sorted(agg.keys(), key=layer_key):
        data = agg[unit]
        lines.append(f"{unit:<12} {data.get('All-Gather (AG)', 0)/1e3:>18.2f} "
                     f"{data.get('Reduce-Scatter (RS)', 0)/1e3:>20.2f} "
                     f"{data.get('Forward Comp (FWD)', 0)/1e3:>15.2f} "
                     f"{data.get('Backward Comp (BWD)', 0)/1e3:>15.2f}")

    total_ag = sum(d.get('All-Gather (AG)', 0) for d in agg.values())
    total_rs = sum(d.get('Reduce-Scatter (RS)', 0) for d in agg.values())
    total_fwd = sum(d.get('Forward Comp (FWD)', 0) for d in agg.values())
    total_bwd = sum(d.get('Backward Comp (BWD)', 0) for d in agg.values())
    lines.append("-"*90)
    lines.append(f"{'TOTAL (all layers)':<12} {total_ag/1e3:>18.2f} {total_rs/1e3:>20.2f} "
                 f"{total_fwd/1e3:>15.2f} {total_bwd/1e3:>15.2f}")
    return "\n".join(lines)

def get_fsdp_chronological_timeline(roots: List[OperationWrapper]) -> str:
    timeline = []

    def collect_timeline(node, current_unit=None, unit_idx=None):
        # Layer-Nummer extrahieren (z.B. "0" aus "Layer 0")
        match = re.search(r"model\.layers\.(\d+)", node.name)
        if match:
            unit_idx = match.group(1)
            current_unit = f"Layer {unit_idx}"

        name_lower = node.name.lower()
        phase_label = None
        
        # Phasen-Erkennung mit Nummerierung
        if unit_idx is not None:
            if "all_gather" in name_lower:
                phase_label = f"AG (Layer {unit_idx})"
            elif "reduce_scatter" in name_lower or (node.raw_event and node.raw_event.get('args', {}).get('Collective name') == "_reduce_scatter_base"):
                phase_label = f"RS (Layer {unit_idx})"
            elif "fsdp::pre_backward" in name_lower or "fsdp::backward_prefetch" in name_lower:
                phase_label = f"Backward {unit_idx}"
            elif "fsdp::pre_forward" in name_lower:
                phase_label = f"Forward {unit_idx}"

        if phase_label:
            timeline.append({
                "ts": node.start_time,
                "phase": phase_label,
                "dur": node.gpu_duration
            })

        for child in node.children:
            collect_timeline(child, current_unit, unit_idx)

    for root in roots:
        collect_timeline(root)

    # Nach Zeit sortieren
    timeline.sort(key=lambda x: x["ts"])

    if not timeline:
        return "No FSDP events found."

    lines = []
    lines.append("\n" + "="*95)
    lines.append("CHRONOLOGICAL FSDP TIMELINE (With Layer Matching)")
    lines.append("="*95)
    lines.append(f"{'Relative Start (ms)':>20} | {'Phase / Operation':<40} | {'GPU Dur (ms)':>15}")
    lines.append("-" * 95)

    first_ts = timeline[0]["ts"]
    for event in timeline:
        rel_ts = (event["ts"] - first_ts) / 1000
        lines.append(f"{rel_ts:>20.2f} | {event['phase']:<40} | {event['dur']/1000:>15.2f}")
    
    return "\n".join(lines)


# ------------------------------------------------------------------------
# bottleneck detection and reporting
class BottleneckDetector:
    @staticmethod
    def detect(node: OperationWrapper) -> List[str]:
        bottlenecks = []
        total = node.total_time
        if total > 0:
            if node.cpu_duration > 1.5 * node.gpu_duration and node.exclusive_cpu > 0.1 * total:
                bottlenecks.append("CPU-bound (high CPU overhead)")
            if node.gpu_duration < 0.5 * node.cpu_duration and node.gpu_duration > 0:
                bottlenecks.append("GPU underutilization (GPU idle waiting)")
            if abs(node.memory_delta) > 1024 * 1024 * 1024:
                bottlenecks.append("High memory footprint")
        for ch in node.children:
            if "all_gather" in ch.name.lower() or "allgather" in ch.name.lower():
                if ch.gpu_duration > 0.3 * node.gpu_duration:
                    bottlenecks.append("Communication bottleneck (all-gather dominates)")
        return bottlenecks

def format_time(us: float) -> str:
    if us < 1e3:
        return f"{us:.2f} µs"
    elif us < 1e6:
        return f"{us/1e3:.2f} ms"
    else:
        return f"{us/1e6:.2f} s"

def format_memory(bytes_: int) -> str:
    ab = abs(bytes_)
    if ab < 1024:
        return f"{bytes_} B"
    elif ab < 1024**2:
        return f"{bytes_/1024:.2f} KB"
    elif ab < 1024**3:
        return f"{bytes_/1024**2:.2f} MB"
    else:
        return f"{bytes_/1024**3:.2f} GB"

def get_top_k_string(roots: List[OperationWrapper], k: int = 10) -> str:
    all_nodes = list(iter_nodes(roots))
    all_nodes.sort(key=lambda n: n.total_time, reverse=True)
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"TOP {k} MOST EXPENSIVE OPERATIONS (async dependencies resolved)")
    lines.append(f"{'='*80}")
    lines.append(f"{'Name':<50} {'CPU':>12} {'GPU':>12} {'Memory Δ':>12} {'Bottleneck'}")
    lines.append(f"{'-'*80}")
    for node in all_nodes[:k]:
        bottle = BottleneckDetector.detect(node)
        bottle_str = bottle[0] if bottle else "-"
        lines.append(f"{node.name[:50]:<50} {format_time(node.cpu_duration):>12} "
                     f"{format_time(node.gpu_duration):>12} {format_memory(node.memory_delta):>12} {bottle_str}")
    return "\n".join(lines)

def get_flame_like_string(roots: List[OperationWrapper], max_depth: int = 3, indent: int = 0) -> str:
    lines = []
    if indent == 0:
        lines.append(f"\n{'='*80}")
        lines.append("FLAME-GRAPH STYLE BREAKDOWN (async dependencies accounted)")
        lines.append(f"{'='*80}")
    for node in roots:
        prefix = "  " * indent
        bottle = BottleneckDetector.detect(node)
        hint = f"  <-- {bottle[0]}" if bottle else ""
        lines.append(f"{prefix}{node.name} : CPU {format_time(node.cpu_duration)}, "
                     f"GPU {format_time(node.gpu_duration)}, "
                     f"Mem {format_memory(node.memory_delta)}{hint}")
        if indent < max_depth:
            lines.append(get_flame_like_string(node.children, max_depth, indent+1))
    return "\n".join(lines)

def generate_report(roots: List[OperationWrapper], output_file: Optional[str] = None):
    report_parts = []
    report_parts.append("PyTorch Trace Analysis Report (Full Stream Dependency Walk)")
    report_parts.append("==========================================================")
    report_parts.append(get_top_k_string(roots))
    report_parts.append(get_flame_like_string(roots))

    agg_phases = extract_fsdp_phases_aggregated(roots)
    report_parts.append(get_fsdp_timeline_aggregated_string(agg_phases))

    report_parts.append(get_fsdp_chronological_timeline(roots))

    full_report = "\n".join(report_parts)
    print(full_report)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_report)


# Benchmarking and comparing
def compare_traces_to_csv(trace1_file, trace2_file, output_csv, op_filter=None):
    print(f"Processing trace 1: {trace1_file}")
    p1 = TraceParser(trace1_file)
    p1.load()
    roots1 = p1.build_tree()
    p1.attribute_gpu_time_with_dependencies(roots1)
    p1.attribute_memory(roots1)
    metrics1 = p1.get_aggregated_metrics(roots1, op_filter)

    print(f"Processing trace 2: {trace2_file}")
    p2 = TraceParser(trace2_file)
    p2.load()
    roots2 = p2.build_tree()
    p2.attribute_gpu_time_with_dependencies(roots2)
    p2.attribute_memory(roots2)
    metrics2 = p2.get_aggregated_metrics(roots2, op_filter)

    all_ops = set(metrics1.keys()) | set(metrics2.keys())
    rows = []
    for op in sorted(all_ops):
        m1 = metrics1.get(op, {})
        m2 = metrics2.get(op, {})
        row = {
            'operation_name': op,
            'count_trace1': m1.get('count', 0),
            'count_trace2': m2.get('count', 0),
            'total_cpu_us_trace1': m1.get('total_cpu_us', 0.0),
            'total_cpu_us_trace2': m2.get('total_cpu_us', 0.0),
            'avg_cpu_us_trace1': m1.get('avg_cpu_us', 0.0),
            'avg_cpu_us_trace2': m2.get('avg_cpu_us', 0.0),
            'total_gpu_us_trace1': m1.get('total_gpu_us', 0.0),
            'total_gpu_us_trace2': m2.get('total_gpu_us', 0.0),
            'avg_gpu_us_trace1': m1.get('avg_gpu_us', 0.0),
            'avg_gpu_us_trace2': m2.get('avg_gpu_us', 0.0),
            'total_mem_bytes_trace1': m1.get('total_mem_bytes', 0),
            'total_mem_bytes_trace2': m2.get('total_mem_bytes', 0),
            'avg_mem_bytes_trace1': m1.get('avg_mem_bytes', 0.0),
            'avg_mem_bytes_trace2': m2.get('avg_mem_bytes', 0.0),
        }
        gpu1 = row['total_gpu_us_trace1']
        gpu2 = row['total_gpu_us_trace2']
        row['gpu_speedup_trace2_vs_trace1'] = gpu2 / gpu1 if gpu1 > 0 else float('inf')
        rows.append(row)

    fieldnames = [
        'operation_name', 'count_trace1', 'count_trace2',
        'total_cpu_us_trace1', 'total_cpu_us_trace2',
        'avg_cpu_us_trace1', 'avg_cpu_us_trace2',
        'total_gpu_us_trace1', 'total_gpu_us_trace2',
        'avg_gpu_us_trace1', 'avg_gpu_us_trace2',
        'total_mem_bytes_trace1', 'total_mem_bytes_trace2',
        'avg_mem_bytes_trace1', 'avg_mem_bytes_trace2',
        'gpu_speedup_trace2_vs_trace1'
    ]
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Comparison CSV saved to {output_csv}")


# ------------------------------------------------------------------------
#  Main
count = 0 
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single trace analysis: python final_trace_analyzer.py <trace.json> [--output report.txt]")
        print("  Benchmark (CSV comparison): python final_trace_analyzer.py --compare trace1.json trace2.json --output comparison.csv [--op OPERATION_NAME]")
        sys.exit(1)

    if sys.argv[1] == "--compare":
        if len(sys.argv) < 4:
            print("Error: --compare requires two trace files.")
            sys.exit(1)
        trace1 = sys.argv[2]
        trace2 = sys.argv[3]
        output_csv = None
        op_filter = None
        for i, arg in enumerate(sys.argv):
            if arg == "--output" and i+1 < len(sys.argv):
                output_csv = sys.argv[i+1]
            if arg == "--op" and i+1 < len(sys.argv):
                op_filter = sys.argv[i+1]
        if not output_csv:
            print("Error: Please specify --output comparison.csv")
            sys.exit(1)
        compare_traces_to_csv(trace1, trace2, output_csv, op_filter)
        sys.exit(0)

    trace_file = sys.argv[1]
    output_file = None
    if len(sys.argv) >= 3 and sys.argv[2] == "--output":
        output_file = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Loading trace from {trace_file}...")
    parser = TraceParser(trace_file)
    parser.load()
    print(f"Loaded {len(parser.cpu_events)} CPU events, {len(parser.gpu_events)} GPU events, {len(parser.memory_events)} memory events.")

    print("Building logical operation tree...")
    roots = parser.build_tree()

    print("Performing full CUDA stream dependency walk...")
    parser.attribute_gpu_time_with_dependencies(roots)

    print("Attributing memory deltas...")
    parser.attribute_memory(roots)

    print("Generating report...")
    generate_report(roots, output_file)
    print("Done.")
    global count
    print(count)

if __name__ == "__main__":
    main()