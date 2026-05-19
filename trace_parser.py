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



# Helper function: iterate through the tree of logical operations
def iter_nodes(roots: List['LogicalOperation']) -> Iterator['LogicalOperation']:
    """Iterate over all nodes in the tree (depth-first, iterative)."""
    stack = list(roots)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


# Data structure for a logical operation / node in the call tree
@dataclass
class LogicalOperation:
    name: str
    start_time: float
    end_time: float
    cpu_duration: float
    gpu_duration: float = 0.0
    children: List['LogicalOperation'] = field(default_factory=list)

    @property
    def total_time(self) -> float:
        return max(self.cpu_duration, self.gpu_duration)

    @property
    def exclusive_cpu(self) -> float:
        return self.cpu_duration - sum(c.cpu_duration for c in self.children)

    @property
    def exclusive_gpu(self) -> float:
        return self.gpu_duration - sum(c.gpu_duration for c in self.children)


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

    def build_tree(self) -> List[LogicalOperation]:
        all_roots = []
        for (pid, tid), events in self.events_by_pid_tid.items():
            events_sorted = sorted(events, key=lambda e: e['ts'])
            stack = []
            for ev in events_sorted:
                start = ev['ts']
                dur = ev.get('dur', 0)
                node = LogicalOperation(
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

    def attribute_gpu_time_with_dependencies(self, roots: List[LogicalOperation]):
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

    def attribute_memory(self, roots: List[LogicalOperation]):
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

    def get_aggregated_metrics(self, roots: List[LogicalOperation], op_filter: Optional[str] = None) -> Dict[str, Dict]:
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


# FSDP specific analysis
def extract_fsdp_phases_aggregated(roots: List[LogicalOperation]) -> Dict[str, Dict[str, float]]:
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


# TODO: bottleneck-detector
# TODO: reporting
# TODO: benchmarking

#  Main
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single trace analysis: python final_trace_analyzer.py <trace.json> [--output report.txt]")
        print("  Benchmark (CSV comparison): python final_trace_analyzer.py --compare trace1.json trace2.json --output comparison.csv [--op OPERATION_NAME]")
        sys.exit(1)

# TODO: implement --compare

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
    # TODO: implement output report

    print("Done.")

if __name__ == "__main__":
    main()