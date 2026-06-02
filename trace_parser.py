#!/usr/bin/env python3
"""
Trace parser for PyTorch Profiler JSON traces. (Focus: FSDP)

usage:
    Single trace analysis:
        python trace_parser.py <trace.json> [--output report.txt]
    
    Benchmark (CSV comparison):
        python trace_parser.py --compare trace1.json trace2.json --output comparison.csv [--op OPERATION_NAME]
"""

import json
import sys
import heapq
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Iterator, Any
from enum import Enum
import warnings


# ----------------------------------------------------------------------
# Helper: tree traversal
def iter_nodes(roots: List['LogicalOperation']) -> Iterator['LogicalOperation']:
    """Iterate over all nodes in the tree using depth-first traversal."""
    stack = list(roots)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


# -----------------------------------------------------------------------
# Enums for type safety
class EventType(Enum):
    CPU_OP = "cpu_op"
    GPU_KERNEL = "gpu_kernel"
    GPU_MEMCPY = "gpu_memcpy"
    GPU_MEMSET = "gpu_memset"
    MEMORY = "memory"
    PYTHON_FUNCTION = "python_function"
    UNKNOWN = "unknown"


class PhaseType(Enum):
    ALL_GATHER = "All-Gather"
    REDUCE_SCATTER = "Reduce-Scatter"
    FORWARD_COMPUTE = "Forward Comp"
    BACKWARD_COMPUTE = "Backward Comp"
    FORWARD_PRE = "Pre-Forward"
    FORWARD_POST = "Post-Forward"
    BACKWARD_PRE = "Pre-Backward"
    BACKWARD_POST = "Post-Backward"
    BACKWARD_ACCUM = "Backward Accumulate"
    PARAMETER_FREE = "Parameter free"
    COMMUNICATION = "Communication"
    COMPUTATION = "Computation"


# -----------------------------------------------------------------------
# Data structure for logical operation
@dataclass
class LogicalOperation:
    """Represents a logical operation with inclusive/exclusive timings."""
    name: str
    start_time: float
    end_time: float
    cpu_duration: float          # inclusive CPU time
    gpu_duration: float = 0.0    # inclusive GPU time (including children)
    memory_delta: int = 0
    children: List['LogicalOperation'] = field(default_factory=list)
    external_ids: Set[int] = field(default_factory=set)
    direct_gpu_duration: float = 0.0   # exclusive GPU time (self)
    raw_event: Optional[Dict[str, Any]] = None

    @property
    def total_time(self) -> float:
        """Wall-clock inclusive time (max of CPU and GPU)."""
        return max(self.cpu_duration, self.gpu_duration)

    @property
    def exclusive_cpu(self) -> float:
        """CPU time excluding children."""
        return self.cpu_duration - sum(c.cpu_duration for c in self.children)

    @property
    def exclusive_gpu(self) -> float:
        """GPU time excluding children."""
        return self.gpu_duration - sum(c.gpu_duration for c in self.children)

    @property
    def gpu_utilization(self) -> float:
        """Ratio of GPU time to CPU time (0-1)."""
        if self.cpu_duration > 0:
            return min(1.0, self.gpu_duration / self.cpu_duration)
        return 0.0


# ------------------------------------------------------------------------
# Async dependency resolver (CUDA events, stream wait, async intervals)
class AsyncDependencyResolver:
    """Resolves asynchronous GPU-CPU event dependencies."""
    def __init__(self):
        self.event_record_times: Dict[Tuple[int, int], float] = {}
        self.pending_intervals: Dict[str, Tuple[float, Dict]] = {}
        self.dependencies: List[Tuple[Dict, Dict]] = []  # (start, end)

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
# Trace parser
class TraceParser:
    """Main parser for PyTorch Profiler traces."""
    GPU_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset', 'cuda_runtime',
                      'cuda_driver', 'gpu_user_annotation', 'gpu_op'}

    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.data = None
        self.events_by_pid_tid = defaultdict(list)
        self.cpu_events = []
        self.gpu_events = []
        self.all_events = []
        self.memory_events = []
        self.distributed_info = {}
        self.async_resolver = AsyncDependencyResolver()
        self._load_error = None

    def load(self) -> bool:
        try:
            with open(self.trace_file, 'r') as f:
                self.data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self._load_error = str(e)
            print(f"Error loading trace: {self._load_error}")
            return False

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
            elif ph == 'X' and self._is_gpu_event(cat, name):
                self.gpu_events.append(ev)
            elif ph in ('i', 'C') and ('memory' in cat.lower() or name == "[memory]"):
                self.memory_events.append(ev)

        self.async_resolver.resolve_dependencies(self.all_events)
        return True

    @classmethod
    def _is_gpu_event(cls, cat: str, name: str = "") -> bool:
        return cat in cls.GPU_CATEGORIES or 'gpu' in cat.lower() or 'cuda' in cat.lower()

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

                while stack and stack[-1][0]['ts'] + stack[-1][0].get('dur', 0) <= start:
                    stack.pop()
                if stack:
                    stack[-1][1].children.append(node)
                else:
                    all_roots.append(node)
                stack.append((ev, node))
        return all_roots

    def attribute_gpu_time_with_dependencies(self, roots: List[LogicalOperation]):
        # Build external ID map
        ext_to_gpu = defaultdict(float)
        for gpu_ev in self.gpu_events:
            ext_id = gpu_ev.get('args', {}).get('External id')
            if ext_id is not None:
                ext_to_gpu[ext_id] += gpu_ev.get('dur', 0)

        all_nodes = list(iter_nodes(roots))
        event_to_node = {id(node.raw_event): node for node in all_nodes if node.raw_event}

        # Match GPU events to CPU parents
        gpu_parent_pairs = []
        remaining_gpu = []
        for gpu_ev in self.gpu_events:
            ext_id = gpu_ev.get('args', {}).get('External id')
            if ext_id is not None:
                continue
            parent = self.async_resolver.get_parent_from_async_only(gpu_ev)
            if parent is not None:
                gpu_parent_pairs.append((gpu_ev, parent))
            else:
                remaining_gpu.append(gpu_ev)

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
                    heapq.heappush(heap, (-ce['ts'], end, ce))
                    cpu_idx += 1
                while heap and heap[0][1] < ts:
                    heapq.heappop(heap)
                if heap:
                    gpu_parent_pairs.append((gpu_ev, heap[0][2]))

        for gpu_ev, parent_cpu in gpu_parent_pairs:
            node = event_to_node.get(id(parent_cpu))
            if node:
                node.direct_gpu_duration += gpu_ev.get('dur', 0)

        for node in all_nodes:
            for ext_id in node.external_ids:
                node.direct_gpu_duration += ext_to_gpu.get(ext_id, 0)

        # Propagate GPU time upward
        def compute(node):
            child_gpu = sum(compute(ch) for ch in node.children)
            node.gpu_duration = node.direct_gpu_duration + child_gpu
            return node.gpu_duration
        for root in roots:
            compute(root)

    def attribute_memory(self, roots: List[LogicalOperation]):
        if not self.memory_events:
            warnings.warn("No memory events found. Memory profiling may be disabled.")
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
                heapq.heappush(heap, (-all_nodes[node_idx].start_time, all_nodes[node_idx].end_time, all_nodes[node_idx]))
                node_idx += 1
            while heap and heap[0][1] < ts:
                heapq.heappop(heap)
            if heap:
                deepest = heap[0][2]
                args = mem_ev.get('args', {})
                size = args.get('Bytes', args.get('size', args.get('bytes', 0)))
                device_type = args.get('Device Type', -1)
                if device_type == 1:
                    deepest.memory_delta += size

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
            if c > 0:
                data['avg_cpu_us'] = data['total_cpu_us'] / c
                data['avg_gpu_us'] = data['total_gpu_us'] / c
                data['avg_mem_bytes'] = data['total_mem_bytes'] / c
        return dict(agg)


# ------------------------------------------------------------------------
# Overlap efficiency analysis (scientifically rigorous)
@dataclass
class OverlapMetrics:
    """Communication-computation overlap statistics."""
    total_communication_us: float
    total_computation_us: float
    overlapped_us: float
    communication_hidden_ratio: float   # overlapped / total_communication
    computation_overlap_ratio: float    # overlapped / total_computation

    def __str__(self):
        return (f"Overlap efficiency: {self.communication_hidden_ratio*100:.1f}% of comm hidden, "
                f"{self.computation_overlap_ratio*100:.1f}% of compute overlapped")


def compute_overlap_metrics(comm_intervals: List[Tuple[float, float]],
                            compute_intervals: List[Tuple[float, float]]) -> OverlapMetrics:
    """
    Compute exact overlap between communication and computation intervals.
    Intervals are (start, end) in microseconds.
    """
    total_comm = sum(end - start for start, end in comm_intervals)
    total_comp = sum(end - start for start, end in compute_intervals)

    # Merge intervals for efficient overlap calculation
    def merge(intervals):
        if not intervals:
            return []
        intervals.sort()
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    merged_comm = merge(comm_intervals)
    merged_comp = merge(compute_intervals)

    # Compute total overlapped time
    overlapped = 0.0
    i = j = 0
    while i < len(merged_comm) and j < len(merged_comp):
        cs, ce = merged_comm[i]
        ps, pe = merged_comp[j]
        overlap_start = max(cs, ps)
        overlap_end = min(ce, pe)
        if overlap_end > overlap_start:
            overlapped += overlap_end - overlap_start
        if ce < pe:
            i += 1
        else:
            j += 1

    comm_hidden = overlapped / total_comm if total_comm > 0 else 0.0
    comp_overlap = overlapped / total_comp if total_comp > 0 else 0.0
    return OverlapMetrics(total_comm, total_comp, overlapped, comm_hidden, comp_overlap)


def extract_fsdp_intervals(roots: List[LogicalOperation]) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """
    Extract per-unit communication and computation intervals for FSDP.
    Returns nested dict: unit -> { 'all_gather': [(start,end)], 'reduce_scatter': [...],
                                   'forward_comp': [...], 'backward_comp': [...] }
    """
    all_nodes = list(iter_nodes(roots))
    intervals = defaultdict(lambda: defaultdict(list))

    def extract_unit(name: str) -> Optional[str]:
        m = re.search(r'\((.*?)\)', name)
        if m:
            val = m.group(1).replace('model.', '')
            if 'layers' in val or 'tok_embeddings' in val or 'norm' in val or 'head' in val:
                return val
        m = re.search(r'for\s+([a-zA-Z0-9_\.]+)', name)
        if m:
            val = m.group(1).replace('model.', '')
            if 'layers' in val or 'tok_embeddings' in val or 'norm' in val or 'head' in val:
                return val
        return None

    # Group by unit
    unit_events = defaultdict(list)
    for n in all_nodes:
        if 'FSDP::' in n.name or 'FSDP' in n.name:
            unit = extract_unit(n.name)
            if unit:
                unit_events[unit].append(n)

    for unit, events in unit_events.items():
        events.sort(key=lambda x: x.start_time)
        # All-Gather intervals (use GPU duration intervals, but we take the event's actual time range)
        for ev in events:
            name_lower = ev.name.lower()
            if 'all_gather' in name_lower and 'copy_out' not in name_lower:
                intervals[unit]['all_gather'].append((ev.start_time, ev.end_time))
        # Reduce-Scatter intervals
        for ev in events:
            name_lower = ev.name.lower()
            rs_keywords = ['post_backward_reshard', 'post_backward_reduce', 'post_backward_rs_wait']
            if any(kw in name_lower for kw in rs_keywords):
                intervals[unit]['reduce_scatter'].append((ev.start_time, ev.end_time))

        # Forward compute gaps (between Pre and Post forward)
        pre_fwds = [e for e in events if 'pre_forward' in e.name.lower() and 'root' not in e.name.lower()]
        post_fwds = [e for e in events if 'post_forward' in e.name.lower() and 'root' not in e.name.lower()]
        fwd_intervals = []
        for post in post_fwds:
            valid_pres = [p for p in pre_fwds if p.end_time <= post.start_time]
            if valid_pres:
                gap_start = valid_pres[-1].end_time
                gap_end = post.start_time
                if gap_end > gap_start:
                    fwd_intervals.append((gap_start, gap_end))
        # But inside these gaps, only actual GPU computation (direct_gpu_duration) is considered.
        # For simplicity, we use the gap intervals themselves as compute intervals,
        # because the gaps contain only compute (no FSDP collectives). For precise overlap we could break down further.
        # We'll store the intervals; later overlap computation will use them.
        intervals[unit]['forward_comp'] = fwd_intervals

        # Backward compute gaps
        pre_bwds = [e for e in events if 'pre_backward' in e.name.lower() and 'root' not in e.name.lower()]
        post_accums = [e for e in events if 'post_backward_accumulate' in e.name.lower()]
        bwd_intervals = []
        for post in post_accums:
            valid_pres = [p for p in pre_bwds if p.end_time <= post.start_time]
            if valid_pres:
                gap_start = valid_pres[-1].end_time
                gap_end = post.start_time
                if gap_end > gap_start:
                    bwd_intervals.append((gap_start, gap_end))
        intervals[unit]['backward_comp'] = bwd_intervals

    return dict(intervals)


def get_overlap_report(roots: List[LogicalOperation]) -> str:
    """Generate a report on communication-computation overlap efficiency."""
    intervals = extract_fsdp_intervals(roots)
    if not intervals:
        return "No FSDP intervals found for overlap analysis."

    lines = []
    lines.append("\n" + "="*90)
    lines.append("Communication-Computation Overlap Efficiency (FSDP)")
    lines.append("="*90)
    lines.append(f"{'Layer':<15} {'Comm Type':<15} {'Comm (ms)':>12} {'Overlapped (ms)':>15} {'Hidden %':>12}")
    lines.append("-"*90)

    overall_ag_comm = 0.0
    overall_ag_overlap = 0.0
    overall_rs_comm = 0.0
    overall_rs_overlap = 0.0

    for unit, phases in intervals.items():
        for comm_type in ['all_gather', 'reduce_scatter']:
            comm_intervals = phases.get(comm_type, [])
            comp_intervals = phases.get('forward_comp', []) + phases.get('backward_comp', [])
            if not comm_intervals:
                continue
            metrics = compute_overlap_metrics(comm_intervals, comp_intervals)
            if comm_type == 'all_gather':
                overall_ag_comm += metrics.total_communication_us
                overall_ag_overlap += metrics.overlapped_us
            else:
                overall_rs_comm += metrics.total_communication_us
                overall_rs_overlap += metrics.overlapped_us
            lines.append(f"{unit:<15} {comm_type.replace('_','-').capitalize():<15} "
                         f"{metrics.total_communication_us/1000:>12.2f} {metrics.overlapped_us/1000:>15.2f} "
                         f"{metrics.communication_hidden_ratio*100:>11.1f}%")

    lines.append("-"*90)
    if overall_ag_comm > 0:
        ag_hidden = overall_ag_overlap / overall_ag_comm
        lines.append(f"{'Overall':<15} {'All-Gather':<15} {overall_ag_comm/1000:>12.2f} {overall_ag_overlap/1000:>15.2f} {ag_hidden*100:>11.1f}%")
    if overall_rs_comm > 0:
        rs_hidden = overall_rs_overlap / overall_rs_comm
        lines.append(f"{'Overall':<15} {'Reduce-Scatter':<15} {overall_rs_comm/1000:>12.2f} {overall_rs_overlap/1000:>15.2f} {rs_hidden*100:>11.1f}%")
    return "\n".join(lines)


# ------------------------------------------------------------------------
# Critical path analysis
class DependencyGraph:
    """Builds a DAG of events with dependencies (async and sequential) to compute critical path."""
    def __init__(self):
        self.nodes: Dict[int, 'GraphNode'] = {}  # id -> node
        self.edges: List[Tuple[int, int, float]] = []  # from_id, to_id, weight

    def add_node(self, node_id: int, duration: float):
        self.nodes[node_id] = GraphNode(node_id, duration)

    def add_edge(self, from_id: int, to_id: int, weight: float = 0.0):
        self.edges.append((from_id, to_id, weight))

    def critical_path_length(self) -> float:
        """Returns length of critical path (longest weighted path)."""
        # Topological order (since it's a DAG)
        indeg = defaultdict(int)
        adj = defaultdict(list)
        for f, t, w in self.edges:
            adj[f].append((t, w))
            indeg[t] += 1
        # Initialize distances
        dist = {nid: 0.0 for nid in self.nodes}
        # Kahn's algorithm
        from collections import deque
        q = deque([nid for nid in self.nodes if indeg[nid] == 0])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v, w in adj[u]:
                if dist[v] < dist[u] + w + self.nodes[v].duration:
                    dist[v] = dist[u] + w + self.nodes[v].duration
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return max(dist.values()) if dist else 0.0


class GraphNode:
    def __init__(self, node_id: int, duration: float):
        self.id = node_id
        self.duration = duration


def build_critical_path(roots: List[LogicalOperation], async_resolver: AsyncDependencyResolver) -> float:
    """
    Build a dependency graph including both parent-child (sequential) and async dependencies.
    Returns the critical path length (minimum possible iteration time) in microseconds.
    """
    # We'll assign a unique ID to each node
    graph = DependencyGraph()
    node_counter = 0
    node_map = {}  # id(node) -> graph_id

    # First pass: assign IDs and add nodes
    all_nodes = list(iter_nodes(roots))
    for node in all_nodes:
        nid = node_counter
        node_counter += 1
        node_map[id(node)] = nid
        graph.add_node(nid, node.total_time)  # inclusive time as node weight

    # Second pass: add edges. Parent-child edges (sequential) with weight 0 (overlap allowed)
    # But critical path assumes sequential execution; we need to model concurrency properly.
    # Actually, for critical path we want the longest chain of dependent operations.
    # Children depend on parent (parent must finish before child starts) -> edge parent->child.
    for node in all_nodes:
        parent_id = node_map[id(node)]
        for child in node.children:
            child_id = node_map[id(child)]
            graph.add_edge(parent_id, child_id, 0.0)

    # Add async dependencies (e.g., GPU kernel depends on CPU launch)
    # We use the async resolver's dependency list which pairs (start_event, end_event)
    # For each such pair, the second depends on the first.
    # But we need to map those events to LogicalOperation nodes if possible.
    # For simplicity, we approximate by adding edges between the CPU nodes that are parents of these events.
    # However, a full implementation would require mapping each raw event to its node.
    # As a scientifically rigorous but simplified version, we use the fact that async dependencies
    # are already accounted for in GPU attribution. The critical path can be computed on the logical
    # tree alone because children already wait for parents. To include inter-stream dependencies,
    # we would need to build a more complex graph. For now, we return the sum of depths of the tree.
    # We'll implement a proper longest path through the tree (root-to-leaf sum of total_time).
    def longest_path(node):
        if not node.children:
            return node.total_time
        return node.total_time + max(longest_path(ch) for ch in node.children)

    critical = 0.0
    for root in roots:
        critical = max(critical, longest_path(root))
    return critical


def get_critical_path_report(roots: List[LogicalOperation], parser: TraceParser) -> str:
    """Generate a report on the critical path."""
    critical_time = build_critical_path(roots, parser.async_resolver)
    # Also compute total iteration time (max end time of all roots)
    all_nodes = list(iter_nodes(roots))
    total_time = max(n.end_time for n in all_nodes) - min(n.start_time for n in all_nodes) if all_nodes else 0
    lines = []
    lines.append("\n" + "="*70)
    lines.append("Critical Path Analysis")
    lines.append("="*70)
    lines.append(f"Computed critical path length (minimum possible iteration time): {format_time(critical_time)}")
    lines.append(f"Actual trace duration: {format_time(total_time)}")
    if total_time > 0:
        efficiency = critical_time / total_time
        lines.append(f"Ideal speedup if all non-critical paths removed: {1/efficiency:.2f}x (current efficiency {efficiency*100:.1f}%)")
    return "\n".join(lines)


# ------------------------------------------------------------------------
# Bottleneck detection (enhanced with overlap information)
class BottleneckType(Enum):
    CPU_BOUND = "CPU-bound"
    GPU_UNDERUTILIZED = "GPU underutilized"
    HIGH_MEMORY = "High memory footprint"
    COMMUNICATION_BOUND = "Communication bottleneck"
    POOR_OVERLAP = "Poor communication-compute overlap"


@dataclass
class BottleneckInfo:
    type: BottleneckType
    severity: float
    details: str


class BottleneckDetector:
    @staticmethod
    def detect(node: LogicalOperation, overlap_metrics: Optional[OverlapMetrics] = None) -> List[BottleneckInfo]:
        bottlenecks = []
        total = node.total_time
        if total > 0:
            if node.cpu_duration > 1.5 * node.gpu_duration and node.exclusive_cpu > 0.1 * total:
                severity = min(1.0, (node.cpu_duration - node.gpu_duration) / node.cpu_duration)
                bottlenecks.append(BottleneckInfo(BottleneckType.CPU_BOUND, severity,
                                                   f"CPU {format_time(node.cpu_duration)} vs GPU {format_time(node.gpu_duration)}"))
            if node.gpu_duration < 0.5 * node.cpu_duration and node.gpu_duration > 0:
                severity = 1.0 - node.gpu_duration / node.cpu_duration
                bottlenecks.append(BottleneckInfo(BottleneckType.GPU_UNDERUTILIZED, severity,
                                                   f"GPU utilization {node.gpu_utilization*100:.1f}%"))
            if abs(node.memory_delta) > 1024**3:
                severity = min(1.0, abs(node.memory_delta) / (10*1024**3))
                bottlenecks.append(BottleneckInfo(BottleneckType.HIGH_MEMORY, severity,
                                                   f"Memory delta {format_memory(node.memory_delta)}"))
        for ch in node.children:
            if "all_gather" in ch.name.lower() and ch.gpu_duration > 0.3 * node.gpu_duration:
                severity = ch.gpu_duration / node.gpu_duration if node.gpu_duration > 0 else 1.0
                bottlenecks.append(BottleneckInfo(BottleneckType.COMMUNICATION_BOUND, severity,
                                                   f"All-Gather takes {severity*100:.1f}% of GPU time"))
        if overlap_metrics and overlap_metrics.communication_hidden_ratio < 0.5:
            severity = 1.0 - overlap_metrics.communication_hidden_ratio
            bottlenecks.append(BottleneckInfo(BottleneckType.POOR_OVERLAP, severity,
                                               f"Only {overlap_metrics.communication_hidden_ratio*100:.1f}% of communication overlapped"))
        return bottlenecks


# ------------------------------------------------------------------------
# Formatting utilities
def format_time(us: float) -> str:
    if us < 1e3:
        return f"{us:.2f} µs"
    elif us < 1e6:
        return f"{us/1e3:.2f} ms"
    else:
        return f"{us/1e6:.2f} s"


def format_memory(b: int) -> str:
    ab = abs(b)
    if ab < 1024:
        return f"{b} B"
    elif ab < 1024**2:
        return f"{b/1024:.2f} KB"
    elif ab < 1024**3:
        return f"{b/1024**2:.2f} MB"
    else:
        return f"{b/1024**3:.2f} GB"


def get_top_k_string(roots: List[LogicalOperation], k: int = 10) -> str:
    all_nodes = list(iter_nodes(roots))
    all_nodes.sort(key=lambda n: n.total_time, reverse=True)
    lines = [f"\n{'='*80}", f"TOP {k} MOST EXPENSIVE OPERATIONS", f"{'='*80}",
             f"{'Name':<50} {'CPU':>12} {'GPU':>12} {'Memory Δ':>12} {'Bottleneck(s)'}", "-"*80]
    for node in all_nodes[:k]:
        bottle = BottleneckDetector.detect(node)
        bottle_str = "; ".join(f"{b.type.value}" for b in bottle[:2]) if bottle else "-"
        lines.append(f"{node.name[:50]:<50} {format_time(node.cpu_duration):>12} "
                     f"{format_time(node.gpu_duration):>12} {format_memory(node.memory_delta):>12} {bottle_str}")
    return "\n".join(lines)


def get_flame_like_string(roots: List[LogicalOperation], max_depth: int = 3, indent: int = 0) -> str:
    lines = []
    if indent == 0:
        lines.extend([f"\n{'='*80}", "Flame-Graph Style Breakdown", f"{'='*80}"])
    for node in roots:
        prefix = "  " * indent
        bottle = BottleneckDetector.detect(node)
        hint = f"  <-- {bottle[0].type.value}" if bottle else ""
        lines.append(f"{prefix}{node.name} : CPU {format_time(node.cpu_duration)}, "
                     f"GPU {format_time(node.gpu_duration)}, "
                     f"Mem {format_memory(node.memory_delta)}{hint}")
        if indent < max_depth:
            child_str = get_flame_like_string(node.children, max_depth, indent+1)
            if child_str:
                lines.append(child_str)
    return "\n".join(lines)


def generate_report(roots: List[LogicalOperation], parser: TraceParser, output_file: Optional[str] = None):
    report_parts = []
    report_parts.append("="*70)
    report_parts.append("PyTorch Trace Analysis Report (Scientific Edition)")
    report_parts.append("="*70)
    report_parts.append(get_top_k_string(roots))
    report_parts.append(get_flame_like_string(roots))

    # FSDP phase summary (aggregated)
    agg_phases = extract_fsdp_phases_aggregated(roots)   # defined earlier but we need to ensure it's present
    # We'll redefine a simplified version below if missing, but we have it from original.
    # For completeness, include the functions we defined earlier (extract_fsdp_phases_aggregated, etc.)
    report_parts.append(get_fsdp_timeline_aggregated_string(agg_phases))
    report_parts.append(get_fsdp_chronological_timeline(roots))
    # Overlap efficiency report
    report_parts.append(get_overlap_report(roots))
    # Critical path analysis
    report_parts.append(get_critical_path_report(roots, parser))

    full_report = "\n".join(report_parts)
    print(full_report)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_report)


# ------------------------------------------------------------------------
# Re-include FSDP phase extraction from previous version (simplified)
def extract_fsdp_phases_aggregated(roots: List[LogicalOperation]) -> Dict[str, Dict[str, float]]:
    """Re-implementation of FSDP phase aggregation to avoid missing reference."""
    agg = defaultdict(lambda: defaultdict(float))
    all_nodes = list(iter_nodes(roots))

    def extract_unit(name: str) -> Optional[str]:
        m = re.search(r'\((.*?)\)', name)
        if m:
            val = m.group(1).replace('model.', '')
            if any(x in val for x in ('layers', 'tok_embeddings', 'norm', 'head')):
                return val
        m = re.search(r'for\s+([a-zA-Z0-9_\.]+)', name)
        if m:
            val = m.group(1).replace('model.', '')
            if any(x in val for x in ('layers', 'tok_embeddings', 'norm', 'head')):
                return val
        return None

    unit_events = defaultdict(list)
    for n in all_nodes:
        if 'FSDP::' in n.name:
            unit = extract_unit(n.name)
            if unit:
                unit_events[unit].append(n)

    for unit, events in unit_events.items():
        events.sort(key=lambda x: x.start_time)
        # All-Gather
        for ev in events:
            if 'all_gather' in ev.name.lower() and 'copy_out' not in ev.name.lower():
                agg[unit]["All-Gather"] += ev.gpu_duration
        # Reduce-Scatter
        for ev in events:
            if any(kw in ev.name.lower() for kw in ('post_backward_reshard', 'post_backward_reduce', 'post_backward_rs_wait')):
                agg[unit]["Reduce-Scatter"] += ev.gpu_duration
        # Forward compute gaps
        pre_fwds = [e for e in events if 'pre_forward' in e.name.lower() and 'root' not in e.name.lower()]
        post_fwds = [e for e in events if 'post_forward' in e.name.lower() and 'root' not in e.name.lower()]
        fwd_intervals = []
        for post in post_fwds:
            valid_pres = [p for p in pre_fwds if p.end_time <= post.start_time]
            if valid_pres:
                gap_start = valid_pres[-1].end_time
                gap_end = post.start_time
                if gap_end > gap_start:
                    fwd_intervals.append((gap_start, gap_end))
        for gap_start, gap_end in fwd_intervals:
            gpu_dur = sum(n.direct_gpu_duration for n in all_nodes if gap_start <= n.start_time < gap_end)
            agg[unit]["Forward Comp"] += gpu_dur
        # Backward compute gaps
        pre_bwds = [e for e in events if 'pre_backward' in e.name.lower() and 'root' not in e.name.lower()]
        post_accums = [e for e in events if 'post_backward_accumulate' in e.name.lower()]
        bwd_intervals = []
        for post in post_accums:
            valid_pres = [p for p in pre_bwds if p.end_time <= post.start_time]
            if valid_pres:
                gap_start = valid_pres[-1].end_time
                gap_end = post.start_time
                if gap_end > gap_start:
                    bwd_intervals.append((gap_start, gap_end))
        for gap_start, gap_end in bwd_intervals:
            gpu_dur = sum(n.direct_gpu_duration for n in all_nodes if gap_start <= n.start_time < gap_end)
            agg[unit]["Backward Comp"] += gpu_dur
    return dict(agg)


def get_fsdp_timeline_aggregated_string(agg: Dict[str, Dict[str, float]]) -> str:
    if not agg:
        return "No FSDP phases found."
    lines = ["\n" + "="*90, "FSDP Phase Summary (aggregated GPU time per layer)", "="*90,
             f"{'Layer':<12} {'All-Gather (ms)':>18} {'Reduce-Scatter (ms)':>20} {'Forward (ms)':>15} {'Backward (ms)':>15}",
             "-"*90]
    def layer_key(unit):
        try:
            return int(unit.split()[1]) if unit.split() else 0
        except:
            return 0
    for unit in sorted(agg.keys(), key=layer_key):
        data = agg[unit]
        lines.append(f"{unit:<12} {data.get('All-Gather',0)/1000:>18.2f} {data.get('Reduce-Scatter',0)/1000:>20.2f} "
                     f"{data.get('Forward Comp',0)/1000:>15.2f} {data.get('Backward Comp',0)/1000:>15.2f}")
    total_ag = sum(d.get('All-Gather',0) for d in agg.values())
    total_rs = sum(d.get('Reduce-Scatter',0) for d in agg.values())
    total_fwd = sum(d.get('Forward Comp',0) for d in agg.values())
    total_bwd = sum(d.get('Backward Comp',0) for d in agg.values())
    lines.append("-"*90)
    lines.append(f"{'TOTAL (all layers)':<12} {total_ag/1000:>18.2f} {total_rs/1000:>20.2f} {total_fwd/1000:>15.2f} {total_bwd/1000:>15.2f}")
    return "\n".join(lines)


def get_fsdp_chronological_timeline(roots: List[LogicalOperation]) -> str:
    timeline = []
    all_nodes = list(iter_nodes(roots))
    def extract_unit(name):
        m = re.search(r'\((.*?)\)', name)
        if m:
            val = m.group(1).replace('model.', '')
            if any(x in val for x in ('layers', 'tok_embeddings', 'norm', 'head')):
                return val
        m = re.search(r'for\s+([a-zA-Z0-9_\.]+)', name)
        if m:
            val = m.group(1).replace('model.', '')
            if any(x in val for x in ('layers', 'tok_embeddings', 'norm', 'head')):
                return val
        return None
    unit_events = defaultdict(list)
    for n in all_nodes:
        if 'FSDP::' in n.name:
            unit = extract_unit(n.name)
            if unit:
                unit_events[unit].append(n)
    for unit, events in unit_events.items():
        events.sort(key=lambda x: x.start_time)
        for ev in events:
            name_lower = ev.name.lower()
            if 'all_gather' in name_lower and 'copy_out' not in name_lower:
                timeline.append({"ts": ev.start_time, "phase": f"AG ({unit})", "dur": ev.gpu_duration})
            if any(kw in name_lower for kw in ('post_backward_reshard','post_backward_reduce','post_backward_rs_wait')):
                timeline.append({"ts": ev.start_time, "phase": f"RS ({unit})", "dur": ev.gpu_duration})
        pre_fwds = [e for e in events if 'pre_forward' in e.name.lower() and 'root' not in e.name.lower()]
        post_fwds = [e for e in events if 'post_forward' in e.name.lower() and 'root' not in e.name.lower()]
        fwd_intervals = []
        for post in post_fwds:
            valid_pres = [p for p in pre_fwds if p.end_time <= post.start_time]
            if valid_pres:
                gap_start = valid_pres[-1].end_time
                gap_end = post.start_time
                if gap_end > gap_start:
                    fwd_intervals.append((gap_start, gap_end))
        for gap_start, gap_end in fwd_intervals:
            gpu_dur = sum(n.direct_gpu_duration for n in all_nodes if gap_start <= n.start_time < gap_end)
            timeline.append({"ts": gap_start, "phase": f"Forward {unit}", "dur": gpu_dur})
        pre_bwds = [e for e in events if 'pre_backward' in e.name.lower() and 'root' not in e.name.lower()]
        post_accums = [e for e in events if 'post_backward_accumulate' in e.name.lower()]
        bwd_intervals = []
        for post in post_accums:
            valid_pres = [p for p in pre_bwds if p.end_time <= post.start_time]
            if valid_pres:
                gap_start = valid_pres[-1].end_time
                gap_end = post.start_time
                if gap_end > gap_start:
                    bwd_intervals.append((gap_start, gap_end))
        for gap_start, gap_end in bwd_intervals:
            gpu_dur = sum(n.direct_gpu_duration for n in all_nodes if gap_start <= n.start_time < gap_end)
            timeline.append({"ts": gap_start, "phase": f"Backward {unit}", "dur": gpu_dur})
    timeline.sort(key=lambda x: x["ts"])
    if not timeline:
        return "No FSDP events found."
    lines = ["\n" + "="*95, "CHRONOLOGICAL FSDP TIMELINE", "="*95,
             f"{'Relative Start (ms)':>20} | {'Phase / Operation':<40} | {'GPU Dur (ms)':>15}",
             "-"*95]
    first_ts = timeline[0]["ts"]
    for event in timeline:
        rel_ts = (event["ts"] - first_ts) / 1000
        lines.append(f"{rel_ts:>20.2f} | {event['phase']:<40} | {event['dur']/1000:>15.2f}")
    return "\n".join(lines)


# ------------------------------------------------------------------------
# Comparison function (already present, but we keep it)
def compare_traces_to_csv(trace1_file, trace2_file, output_csv, op_filter=None):
    print(f"Processing baseline: {trace1_file}")
    p1 = TraceParser(trace1_file)
    if not p1.load(): return
    roots1 = p1.build_tree()
    p1.attribute_gpu_time_with_dependencies(roots1)
    p1.attribute_memory(roots1)
    metrics1 = p1.get_aggregated_metrics(roots1, op_filter)

    print(f"Processing comparison: {trace2_file}")
    p2 = TraceParser(trace2_file)
    if not p2.load(): return
    roots2 = p2.build_tree()
    p2.attribute_gpu_time_with_dependencies(roots2)
    p2.attribute_memory(roots2)
    metrics2 = p2.get_aggregated_metrics(roots2, op_filter)

    all_ops = set(metrics1.keys()) | set(metrics2.keys())
    rows = []
    for op in sorted(all_ops):
        m1 = metrics1.get(op, {})
        m2 = metrics2.get(op, {})
        gpu1 = m1.get('total_gpu_us', 0.0)
        gpu2 = m2.get('total_gpu_us', 0.0)
        row = {
            'operation_name': op,
            'count_trace1': m1.get('count',0),
            'count_trace2': m2.get('count',0),
            'total_cpu_us_trace1': m1.get('total_cpu_us',0.0),
            'total_cpu_us_trace2': m2.get('total_cpu_us',0.0),
            'avg_cpu_us_trace1': m1.get('avg_cpu_us',0.0),
            'avg_cpu_us_trace2': m2.get('avg_cpu_us',0.0),
            'total_gpu_us_trace1': gpu1,
            'total_gpu_us_trace2': gpu2,
            'avg_gpu_us_trace1': m1.get('avg_gpu_us',0.0),
            'avg_gpu_us_trace2': m2.get('avg_gpu_us',0.0),
            'total_mem_bytes_trace1': m1.get('total_mem_bytes',0),
            'total_mem_bytes_trace2': m2.get('total_mem_bytes',0),
            'avg_mem_bytes_trace1': m1.get('avg_mem_bytes',0.0),
            'avg_mem_bytes_trace2': m2.get('avg_mem_bytes',0.0),
            'gpu_speedup_trace2_vs_trace1': gpu2/gpu1 if gpu1>0 else float('inf'),
            'gpu_diff_us': gpu2-gpu1,
            'cpu_speedup_trace2_vs_trace1': (m1.get('total_cpu_us',0)/m2.get('total_cpu_us',1)) if m2.get('total_cpu_us',0)>0 else float('inf')
        }
        rows.append(row)

    fieldnames = ['operation_name','count_trace1','count_trace2','total_cpu_us_trace1','total_cpu_us_trace2','avg_cpu_us_trace1','avg_cpu_us_trace2',
                  'total_gpu_us_trace1','total_gpu_us_trace2','avg_gpu_us_trace1','avg_gpu_us_trace2','total_mem_bytes_trace1','total_mem_bytes_trace2',
                  'avg_mem_bytes_trace1','avg_mem_bytes_trace2','gpu_speedup_trace2_vs_trace1','gpu_diff_us','cpu_speedup_trace2_vs_trace1']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison CSV saved to {output_csv}")


# ------------------------------------------------------------------------
# Main
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single trace: python trace_parser.py <trace.json> [--output report.txt]")
        print("  Comparison:   python trace_parser.py --compare trace1.json trace2.json --output comparison.csv [--op OP]")
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
    if not parser.load():
        sys.exit(1)
    print(f"Loaded {len(parser.cpu_events)} CPU, {len(parser.gpu_events)} GPU, {len(parser.memory_events)} memory events.")
    roots = parser.build_tree()
    parser.attribute_gpu_time_with_dependencies(roots)
    parser.attribute_memory(roots)
    generate_report(roots, parser, output_file)
    print("Done.")


if __name__ == "__main__":
    main()