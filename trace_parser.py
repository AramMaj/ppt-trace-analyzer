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
from typing import Dict, List, Optional, Tuple, Set, Iterator, Any, Union
from enum import Enum
import warnings

# ----------------------------------------------------------------------
# Helper function: iterate through the tree of logical operations
def iter_nodes(roots: List['LogicalOperation']) -> Iterator['LogicalOperation']:
    """Iterate over all nodes in the tree using depth-first traversal.
    
    This function provides a memory-efficient way to traverse the entire
    operation tree without recursion depth limitations.
    
    Args:
        roots: List of root nodes in the call tree.
    
    Yields:
        Each node in the tree in depth-first order.
    """
    stack = list(roots)
    while stack:
        node = stack.pop()
        yield node
        # Extend with children in reverse order to maintain approximate order
        stack.extend(reversed(node.children))


# -----------------------------------------------------------------------
# Enums for better type safety and readability
class EventType(Enum):
    """Classification of trace events."""
    CPU_OP = "cpu_op"
    GPU_KERNEL = "gpu_kernel"
    GPU_MEMCPY = "gpu_memcpy"
    GPU_MEMSET = "gpu_memset"
    MEMORY = "memory"
    PYTHON_FUNCTION = "python_function"
    UNKNOWN = "unknown"


class PhaseType(Enum):
    """FSDP phase classification."""
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
# Data structure for a logical operation / node in the call tree
@dataclass
class LogicalOperation:
    """Represents a logical operation in the execution trace.
    
    This data structure encapsulates all timing and resource information
    for a single operation in the PyTorch trace. It maintains a hierarchical
    parent-child relationship for accurate attribution.
    
    Attributes:
        name: Operation name as reported by the profiler.
        start_time: Start timestamp in microseconds.
        end_time: End timestamp in microseconds.
        cpu_duration: Duration on CPU in microseconds.
        gpu_duration: Attributed GPU duration in microseconds (including children).
        memory_delta: Net memory change in bytes attributed to this operation.
        children: List of child operations.
        external_ids: Set of external IDs linking to GPU events.
        direct_gpu_duration: GPU duration directly attributed (excluding children).
        raw_event: Original raw event dictionary for debugging.
    """
    name: str
    start_time: float
    end_time: float
    cpu_duration: float
    gpu_duration: float = 0.0
    memory_delta: int = 0
    children: List['LogicalOperation'] = field(default_factory=list)
    external_ids: Set[int] = field(default_factory=set)
    direct_gpu_duration: float = 0.0
    raw_event: Optional[Dict[str, Any]] = None
    
    @property
    def total_time(self) -> float:
        """Maximum of CPU and GPU duration in microseconds.
        
        This represents the wall-clock time impact of this operation.
        """
        return max(self.cpu_duration, self.gpu_duration)

    @property
    def exclusive_cpu(self) -> float:
        """CPU duration excluding children in microseconds.
        
        This represents the pure CPU overhead of this operation itself.
        """
        return self.cpu_duration - sum(c.cpu_duration for c in self.children)

    @property
    def exclusive_gpu(self) -> float:
        """GPU duration excluding children in microseconds.
        
        This represents the pure GPU work done by this operation itself.
        """
        return self.gpu_duration - sum(c.gpu_duration for c in self.children)
    
    @property
    def gpu_utilization(self) -> float:
        """GPU utilization relative to CPU duration.
        
        Returns:
            Ratio of GPU time to CPU time (clipped to [0, 1]).
        """
        if self.cpu_duration > 0:
            return min(1.0, self.gpu_duration / self.cpu_duration)
        return 0.0


# ------------------------------------------------------------------------
# Async dependency resolver (CUDA events, stream wait, async intervals)
class AsyncDependencyResolver:
    """Resolves asynchronous GPU-CPU event dependencies in profiler traces.
    
    This class implements algorithms for matching async events across different
    streams and threads, crucial for accurate GPU time attribution.
    
    Key algorithms:
        - CUDA event record-wait pattern detection
        - Flow event (async bracket) matching
        - Memory allocation event tracking
    """
    
    def __init__(self):
        self.event_record_times: Dict[Tuple[int, int], float] = {}  # (event_ptr, stream_ptr) -> timestamp
        self.pending_intervals: Dict[str, Tuple[float, Dict[str, Any]]] = {}  # id -> (start_time, event)
        self.dependencies: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []  # (start_event, end_event)
        
    def process_event(self, ev: Dict[str, Any]) -> None:
        """Process a single event for async dependency detection.
        
        This method handles multiple async patterns:
        1. CUDA event recording
        2. CUDA stream wait events
        3. Async brackets (b/e)
        4. Flow events (s/f)
        
        Args:
            ev: Raw event dictionary from trace.
        """
        ph = ev.get('ph', '')
        cat = ev.get('cat', '')
        name = ev.get('name', '')
        args = ev.get('args', {})
        ev_id = args.get('id') or ev.get('id')
        ts = ev.get('ts', 0)

        # Handle CUDA event recording
        if 'cudaEventRecord' in name or (cat == 'cuda_runtime' and 'record' in name.lower()):
            event_ptr = args.get('event')
            stream_ptr = args.get('stream')
            if event_ptr is not None:
                self.event_record_times[(event_ptr, stream_ptr)] = ts
            return

        # Handle CUDA stream wait events
        if 'cudaStreamWaitEvent' in name or (cat == 'cuda_runtime' and 'wait' in name.lower()):
            event_ptr = args.get('event')
            # Find matching event record
            for (ev_ptr, _), rec_ts in self.event_record_times.items():
                if ev_ptr == event_ptr:
                    self.pending_intervals[f"wait_{ts}_{event_ptr}"] = (rec_ts, ev)
                    break
            return

        # Handle async brackets (b/e)
        if ph in ('b', 's') and ev_id is not None:
            self.pending_intervals[ev_id] = (ts, ev)
        elif ph in ('e', 'f') and ev_id is not None:
            if ev_id in self.pending_intervals:
                start_ts, start_ev = self.pending_intervals.pop(ev_id)
                self.dependencies.append((start_ev, ev))
                ev['_async_start'] = start_ts
                ev['_async_end'] = ts
                start_ev['_async_interval'] = (start_ts, ts)

        # Handle flow events (additional async tracking)
        if ph == 'f' and 'bp' in args and ev_id is not None:
            if args.get('bp') == 's':
                self.pending_intervals[f"flow_{ev_id}"] = (ts, ev)
            elif args.get('bp') == 'e':
                key = f"flow_{ev_id}"
                if key in self.pending_intervals:
                    start_ts, start_ev = self.pending_intervals.pop(key)
                    self.dependencies.append((start_ev, ev))
                    start_ev['_async_interval'] = (start_ts, ts)

    def resolve_dependencies(self, all_events: List[Dict[str, Any]]) -> None:
        """Resolve all async dependencies in the event list.
        
        Args:
            all_events: Complete list of events from the trace.
        """
        for ev in all_events:
            self.process_event(ev)

    def get_parent_from_async_only(self, gpu_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the CPU parent for a GPU event using async intervals.
        
        Args:
            gpu_event: GPU event to find parent for.
            
        Returns:
            The CPU event that triggered this GPU event, or None if not found.
        """
        ts = gpu_event.get('ts', 0)
        for start_ev, _ in self.dependencies:
            interval = start_ev.get('_async_interval')
            if interval and interval[0] <= ts <= interval[1]:
                return start_ev
        return None


# ------------------------------------------------------------------------
# Trace parser class
class TraceParser:
    """Main parser for PyTorch Profiler traces.
    
    This class handles loading, parsing, and building a hierarchical
    representation of the trace data. It implements algorithms for:
    - Building the CPU operation tree
    - Attributing GPU events to CPU operations
    - Tracing memory allocations
    - Supporting FSDP-specific analysis
    """
    
    # GPU event categories based on PyTorch Profiler classification
    GPU_CATEGORIES: Set[str] = {
        'kernel', 'gpu_memcpy', 'gpu_memset', 'cuda_runtime',
        'cuda_driver', 'gpu_user_annotation', 'gpu_op'
    }
    
    def __init__(self, trace_file: str):
        """Initialize the parser with a trace file path.
        
        Args:
            trace_file: Path to the JSON trace file.
        """
        self.trace_file = trace_file
        self.data: Optional[Dict[str, Any]] = None
        self.events_by_pid_tid: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        self.cpu_events: List[Dict[str, Any]] = []
        self.gpu_events: List[Dict[str, Any]] = []
        self.all_events: List[Dict[str, Any]] = []
        self.memory_events: List[Dict[str, Any]] = []
        self.distributed_info: Dict[str, Any] = {}
        self.async_resolver = AsyncDependencyResolver()
        self._load_error: Optional[str] = None

    def load(self) -> bool:
        """Load and preprocess the trace file.
        
        Returns:
            True if loading succeeded, False otherwise.
        """
        try:
            with open(self.trace_file, 'r') as f:
                self.data = json.load(f)
        except json.JSONDecodeError as e:
            self._load_error = f"Invalid JSON: {e}"
            print(f"Error loading trace: {self._load_error}")
            return False
        except FileNotFoundError:
            self._load_error = f"File not found: {self.trace_file}"
            print(self._load_error)
            return False
        
        self.distributed_info = self.data.get('distributedInfo', {})
        
        for ev in self.data.get('traceEvents', []):
            self.all_events.append(ev)
            pid, tid = ev.get('pid', 0), ev.get('tid', 0)
            cat = ev.get('cat', '')
            ph = ev.get('ph', '')
            name = ev.get('name', '')

            # Classify events by type
            if ph == 'X' and cat in ('cpu_op', 'user_annotation'):
                self.cpu_events.append(ev)
                self.events_by_pid_tid[(pid, tid)].append(ev)
            elif ph == 'X' and self._is_gpu_event(cat, name):
                self.gpu_events.append(ev)
            elif ph in ('i', 'C') and ('memory' in cat.lower() or name == "[memory]"):
                self.memory_events.append(ev)

        # Resolve async dependencies for accurate GPU attribution
        self.async_resolver.resolve_dependencies(self.all_events)
        return True

    @classmethod
    def _is_gpu_event(cls, cat: str, name: str = "") -> bool:
        """Determine if an event is a GPU event.
        
        Args:
            cat: Event category.
            name: Event name (optional).
            
        Returns:
            True if the event is GPU-related.
        """
        return cat in cls.GPU_CATEGORIES or 'gpu' in cat.lower() or 'cuda' in cat.lower()

    def build_tree(self) -> List[LogicalOperation]:
        """Build a hierarchical tree of logical operations.
        
        This uses nested intervals to reconstruct the parent-child relationships
        from the flat event list.
        
        Returns:
            List of root LogicalOperation nodes.
        """
        all_roots = []
        
        for (pid, tid), events in self.events_by_pid_tid.items():
            # Sort by start time for interval-based nesting
            events_sorted = sorted(events, key=lambda e: e['ts'])
            stack = []  # List of (event, node) pairs
            
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

    def attribute_gpu_time_with_dependencies(self, roots: List[LogicalOperation]) -> None:
        """Attribute GPU execution time to CPU operations with async handling.
        
        This implements a three-step algorithm:
        1. Direct External ID mapping (fast path)
        2. Async dependency matching (accurate path)
        3. Greedy interval matching (fallback)
        
        Args:
            roots: List of root operations to process.
        """
        # Step 1: Build external ID mapping for fast direct lookup
        ext_to_gpu = defaultdict(float)
        for gpu_ev in self.gpu_events:
            ext_id = gpu_ev.get('args', {}).get('External id')
            if ext_id is not None:
                ext_to_gpu[ext_id] += gpu_ev.get('dur', 0)

        # Step 2: Build node lookup dictionary
        all_nodes = list(iter_nodes(roots))
        event_to_node = {id(node.raw_event): node for node in all_nodes if node.raw_event is not None}

        # Step 3: Match GPU events to CPU parents
        gpu_parent_pairs = []  # (gpu_event, parent_cpu_event)
        
        # Process GPU events with External ID separately
        remaining_gpu = []
        for gpu_ev in self.gpu_events:
            ext_id = gpu_ev.get('args', {}).get('External id')
            if ext_id is not None:
                continue  # Will be handled via ext_to_gpu mapping
            # Try async dependency resolution
            parent = self.async_resolver.get_parent_from_async_only(gpu_ev)
            if parent is not None:
                gpu_parent_pairs.append((gpu_ev, parent))
            else:
                remaining_gpu.append(gpu_ev)

        # Step 4: Greedy interval matching for remaining GPU events
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
                # Add all CPU events that started before or at this GPU event
                while cpu_idx < n_cpu and all_cpu[cpu_idx]['ts'] <= ts:
                    ce = all_cpu[cpu_idx]
                    dur = ce.get('dur', 0)
                    end = ce['ts'] + dur
                    # Use negative start to get innermost active interval
                    heapq.heappush(heap, (-ce['ts'], end, ce))
                    cpu_idx += 1
                # Remove CPU events that ended before this GPU event
                while heap and heap[0][1] < ts:
                    heapq.heappop(heap)
                if heap:
                    gpu_parent_pairs.append((gpu_ev, heap[0][2]))

        # Step 5: Attribute GPU time to nodes (direct + external)
        for gpu_ev, parent_cpu in gpu_parent_pairs:
            node = event_to_node.get(id(parent_cpu))
            if node is not None:
                node.direct_gpu_duration += gpu_ev.get('dur', 0)

        # Add external ID contributions (avoid double counting)
        for node in all_nodes:
            for ext_id in node.external_ids:
                node.direct_gpu_duration += ext_to_gpu.get(ext_id, 0)

        # Step 6: Propagate GPU time upward
        def _compute_gpu(node: LogicalOperation) -> float:
            child_gpu = sum(_compute_gpu(ch) for ch in node.children)
            node.gpu_duration = node.direct_gpu_duration + child_gpu
            return node.gpu_duration
            
        for root in roots:
            _compute_gpu(root)

    def attribute_memory(self, roots: List[LogicalOperation]) -> None:
        """Attribute memory allocations to operations.
        
        This matches memory events to the innermost active operation
        and propagates net changes upward.
        
        Args:
            roots: List of root operations to process.
        """
        if not self.memory_events:
            warnings.warn("No memory events found. Memory profiling may be disabled in the trace.")
            return

        all_nodes = list(iter_nodes(roots))
        all_nodes.sort(key=lambda n: n.start_time)

        mem_events_sorted = sorted(self.memory_events, key=lambda e: e['ts'])
        heap = []  # (-start_time, end_time, node)
        node_idx = 0
        n_nodes = len(all_nodes)

        for mem_ev in mem_events_sorted:
            ts = mem_ev['ts']
            # Add all nodes that started before or at this memory event
            while node_idx < n_nodes and all_nodes[node_idx].start_time <= ts:
                node = all_nodes[node_idx]
                heapq.heappush(heap, (-node.start_time, node.end_time, node))
                node_idx += 1
            # Remove nodes that ended before this memory event
            while heap and heap[0][1] < ts:
                heapq.heappop(heap)
            if heap:
                deepest_node = heap[0][2]
                args = mem_ev.get('args', {})
                size = args.get('Bytes', args.get('size', args.get('bytes', 0)))
                device_type = args.get('Device Type', -1)
                # Only attribute GPU memory (device_type == 1)
                if device_type == 1:
                    deepest_node.memory_delta += size

        # Propagate memory deltas upward
        def _propagate_memory(node: LogicalOperation) -> int:
            total = node.memory_delta
            for ch in node.children:
                total += _propagate_memory(ch)
            node.memory_delta = total
            return total
            
        for root in roots:
            _propagate_memory(root)

    def get_aggregated_metrics(self, roots: List[LogicalOperation], op_filter: Optional[str] = None) -> Dict[str, Dict]:
        """Compute aggregated metrics across all operations.
        
        Args:
            roots: List of root operations to analyze.
            op_filter: Optional name filter (substring match).
            
        Returns:
            Dictionary mapping operation names to metric dictionaries.
        """
        all_nodes = list(iter_nodes(roots))
        agg = defaultdict(lambda: {
            'count': 0,
            'total_cpu_us': 0.0,
            'total_gpu_us': 0.0,
            'total_mem_bytes': 0
        })

        for node in all_nodes:
            if op_filter and op_filter not in node.name:
                continue
            agg[node.name]['count'] += 1
            agg[node.name]['total_cpu_us'] += node.cpu_duration
            agg[node.name]['total_gpu_us'] += node.gpu_duration
            agg[node.name]['total_mem_bytes'] += node.memory_delta

        # Compute averages
        for name, data in agg.items():
            c = data['count']
            if c > 0:
                data['avg_cpu_us'] = data['total_cpu_us'] / c
                data['avg_gpu_us'] = data['total_gpu_us'] / c
                data['avg_mem_bytes'] = data['total_mem_bytes'] / c
            else:
                data['avg_cpu_us'] = 0.0
                data['avg_gpu_us'] = 0.0
                data['avg_mem_bytes'] = 0.0

        return dict(agg)


# ------------------------------------------------------------------------
# FSDP specific analysis
def extract_unit_from_name(name: str) -> Optional[str]:
    """Extract the FSDP unit (layer name) from an operation name.
    
    This function handles two naming patterns:
    1. Pattern in parentheses: "FSDP::all_gather (layers.0)"
    2. Pattern after 'for': "Post-Forward for layers.1"
    
    Args:
        name: Operation name string.
        
    Returns:
        Extracted unit name or None if not found.
    """
    # Pattern 1: Operation name with parentheses
    m = re.search(r'\((.*?)\)', name)
    if m:
        val = m.group(1).replace('model.', '')
        if 'layers' in val or 'tok_embeddings' in val or 'norm' in val or 'head' in val:
            return val
    # Pattern 2: "for ..." pattern
    m = re.search(r'for\s+([a-zA-Z0-9_\.]+)', name)
    if m:
        val = m.group(1).replace('model.', '')
        if 'layers' in val or 'tok_embeddings' in val or 'norm' in val or 'head' in val:
            return val
    return None


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or adjacent time intervals.
    
    Args:
        intervals: List of (start, end) timestamp pairs.
        
    Returns:
        Merged list of non-overlapping intervals.
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for cur in intervals[1:]:
        prev = merged[-1]
        if cur[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], cur[1]))
        else:
            merged.append(cur)
    return merged


def extract_fsdp_phases_aggregated(roots: List[LogicalOperation]) -> Dict[str, Dict[str, float]]:
    """Extract FSDP phase times aggregated by layer unit.
    
    This function identifies and quantifies:
    - All-Gather (AG) communication operations
    - Reduce-Scatter (RS) communication operations  
    - Forward computation gaps between Pre-Forward and Post-Forward
    - Backward computation gaps between Pre-Backward and Accumulate
    
    Args:
        roots: List of root operations to analyze.
        
    Returns:
        Nested dictionary: unit -> {phase_name: total_gpu_us}
    """
    agg = defaultdict(lambda: defaultdict(float))
    all_nodes = list(iter_nodes(roots))

    # Group operations by unit (layer)
    unit_events = defaultdict(list)
    for n in all_nodes:
        if 'FSDP::' in n.name or 'FSDP' in n.name:
            unit = extract_unit_from_name(n.name)
            if unit:
                unit_events[unit].append(n)

    for unit, events in unit_events.items():
        events.sort(key=lambda x: x.start_time)
        
        # Phase 1: All-Gather detection
        for ev in events:
            name_lower = ev.name.lower()
            if 'all_gather' in name_lower and 'copy_out' not in name_lower:
                agg[unit][PhaseType.ALL_GATHER.value] += ev.gpu_duration
                
        # Phase 2: Reduce-Scatter detection
        for ev in events:
            name_lower = ev.name.lower()
            rs_keywords = ['post_backward_reshard', 'post_backward_reduce', 'post_backward_rs_wait']
            if any(kw in name_lower for kw in rs_keywords):
                agg[unit][PhaseType.REDUCE_SCATTER.value] += ev.gpu_duration

        # Phase 3: Forward compute gaps (between Pre-Forward and Post-Forward)
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
                    
        for gap_start, gap_end in merge_intervals(fwd_intervals):
            gpu_dur = sum(n.direct_gpu_duration for n in all_nodes if gap_start <= n.start_time < gap_end)
            agg[unit][PhaseType.FORWARD_COMPUTE.value] += gpu_dur

        # Phase 4: Backward compute gaps (between Pre-Backward and Accumulate)
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
                    
        for gap_start, gap_end in merge_intervals(bwd_intervals):
            gpu_dur = sum(n.direct_gpu_duration for n in all_nodes if gap_start <= n.start_time < gap_end)
            agg[unit][PhaseType.BACKWARD_COMPUTE.value] += gpu_dur

        # Phase 5: Parameter-free detection (no communication phases)
        if agg[unit][PhaseType.ALL_GATHER.value] == 0 and agg[unit][PhaseType.REDUCE_SCATTER.value] == 0:
            agg[unit][PhaseType.PARAMETER_FREE.value] = (
                agg[unit][PhaseType.FORWARD_COMPUTE.value] + 
                agg[unit][PhaseType.BACKWARD_COMPUTE.value]
            )

    return dict(agg)


def get_fsdp_timeline_aggregated_string(agg: Dict[str, Dict[str, float]]) -> str:
    """Generate a formatted string for aggregated FSDP timeline.
    
    Args:
        agg: Aggregated FSDP phase data from extract_fsdp_phases_aggregated.
        
    Returns:
        Formatted string for display or file output.
    """
    if not agg:
        return "No FSDP phases found in the trace."

    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("FSDP Phase Summary (Aggregated GPU Time per Layer)")
    lines.append("=" * 90)
    lines.append(
        f"{'Layer':<15} {'All-Gather (ms)':>18} {'Reduce-Scatter (ms)':>20} "
        f"{'Forward (ms)':>15} {'Backward (ms)':>15}"
    )
    lines.append("-" * 90)

    def _layer_key(unit: str) -> int:
        """Extract layer number for sorting."""
        try:
            parts = unit.split()
            if parts and parts[-1].isdigit():
                return int(parts[-1])
            # Try to find any number in the string
            numbers = re.findall(r'\d+', unit)
            return int(numbers[0]) if numbers else 0
        except (ValueError, IndexError):
            return 0

    for unit in sorted(agg.keys(), key=_layer_key):
        data = agg[unit]
        lines.append(
            f"{unit:<15} {data.get('All-Gather', 0) / 1000:>18.2f} "
            f"{data.get('Reduce-Scatter', 0) / 1000:>20.2f} "
            f"{data.get('Forward Comp', 0) / 1000:>15.2f} "
            f"{data.get('Backward Comp', 0) / 1000:>15.2f}"
        )

    total_ag = sum(d.get('All-Gather', 0) for d in agg.values())
    total_rs = sum(d.get('Reduce-Scatter', 0) for d in agg.values())
    total_fwd = sum(d.get('Forward Comp', 0) for d in agg.values())
    total_bwd = sum(d.get('Backward Comp', 0) for d in agg.values())
    lines.append("-" * 90)
    lines.append(
        f"{'TOTAL (all layers)':<15} {total_ag / 1000:>18.2f} {total_rs / 1000:>20.2f} "
        f"{total_fwd / 1000:>15.2f} {total_bwd / 1000:>15.2f}"
    )
    return "\n".join(lines)


def get_fsdp_chronological_timeline(roots: List[LogicalOperation]) -> str:
    """Generate a chronological timeline of FSDP events.
    
    Args:
        roots: List of root operations to analyze.
        
    Returns:
        Formatted timeline string.
    """
    timeline = []
    all_nodes = list(iter_nodes(roots))
    all_nodes.sort(key=lambda n: n.start_time)

    # Group by unit
    unit_events = defaultdict(list)
    for n in all_nodes:
        if 'FSDP::' in n.name or 'FSDP' in n.name:
            unit = extract_unit_from_name(n.name)
            if unit:
                unit_events[unit].append(n)

    for unit, events in unit_events.items():
        events.sort(key=lambda x: x.start_time)
        
        for ev in events:
            name_lower = ev.name.lower()
            if 'all_gather' in name_lower and 'copy_out' not in name_lower:
                timeline.append({
                    "ts": ev.start_time,
                    "phase": f"All-Gather ({unit})",
                    "dur": ev.gpu_duration
                })
            elif any(kw in name_lower for kw in ['post_backward_reshard', 'post_backward_reduce', 'post_backward_rs_wait']):
                timeline.append({
                    "ts": ev.start_time,
                    "phase": f"Reduce-Scatter ({unit})",
                    "dur": ev.gpu_duration
                })

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
                    
        for gap_start, gap_end in merge_intervals(fwd_intervals):
            gpu_dur = sum(n.direct_gpu_duration for n in all_nodes if gap_start <= n.start_time < gap_end)
            timeline.append({
                "ts": gap_start,
                "phase": f"Forward Compute ({unit})",
                "dur": gpu_dur
            })

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
                    
        for gap_start, gap_end in merge_intervals(bwd_intervals):
            gpu_dur = sum(n.direct_gpu_duration for n in all_nodes if gap_start <= n.start_time < gap_end)
            timeline.append({
                "ts": gap_start,
                "phase": f"Backward Compute ({unit})",
                "dur": gpu_dur
            })

    timeline.sort(key=lambda x: x["ts"])

    if not timeline:
        return "No FSDP events found in the trace."

    lines = []
    lines.append("\n" + "=" * 95)
    lines.append("Chronological FSDP Timeline")
    lines.append("=" * 95)
    lines.append(
        f"{'Relative Start (ms)':>20} | {'Phase / Operation':<45} | {'GPU Duration (ms)':>15}"
    )
    lines.append("-" * 95)

    first_ts = timeline[0]["ts"]
    for event in timeline:
        rel_ts = (event["ts"] - first_ts) / 1000  # Convert µs to ms
        lines.append(
            f"{rel_ts:>20.2f} | {event['phase']:<45} | {event['dur'] / 1000:>15.2f}"
        )
    
    return "\n".join(lines)


# ------------------------------------------------------------------------
# Bottleneck detection and reporting
class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU_BOUND = "CPU-bound (high CPU overhead)"
    GPU_UNDERUTILIZED = "GPU underutilization (GPU idle waiting)"
    HIGH_MEMORY = "High memory footprint"
    COMMUNICATION_BOUND = "Communication bottleneck"
    COMPUTE_BOUND = "Compute bound"


@dataclass
class BottleneckInfo:
    """Information about a performance bottleneck."""
    type: BottleneckType
    severity: float  # 0-1 scale
    details: str
    
    def __str__(self) -> str:
        return f"{self.type.value} (severity: {self.severity:.2f})"


class BottleneckDetector:
    """Detects and classifies performance bottlenecks in operations.
    
    This class analyzes timing and memory patterns to identify common
    performance issues in PyTorch training workloads.
    
    Criteria:
        - CPU-bound: CPU time > 1.5x GPU time and exclusive CPU > 10% of total
        - GPU underutilization: GPU time < 50% of CPU time and GPU time > 0
        - High memory: Net memory delta > 1 GB
        - Communication bottleneck: Communication operations dominate
    """
    
    @staticmethod
    def detect(node: LogicalOperation) -> List[BottleneckInfo]:
        """Detect bottlenecks in a given operation node.
        
        Args:
            node: Operation node to analyze.
            
        Returns:
            List of detected bottleneck info objects.
        """
        bottlenecks = []
        total = node.total_time
        
        if total > 0:
            # CPU-bound detection
            if node.cpu_duration > 1.5 * node.gpu_duration and node.exclusive_cpu > 0.1 * total:
                severity = min(1.0, (node.cpu_duration - node.gpu_duration) / node.cpu_duration)
                bottlenecks.append(BottleneckInfo(
                    type=BottleneckType.CPU_BOUND,
                    severity=severity,
                    details=f"CPU: {node.cpu_duration:.2f}µs, GPU: {node.gpu_duration:.2f}µs"
                ))
            
            # GPU underutilization detection
            if node.gpu_duration < 0.5 * node.cpu_duration and node.gpu_duration > 0:
                severity = 1.0 - (node.gpu_duration / node.cpu_duration)
                bottlenecks.append(BottleneckInfo(
                    type=BottleneckType.GPU_UNDERUTILIZED,
                    severity=severity,
                    details=f"GPU only {node.gpu_duration/node.cpu_duration:.1%} of CPU time"
                ))
            
            # High memory footprint
            if abs(node.memory_delta) > 1024 * 1024 * 1024:  # 1 GB
                severity = min(1.0, abs(node.memory_delta) / (10 * 1024 * 1024 * 1024))  # Cap at 10GB
                bottlenecks.append(BottleneckInfo(
                    type=BottleneckType.HIGH_MEMORY,
                    severity=severity,
                    details=f"Memory delta: {node.memory_delta / (1024**3):.2f} GB"
                ))
        
        # Communication bottleneck in children
        for ch in node.children:
            if "all_gather" in ch.name.lower() or "allgather" in ch.name.lower():
                if ch.gpu_duration > 0.3 * node.gpu_duration:
                    severity = ch.gpu_duration / node.gpu_duration if node.gpu_duration > 0 else 1.0
                    bottlenecks.append(BottleneckInfo(
                        type=BottleneckType.COMMUNICATION_BOUND,
                        severity=severity,
                        details=f"All-Gather takes {ch.gpu_duration/node.gpu_duration:.1%} of GPU time"
                    ))
                    
        return bottlenecks


# ------------------------------------------------------------------------
# Formatting utilities
def format_time(us: float) -> str:
    """Format time in microseconds to human-readable string.
    
    Args:
        us: Time in microseconds.
        
    Returns:
        Formatted string with appropriate units (µs, ms, s).
    """
    if us < 1e3:
        return f"{us:.2f} µs"
    elif us < 1e6:
        return f"{us / 1e3:.2f} ms"
    else:
        return f"{us / 1e6:.2f} s"


def format_memory(bytes_val: int) -> str:
    """Format memory in bytes to human-readable string.
    
    Args:
        bytes_val: Memory in bytes.
        
    Returns:
        Formatted string with appropriate units (B, KB, MB, GB).
    """
    abs_bytes = abs(bytes_val)
    if abs_bytes < 1024:
        return f"{bytes_val} B"
    elif abs_bytes < 1024 ** 2:
        return f"{bytes_val / 1024:.2f} KB"
    elif abs_bytes < 1024 ** 3:
        return f"{bytes_val / 1024 ** 2:.2f} MB"
    else:
        return f"{bytes_val / 1024 ** 3:.2f} GB"


def get_top_k_string(roots: List[LogicalOperation], k: int = 10) -> str:
    """Generate a string of the k most expensive operations.
    
    Args:
        roots: List of root operations.
        k: Number of top operations to display.
        
    Returns:
        Formatted string table.
    """
    all_nodes = list(iter_nodes(roots))
    all_nodes.sort(key=lambda n: n.total_time, reverse=True)
    
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"TOP {k} MOST EXPENSIVE OPERATIONS (GPU+CPU Time)")
    lines.append(f"{'=' * 80}")
    lines.append(f"{'Name':<50} {'CPU':>12} {'GPU':>12} {'Memory Δ':>12} {'Bottleneck(s)'}")
    lines.append(f"{'-' * 80}")
    
    for node in all_nodes[:k]:
        bottlenecks = BottleneckDetector.detect(node)
        bottle_str = "; ".join(str(b) for b in bottlenecks[:2]) if bottlenecks else "-"
        lines.append(
            f"{node.name[:50]:<50} {format_time(node.cpu_duration):>12} "
            f"{format_time(node.gpu_duration):>12} {format_memory(node.memory_delta):>12} {bottle_str}"
        )
    return "\n".join(lines)


def get_flame_like_string(roots: List[LogicalOperation], max_depth: int = 3, indent: int = 0) -> str:
    """Generate a flame-graph-like breakdown of operations.
    
    Args:
        roots: List of root operations.
        max_depth: Maximum recursion depth to display.
        indent: Current indentation level.
        
    Returns:
        Formatted hierarchical string.
    """
    lines = []
    if indent == 0:
        lines.append(f"\n{'=' * 80}")
        lines.append("Flame-Graph Style Breakdown (Hierarchical)")
        lines.append(f"{'=' * 80}")
    
    for node in roots:
        prefix = "  " * indent
        bottlenecks = BottleneckDetector.detect(node)
        hint = f"  <-- {bottlenecks[0]}" if bottlenecks else ""
        lines.append(
            f"{prefix}{node.name} : CPU {format_time(node.cpu_duration)}, "
            f"GPU {format_time(node.gpu_duration)}, "
            f"Mem {format_memory(node.memory_delta)}{hint}"
        )
        if indent < max_depth:
            child_str = get_flame_like_string(node.children, max_depth, indent + 1)
            if child_str:
                lines.append(child_str)
    return "\n".join(lines)


def generate_report(roots: List[LogicalOperation], output_file: Optional[str] = None) -> None:
    """Generate a comprehensive analysis report.
    
    Args:
        roots: List of root operations.
        output_file: Optional output file path.
    """
    report_parts = []
    report_parts.append("=" * 70)
    report_parts.append("PyTorch Trace Analysis Report")
    report_parts.append("=" * 70)
    report_parts.append(f"Root operations: {len(roots)}")
    
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
        print(f"\nReport saved to {output_file}")


# ------------------------------------------------------------------------
# Benchmarking and comparison
def compare_traces_to_csv(
    trace1_file: str, 
    trace2_file: str, 
    output_csv: str, 
    op_filter: Optional[str] = None
) -> None:
    """Compare two traces and generate a CSV report.
    
    This function computes both absolute and relative differences between
    two traces, enabling quantitative performance regression analysis.
    
    Args:
        trace1_file: Path to first trace file (baseline).
        trace2_file: Path to second trace file (comparison).
        output_csv: Output CSV file path.
        op_filter: Optional operation name filter.
    """
    print(f"Processing baseline trace: {trace1_file}")
    p1 = TraceParser(trace1_file)
    if not p1.load():
        print(f"Failed to load {trace1_file}")
        return
    roots1 = p1.build_tree()
    p1.attribute_gpu_time_with_dependencies(roots1)
    p1.attribute_memory(roots1)
    metrics1 = p1.get_aggregated_metrics(roots1, op_filter)

    print(f"Processing comparison trace: {trace2_file}")
    p2 = TraceParser(trace2_file)
    if not p2.load():
        print(f"Failed to load {trace2_file}")
        return
    roots2 = p2.build_tree()
    p2.attribute_gpu_time_with_dependencies(roots2)
    p2.attribute_memory(roots2)
    metrics2 = p2.get_aggregated_metrics(roots2, op_filter)

    all_ops = set(metrics1.keys()) | set(metrics2.keys())
    rows = []
    
    for op in sorted(all_ops):
        m1 = metrics1.get(op, {})
        m2 = metrics2.get(op, {})
        
        # Helper to get values with defaults
        def _get(m, key, default=0):
            return m.get(key, default)
        
        gpu1 = _get(m1, 'total_gpu_us')
        gpu2 = _get(m2, 'total_gpu_us')
        
        row = {
            'operation_name': op,
            'count_trace1': _get(m1, 'count'),
            'count_trace2': _get(m2, 'count'),
            'total_cpu_us_trace1': _get(m1, 'total_cpu_us'),
            'total_cpu_us_trace2': _get(m2, 'total_cpu_us'),
            'avg_cpu_us_trace1': _get(m1, 'avg_cpu_us'),
            'avg_cpu_us_trace2': _get(m2, 'avg_cpu_us'),
            'total_gpu_us_trace1': gpu1,
            'total_gpu_us_trace2': gpu2,
            'avg_gpu_us_trace1': _get(m1, 'avg_gpu_us'),
            'avg_gpu_us_trace2': _get(m2, 'avg_gpu_us'),
            'total_mem_bytes_trace1': _get(m1, 'total_mem_bytes'),
            'total_mem_bytes_trace2': _get(m2, 'total_mem_bytes'),
            'avg_mem_bytes_trace1': _get(m1, 'avg_mem_bytes'),
            'avg_mem_bytes_trace2': _get(m2, 'avg_mem_bytes'),
            'gpu_speedup_trace2_vs_trace1': gpu2 / gpu1 if gpu1 > 0 else float('inf'),
            'gpu_diff_us': gpu2 - gpu1,
            'cpu_speedup_trace2_vs_trace1': (
                _get(m1, 'total_cpu_us') / _get(m2, 'total_cpu_us') 
                if _get(m2, 'total_cpu_us') > 0 else float('inf')
            ),
        }
        rows.append(row)

    fieldnames = [
        'operation_name', 'count_trace1', 'count_trace2',
        'total_cpu_us_trace1', 'total_cpu_us_trace2',
        'avg_cpu_us_trace1', 'avg_cpu_us_trace2',
        'total_gpu_us_trace1', 'total_gpu_us_trace2',
        'avg_gpu_us_trace1', 'avg_gpu_us_trace2',
        'total_mem_bytes_trace1', 'total_mem_bytes_trace2',
        'avg_mem_bytes_trace1', 'avg_mem_bytes_trace2',
        'gpu_speedup_trace2_vs_trace1', 'gpu_diff_us', 'cpu_speedup_trace2_vs_trace1'
    ]
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Comparison CSV saved to {output_csv}")
    print(f"Processed {len(rows)} operations")


# ------------------------------------------------------------------------
# Main entry point
def main() -> None:
    """Command-line interface for trace analysis."""
    if len(sys.argv) < 2:
        print("=" * 70)
        print("PyTorch Trace Parser - Scientific Trace Analysis")
        print("=" * 70)
        print("\nUsage:")
        print("  Single trace analysis:")
        print("      python trace_parser.py <trace.json> [--output report.txt]")
        print("\n  Benchmark (CSV comparison):")
        print("      python trace_parser.py --compare trace1.json trace2.json --output comparison.csv [--op OPERATION_NAME]")
        print("\nExamples:")
        print("  python trace_parser.py trace.json")
        print("  python trace_parser.py trace.json --output report.txt")
        print("  python trace_parser.py --compare baseline.json optimized.json --output perf.csv")
        print("  python trace_parser.py --compare trace1.json trace2.json --output fwd.csv --op 'all_gather'")
        sys.exit(1)

    # Comparison mode
    if sys.argv[1] == "--compare":
        if len(sys.argv) < 4:
            print("Error: --compare requires two trace files.")
            sys.exit(1)
        trace1 = sys.argv[2]
        trace2 = sys.argv[3]
        output_csv = None
        op_filter = None
        
        for i, arg in enumerate(sys.argv):
            if arg == "--output" and i + 1 < len(sys.argv):
                output_csv = sys.argv[i + 1]
            if arg == "--op" and i + 1 < len(sys.argv):
                op_filter = sys.argv[i + 1]
        
        if not output_csv:
            print("Error: Please specify --output comparison.csv")
            sys.exit(1)
            
        compare_traces_to_csv(trace1, trace2, output_csv, op_filter)
        sys.exit(0)

    # Single trace mode
    trace_file = sys.argv[1]
    output_file = None
    if len(sys.argv) >= 3 and sys.argv[2] == "--output":
        output_file = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Loading trace from {trace_file}...")
    parser = TraceParser(trace_file)
    if not parser.load():
        sys.exit(1)
        
    print(f"Loaded {len(parser.cpu_events)} CPU events, {len(parser.gpu_events)} GPU events, {len(parser.memory_events)} memory events.")

    print("Building logical operation tree...")
    roots = parser.build_tree()
    print(f"Built {len(roots)} root operations.")

    print("Performing CUDA dependency resolution...")
    parser.attribute_gpu_time_with_dependencies(roots)

    print("Attributing memory deltas...")
    parser.attribute_memory(roots)

    print("Generating report...")
    generate_report(roots, output_file)
    print("Done.")


if __name__ == "__main__":
    main()