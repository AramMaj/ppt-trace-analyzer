"""
Main parser for PyTorch Profiler traces

"""
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set, Iterator, Any
from dataclasses import dataclass, field
import heapq
import warnings

class TraceParserHelper: 
    @staticmethod
    def iter_nodes(roots: List['LogicalOperation']) -> Iterator['LogicalOperation']:
        """Iterate over all nodes in the tree using depth-first traversal."""
        stack = list(roots)
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

""" 
Data structure for logical operations

"""
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
    direct_gpu_kernels: Optional[Dict[str, Any]] = field(default_factory=list)
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

"""
Trace Parser

"""
class TraceParser: 
    GPU_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset', 'cuda_runtime',
                      'cuda_driver', 'gpu_user_annotation', 'gpu_op'}

    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.cpu_events = []
        self.cpu_events_by_pid_tid = defaultdict(list)
        self.gpu_events = []
        self.gpu_events_by_stream = defaultdict(list)
        self.all_events = []
        self.memory_events = []
        self.async_resolver = AsyncDependencyResolver()

    def load(self) -> bool:
        try:
            with open(self.trace_file, 'r') as f:
                self.data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self._load_error = str(e)
            print(f"Error loading trace: {self._load_error}")
            return False

        for ev in self.data.get('traceEvents', []):
            self.all_events.append(ev)
            pid, tid = ev.get('pid', 0), ev.get('tid', 0)
            cat = ev.get('cat', '')
            ph = ev.get('ph', '')
            name = ev.get('name', '')
            args = ev.get('args', {})
            stream = args.get('stream', -1)

            if ph == 'X' and cat in ('cpu_op', 'user_annotation'):
                self.cpu_events.append(ev)
                self.cpu_events_by_pid_tid[(pid, tid)].append(ev)
            elif stream != -1: 
                self.gpu_events.append(ev)
                self.gpu_events_by_stream[stream] = ev
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
        for (pid, tid), events in self.cpu_events_by_pid_tid.items():
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

    def attribute_gpu_kernel_with_logical_operation(self, roots: List[LogicalOperation]):
        # Build external ID map
        ext_to_gpu = defaultdict(float)
        ext_to_gpu_node = defaultdict(list)
        for gpu_ev in self.gpu_events:
            ext_id = gpu_ev.get('args', {}).get('External id')
            if ext_id is not None:
                ext_to_gpu[ext_id] += gpu_ev.get('dur', 0)
                ext_to_gpu_node[ext_id].append(gpu_ev)
        
        all_nodes = list(TraceParserHelper.iter_nodes(roots))
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
            for evlist in self.cpu_events_by_pid_tid.values():
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
                node.direct_gpu_kernels.append(gpu_ev)

        for node in all_nodes:
            if "cudalaunch" in node.name.lower() or "cudamemcpyasync" in node.name.lower(): 
                for ext_id in node.external_ids:
                    node.direct_gpu_duration += ext_to_gpu.get(ext_id, 0)
                    node.direct_gpu_kernels.append(ext_to_gpu_node[ext_id])

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

        all_nodes = list(TraceParserHelper.iter_nodes(roots))
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

"""
Async Dependency Resolver 

"""
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
    
