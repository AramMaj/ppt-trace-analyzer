"""
Parser for converting Pytorch Profiler JSON Files in Chrome Trace Format 
to internal representations. 

Main parser steps: 
    load():              Traces are extracted from input trace file and get 
                         classified as CPU, Memory and GPU operations

    build_tree():        CPU events are structured as tree signifying 
                         caller and callee

    attribute_kernels(): Connect GPU Kernels to initiating CPU operations 

    attribute_memory():  Attribute memory usage to logical operations 
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set, Iterator, Any
from dataclasses import dataclass, field
import heapq
import warnings


# ---------------------------------------------------------------------------
# Tree iteration helper
# ---------------------------------------------------------------------------

class TraceParserHelper:
    """Static helpers for tree operations, namespaced per PPT convention."""

    @staticmethod
    def iter_nodes(roots: List['LogicalOperation']) -> Iterator['LogicalOperation']:
        """DFS over the tree — yields every node.  Stack-based with ``reversed(children)``
        for left-to-right traversal order (matches Chrome Trace Viewer).
        """
        stack = list(roots)
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass
class LogicalOperation:
    """One CPU trace event, enriched with GPU time, memory, and tree structure.

    Fields:
        name:              Operation name (e.g. ``"aten::linear"``)
        start_time, end_time, cpu_duration: Wall-clock bounds in µs.
        gpu_duration:      Inclusive GPU time (``direct_gpu_duration + sum(children)``).
        memory_delta:      Cumulative memory delta in bytes (self + children).
        children:          Time-contained sub-operations.
        external_ids:      ``External id`` correlation values for GPU attribution.
        direct_gpu_duration: GPU time attributed directly to this node only.
        direct_gpu_kernels:  Raw GPU event dicts attributed directly here.
        raw_event:         Original JSON event dict (for debugging / extra args).
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
    direct_gpu_kernels: List[dict] = field(default_factory=list)
    raw_event: Optional[Dict[str, Any]] = None

    @property
    def total_time(self) -> float:
        """Rough upper bound: whichever is larger between CPU dispatch and GPU execution.
        For a CPU-bound ``aten::empty`` this is the CPU duration; for a fused NVIDIA
        kernel launched by the backward stream thread it's the GPU duration.
        """
        return max(self.cpu_duration, self.gpu_duration)

    @property
    def exclusive_cpu(self) -> float:
        """CPU dispatch time attributed to this op alone (children subtracted).
        For ``aten::linear`` under an FSDP ``fwd_compute``, this excludes the
        sub-operation dispatch overhead — what the GEMM itself costs on the CPU side.
        """
        return self.cpu_duration - sum(c.cpu_duration for c in self.children)

    @property
    def exclusive_gpu(self) -> float:
        """GPU kernel time attributed solely to this node — no children included.
        After external-ID attribution, a node like ``all_gather_copy_out`` carries
        its split/memcpy GPU kernels here; the NCCL all-gather sibling is separate.
        """
        return self.gpu_duration - sum(c.gpu_duration for c in self.children)

    @property
    def gpu_utilization(self) -> float:
        """GPU busy-fraction vs CPU wall (capped at 1.0).
        Sub-0.2 is diagnostic of a host-bound layer where NCCL synchronisation or
        CUDA malloc dominates wall time — a common FSDP2 bottleneck on the 8B trace.
        """
        if self.cpu_duration > 0:
            return min(1.0, self.gpu_duration / self.cpu_duration)
        return 0.0


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class TraceParser:
    """ TraceParser loads trace file, classifies trace events and builds a tree of CPU operations.
        CPU operations get attributed with initiated GPU and memory usage. 
    """

    GPU_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset',
                      'gpu_user_annotation', 'gpu_op'}
    
    # Profiler wrapper events (annotation, op) are excluded from GPU_WORK — they
    # have zero duration and would inflate double-counting if attributed as work.
    GPU_WORK_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset'}

    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self._load_error: Optional[str] = None
        self.cpu_events: List[dict] = []
        self.cpu_events_by_pid_tid: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
        self.gpu_events: List[dict] = []
        self.gpu_events_by_stream: Dict[int, List[dict]] = defaultdict(list)
        self.all_events: List[dict] = []
        self.memory_events: List[dict] = []
        self.async_resolver = AsyncDependencyResolver()

    def load(self) -> bool:
        """Load JSON, classify events into CPU/GPU/memory lists, and resolve async dependencies.
        Validates minimal trace structure: rejects empty/degenerate files, counts
        events per category, and reports load errors via stderr rather than crashing.
        """
        try:
            with open(self.trace_file, 'r') as f:
                self.trace_json = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self._load_error = str(e)
            print(f"  Error loading trace: {self._load_error}")
            return False
        except PermissionError as e:
            self._load_error = str(e)
            print(f"  Permission denied: {self._load_error}")
            return False

        events = self.trace_json.get('traceEvents')
        if events is None:
            self._load_error = "traceEvents key missing from JSON root"
            print(f"  Error: {self._load_error}")
            return False
        if not isinstance(events, list):
            self._load_error = "traceEvents is not a list"
            print(f"  Error: {self._load_error}")
            return False
        if len(events) == 0:
            self._load_error = "traceEvents list is empty"
            print(f"  Warning: {self._load_error}")
            # Not fatal — allow analysis with warnings

        for ev in events:
            self.all_events.append(ev)
            pid, tid = ev.get('pid', 0), ev.get('tid', 0)
            cat = ev.get('cat', '')
            ph = ev.get('ph', '')
            name = ev.get('name', '')
            args = ev.get('args', {})
            stream = args.get('stream', args.get('Stream', -1))

            if ph == 'X' and cat in ('cpu_op', 'user_annotation', 'cuda_runtime', 'cuda_driver'):
                self.cpu_events.append(ev)
                if cat in ('cpu_op', 'user_annotation'):
                    self.cpu_events_by_pid_tid[(pid, tid)].append(ev)
            elif self._is_gpu_event(cat, name):
                # Augment GPU events with PG info for FSDP/TP classification
                pg_desc = args.get('Process Group Description', '')
                if pg_desc:
                    ev['_pg_desc'] = pg_desc
                coll_name = args.get('Collective name', '')
                if coll_name:
                    ev['_coll_name'] = coll_name
                self.gpu_events.append(ev)
            elif stream != -1 and cat not in ('cpu_op', 'user_annotation', 'cuda_runtime', 'cuda_driver'):
                # Fallback: events with a stream but non-CPU category are also GPU events
                pg_desc = args.get('Process Group Description', '')
                if pg_desc:
                    ev['_pg_desc'] = pg_desc
                coll_name = args.get('Collective name', '')
                if coll_name:
                    ev['_coll_name'] = coll_name
                self.gpu_events.append(ev)
                self.gpu_events_by_stream[stream].append(ev)
            elif ph in ('i', 'C') and ('memory' in cat.lower() or name == "[memory]"):
                self.memory_events.append(ev)
        self.async_resolver.resolve_dependencies(self.all_events)
        return True

    @classmethod
    def _is_gpu_event(cls, cat: str, name: str = "") -> bool:
        """Classify an event as GPU work: known GPU category, or contains "gpu"
        while NOT being a CPU category (``cpu_op``, ``user_annotation``).
        The ``GPU_CATEGORIES`` set lists all known GPU categories from PyTorch's
        Chrome Trace output (kernel, gpu_memcpy, gpu_memset, plus the wrapper
        events gpu_user_annotation and gpu_op that we exclude from attribution).
        """
        return cat in cls.GPU_CATEGORIES or ('gpu' in cat.lower() and cat not in ('cpu_op', 'user_annotation'))

    @classmethod
    def _is_gpu_work_event(cls, ev: dict) -> bool:
        """True for actual GPU work (kernel, memcpy, memset); False for profiler wrapper annotations.
        Prevents double-counting during attribution.
        """
        return ev.get('cat') in cls.GPU_WORK_CATEGORIES

    def build_tree(self) -> List[LogicalOperation]:
        """Build a parent-child tree per (pid, tid) using time-nesting (same logic as Chrome Trace Viewer).
        An event A is a child of B iff A's interval is fully contained within B's.

        Assumes events on the same thread are well-nested.  

        Warns on threads with zero events per (pid, tid) to flag empty trace segments.
        """
        all_roots = []
        for (pid, tid), events in self.cpu_events_by_pid_tid.items():
            if not events:
                warnings.warn(f"No CPU events for (pid={pid}, tid={tid}) — possible trace corruption")
                continue
            events_sorted = sorted(events, key=lambda e: e.get('ts', 0))
            stack = []
            for ev in events_sorted:
                start = ev.get('ts', 0)
                dur = ev.get('dur', 0)
                if dur < 0:
                    warnings.warn(f"Negative duration {dur} for {ev.get('name')} on tid={tid} — clamping to 0")
                    dur = 0
                node = LogicalOperation(
                    name=ev['name'],
                    start_time=start,
                    end_time=start + dur,
                    cpu_duration=dur,
                    raw_event=ev
                )
                args_ev = ev.get('args', {})
                ext_id = args_ev.get('External id') or args_ev.get('external_id')
                if ext_id is not None:
                    node.external_ids.add(int(ext_id) if not isinstance(ext_id, int) else ext_id)

                while stack and stack[-1][1].end_time <= start:
                    stack.pop()
                if stack:
                    stack[-1][1].children.append(node)
                else:
                    all_roots.append(node)
                stack.append((ev, node))
        return all_roots

    def attribute_gpu_kernel_with_logical_operation(self, roots: List[LogicalOperation]):
        """Attribute GPU kernel time to the CPU nodes that launched them, using
        three complementary strategies in priority order:
            1. **External-ID correlation** — match GPU events to CPU events
               by shared ``External id`` (most reliable, set by PyTorch profiler).
            2. **Correlation-ID matching** — for events lacking External ID,
               match via ``args.correlation`` to the ``cudaLaunchKernel``
               CPU event (set by CUDA runtime).  Disambiguates siblings at
               the same tree depth that the time heuristic cannot separate.
            3. **Time-overlap heuristic** — for events with neither External
               ID nor correlation, find the most-recently-started still-active
               CPU span at the kernel's timestamp (max-heap sweep).

        After direct attribution, ``gpu_duration`` is propagated upward so every node
        carries the cumulative GPU time for its subtree.
        """
        if not self.gpu_events:
            warnings.warn("No GPU events to attribute — trace may be CPU-only")
            return

        all_nodes = list(TraceParserHelper.iter_nodes(roots))
        ext_to_gpu = defaultdict(float)
        ext_to_gpu_node = defaultdict(list)

        for gpu_ev in self.gpu_events:
            if not self._is_gpu_work_event(gpu_ev):
                continue
            ext_id = gpu_ev.get('args', {}).get('External id')
            if ext_id is not None:
                ext_to_gpu[ext_id] += gpu_ev.get('dur', 0)
                ext_to_gpu_node[ext_id].append(gpu_ev)

        event_to_node = {id(node.raw_event): node for node in all_nodes if node.raw_event}

        # Build correlation-ID → node map for cudaLaunchKernel events
        corr_to_node = {}
        for node in all_nodes:
            raw = node.raw_event or {}
            corr = raw.get('args', {}).get('correlation')
            if corr is not None:
                corr_to_node.setdefault(corr, []).append(node)
        # Prefer the deepest node for each correlation ID
        def _subtree_size(n):
            return 1 + sum(_subtree_size(c) for c in n.children)
        for corr in corr_to_node:
            if len(corr_to_node[corr]) > 1:
                corr_to_node[corr] = [max(corr_to_node[corr], key=lambda n: _subtree_size(n))]

        all_cpu = [ev for evlist in self.cpu_events_by_pid_tid.values() for ev in evlist]
        all_cpu.sort(key=lambda e: e['ts'])
        gpu_no_extid = [ev for ev in self.gpu_events
                       if ev.get('args', {}).get('External id') is None
                       and self._is_gpu_work_event(ev)]
        gpu_no_extid.sort(key=lambda e: e['ts'])
        heap = []
        cpu_idx = 0
        n_cpu = len(all_cpu)
        for gpu_ev in gpu_no_extid:
            ts = gpu_ev['ts']
            corr = gpu_ev.get('args', {}).get('correlation')
            # Strategy 2: correlation-ID match (deepest node)
            if corr is not None and corr in corr_to_node:
                node = corr_to_node[corr][0]
                node.direct_gpu_duration += gpu_ev.get('dur', 0)
                node.direct_gpu_kernels.append(gpu_ev)
                continue
            # Strategy 3: time-overlap heuristic
            while cpu_idx < n_cpu and all_cpu[cpu_idx]['ts'] <= ts:
                ce = all_cpu[cpu_idx]
                heapq.heappush(heap, (-ce['ts'], ce['ts'] + ce.get('dur', 0), ce))
                cpu_idx += 1
            while heap and heap[0][1] < ts:
                heapq.heappop(heap)
            if heap:
                node = event_to_node.get(id(heap[0][2]))
                if node:
                    node.direct_gpu_duration += gpu_ev.get('dur', 0)
                    node.direct_gpu_kernels.append(gpu_ev)

        # Attribute GPU time to the deepest node for each ext_id (avoid double-counting)
        def subtree_size(n):
            return 1 + sum(subtree_size(c) for c in n.children)
        for node in sorted(all_nodes, key=lambda n: -subtree_size(n)):
            for ext_id in list(node.external_ids):
                if ext_id in ext_to_gpu:
                    node.direct_gpu_duration += ext_to_gpu.pop(ext_id)
                    node.direct_gpu_kernels.extend(ext_to_gpu_node.pop(ext_id))

        # Remaining unclaimed ext_ids: these GPU kernels have no CPU match in the
        # active step's tree (common in multi-step traces where Step#1's GPU kernels
        # still reference Step#1 CPU events that were pruned).  They're silently dropped.
        if ext_to_gpu:
            unclaimed = len(ext_to_gpu)
            total_us = sum(ext_to_gpu.values())
            if unclaimed > 100:
                warnings.warn(f"{unclaimed} ext_ids unclaimed ({total_us:.0f}µs GPU) — likely from pruned steps")

        # Propagate GPU time upward
        def accumulate_gpu(node):
            child_gpu = sum(accumulate_gpu(ch) for ch in node.children)
            node.gpu_duration = node.direct_gpu_duration + child_gpu
            return node.gpu_duration
        for root in roots:
            accumulate_gpu(root)
    
    def attribute_memory(self, roots: List[LogicalOperation]):
        """Assign ``[memory]`` event deltas to the deepest enclosing CPU node
        (same sweep-line max-heap algorithm as GPU attribution), then propagate
        upward.  Only GPU memory (Device Type == 1) is counted.

        Memory profiling is optional and disabled by default — this is a no-op
        when no memory events exist.  Neither the 8B TP trace nor the async TP
        trace has memory events enabled, so this path is largely untested against
        real data.
        """
        if not self.memory_events:
            warnings.warn("No memory events found. Memory profiling may be disabled in the profiler config.")
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
                if device_type == 1:  # GPU memory only
                    try:
                        deepest.memory_delta += int(size)
                    except (TypeError, ValueError):
                        warnings.warn(f"Non-numeric memory size at ts={mem_ev.get('ts')}: {size!r}")
                        continue

        def accumulate_memory(node):
            total = node.memory_delta
            for ch in node.children:
                total += accumulate_memory(ch)
            node.memory_delta = total
            return total
        for root in roots:
            accumulate_memory(root)

class AsyncDependencyResolver:
    """Resolves async GPU-CPU dependencies (cudaEventRecord/cudaStreamWaitEvent + flow events)."""
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
        """Find the async-start event whose interval contains this GPU event's timestamp."""
        ts = gpu_event.get('ts', 0)
        for start_ev, _ in self.dependencies:
            interval = start_ev.get('_async_interval')
            if interval and interval[0] <= ts <= interval[1]:
                return start_ev
        return None

