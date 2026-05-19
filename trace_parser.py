#!/usr/bin/env python3


import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


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
        self.events_by_pid_tid = defaultdict(list)

    def load(self):
        with open(self.trace_file, 'r') as f:
            data = json.load(f)
        for ev in data.get('traceEvents', []):
            if ev.get('ph') == 'X':
                pid, tid = ev.get('pid', 0), ev.get('tid', 0)
                self.events_by_pid_tid[(pid, tid)].append(ev)
    
    def build_tree(self) -> List[LogicalOperation]:
        all_roots = []
        for (pid, tid), events in self.events_by_pid_tid.items():
            events_sorted = sorted(events, key=lambda e: e['ts'])
            stack = []
            for ev in events_sorted:
                node = LogicalOperation(ev['name'], ev['ts'], ev['ts']+ev.get('dur',0), ev.get('dur',0))
                while stack and stack[-1].end_time <= ev['ts']:
                    stack.pop()
                if stack:
                    stack[-1].children.append(node)
                else:
                    all_roots.append(node)
                stack.append(node)
        return all_roots
    