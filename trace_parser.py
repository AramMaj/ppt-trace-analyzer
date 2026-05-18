import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

@dataclass
class LogicalOperation:
    name: str
    start_time: float
    end_time: float
    cpu_duration: float
    gpu_duration: float = 0.0
    children: List['LogicalOperation'] = field(default_factory=list)

# Helper function: iterate through the tree of logical operations
def iter_nodes(roots):
    stack = list(roots)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)

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
    