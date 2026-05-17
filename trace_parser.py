import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

@dataclass
class LogicalOperation:
    name: str
    start_time: float
    end_time: float
    cpu_duration: float
    gpu_duration: float = 0.0
    children: List['LogicalOperation'] = field(default_factory=list)

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