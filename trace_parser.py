from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import defaultdict

@dataclass
class RawEvent:
    name: str
    cat: str
    ts: float
    dur: float
    tid: int
    pid: int
    args: dict

    @property
    def end(self) -> float:
        return self.ts + self.dur

    @classmethod
    def from_dict(cls, d: dict) -> Optional["RawEvent"]:
        if d.get("ph") != "X":
            return None
        return cls(
            name=d.get("name", ""),
            cat=d.get("cat", ""),
            ts=float(d.get("ts", 0)),
            dur=float(d.get("dur", 0)),
            tid=d.get("tid", 0),
            pid=d.get("pid", 0),
            args=d.get("args", {}),
        )

@dataclass
class LogicalOp:
    """a logical (high-level) operation, from nested raw events."""
    name: str
    ts: float
    dur: float
    input_shapes: List[str]
    memory_bytes: int

    parent: Optional["LogicalOp"] = field(default=None, repr=False)
    children: List["LogicalOp"] = field(default_factory=list, repr=False)
    gpu_kernels: List[RawEvent] = field(default_factory=list, repr=False)

    @property
    def end(self) -> float:
        return self.ts + self.dur

class TraceParser:
    """ Parse a PyTorch trace file and build a tree of logical operations"""
    CPU_CATS = {"cpu_op", "python_function"}

    def __init__(self, path: str):
        """Initialize the parser wih the path to the trace file."""
        self.path = path
        self.raw_events: List[RawEvent] = []
        self.roots: List[LogicalOp] = []

    def load(self) -> "TraceParser":
        """Load the trace file and extract raw events"""
        with open(self.path) as f:
            data = json.load(f)
        raw = data.get("traceEvents", data) if isinstance(data, dict) else data
        for d in raw:
            ev = RawEvent.from_dict(d)
            if ev: self.raw_events.append(ev)
        return self

    def parse(self) -> "TraceParser":
        """Extract the logical operation tree from the raw events"""
        # Filter for logical CPU operations
        cpu_events = [e for e in self.raw_events if e.cat in self.CPU_CATS]

        # Group by thread to avoid interleaving issues
        by_thread: Dict[int, List[RawEvent]] = defaultdict(list)
        for ev in cpu_events:
            by_thread[ev.tid].append(ev)

        # Build trees per thread
        for tid, events in by_thread.items():
            self.roots.extend(self._build_tree(events))
        
        return self

    def _build_tree(self, events: List[RawEvent]) -> List[LogicalOp]:
        """Build a tree of logical operations from a list of raw events"""

        # Sort: earlier start first; for ties, longer duration first
        events = sorted(events, key=lambda e: (e.ts, -e.dur))

        stack: List[LogicalOp] = []
        roots: List[LogicalOp] = []

        for ev in events:
            op = LogicalOp(
                name=ev.name,
                ts=ev.ts,
                dur=ev.dur,
                input_shapes=ev.args.get("Input Dims", []),
                memory_bytes=ev.args.get("Memory bytes allocated", 0),
            )

            # Pop stack entries that have ended before this event starts
            while stack and stack[-1].end <= ev.ts:
                stack.pop()

            if stack:
                op.parent = stack[-1]
                stack[-1].children.append(op)
            else:
                roots.append(op)

            stack.append(op)
        return roots