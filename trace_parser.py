"""
trace_parser.py
A simple parser for PyTorch trace files.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import defaultdict

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RawEvent:
    name: str
    cat: str
    ts: float       # start timestamp (µs)
    dur: float      # duration (µs)
    tid: int
    pid: int
    args: dict

    @property
    def end(self) -> float:
        return self.ts + self.dur

    @classmethod
    def from_dict(cls, d: dict) -> Optional["RawEvent"]:
        # (For now) we are only considering complete events (“X”)
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
    """A logical (high-level) operation, from nested raw events."""
    name: str
    ts: float
    dur: float
    input_shapes: List[str]
    memory_bytes: int

    # Hierarchy
    parent: Optional["LogicalOp"] = field(default=None, repr=False)
    children: List["LogicalOp"] = field(default_factory=list, repr=False)

    # Composition (GPU-Kernels that are part of this logical op)
    gpu_kernels: List[RawEvent] = field(default_factory=list, repr=False)

    @property
    def end(self) -> float:
        return self.ts + self.dur

    @property
    def gpu_time_us(self) -> float:
        return sum(k.dur for k in self.gpu_kernels)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TraceParser:
    """Parse a PyTorch trace file and build a tree of logical operations."""
    CPU_CATS = {"cpu_op", "python_function"}
    GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

    def __init__(self, path: str):
        """Initialize the parser with the path to the trace file."""
        self.path = path
        self.raw_events: List[RawEvent] = []
        self.roots: List[LogicalOp] = []
        self._all_ops: List[LogicalOp] = []

    def load(self) -> "TraceParser":
        """Load the trace file and extract raw events."""
        with open(self.path) as f:
            data = json.load(f)

        raw = data.get("traceEvents", data) if isinstance(data, dict) else data

        for d in raw:
            ev = RawEvent.from_dict(d)
            if ev:
                self.raw_events.append(ev)
        
        print(f"Loaded {len(self.raw_events)} events.")
        return self

    def parse(self) -> "TraceParser":
        """Extract the logical operation tree from the raw events."""
        # 1. Filter for CPU and GPU events
        cpu_events = [e for e in self.raw_events if e.cat in self.CPU_CATS]
        gpu_events = [e for e in self.raw_events if e.cat in self.GPU_CATS]

        # 2. Group by thread in order to avoid interleaving issues
        by_thread: Dict[int, List[RawEvent]] = defaultdict(list)
        for ev in cpu_events:
            by_thread[ev.tid].append(ev)

        # 3. Build trees per thread
        for tid, events in by_thread.items():
            self.roots.extend(self._build_tree(events))

        # 4. Collect all operations for kernel attachment
        self._all_ops = []
        for root in self.roots:
            self._dfs_collect(root)

        # 5. Attach GPU kernels to the innermost (most specific) enclosing cpu_op
        self._attach_kernels(gpu_events)

        # Debug-output: number of parsed logical operations
        print(f"Parsed {len(self._all_ops)} logical operations.")
        return self

    def _build_tree(self, events: List[RawEvent]) -> List[LogicalOp]:
        """Build a tree of logical operations from a list of raw events."""
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

    def _dfs_collect(self, op: LogicalOp):
        """Depth-first search to collect all operations."""
        self._all_ops.append(op)
        for child in op.children:
            self._dfs_collect(child)

    def _attach_kernels(self, gpu_events: List[RawEvent]):
        """Attach GPU kernels to the innermost (most specific) enclosing cpu_op."""
        # Sort operations by duration (most specific first)
        ops_by_specificity = sorted(self._all_ops, key=lambda o: o.dur)

        for kernel in gpu_events:
            for op in ops_by_specificity:
                if op.ts <= kernel.ts and kernel.end <= op.end:
                    op.gpu_kernels.append(kernel)
                    break 

if __name__ == "__main__":
    # Test-code: Load and parse a trace file, then print some debug info about the first few operations
    import sys
    if len(sys.argv) > 1:
        tp = TraceParser(sys.argv[1]).load().parse()
        # Print the first few operations with their GPU time
        for op in tp.roots[:5]:
            print(f"Op: {op.name}, Wall: {op.dur}us, GPU-Time: {op.gpu_time_us}us")