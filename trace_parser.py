"""
trace_parser.py
Parses a PyTorch profiler trace (Chrome trace format JSON) and:
  1. Reconstructs logical operations from raw events (via ts/dur nesting)
  2. Extracts runtime, memory, and composition per logical operation

Usage:
    python trace_parser.py sample_trace.json             
    python trace_parser.py sample_trace.json --json     # also writes sample_trace.summary.json
    python trace_parser.py sample_trace.json --depth x  # limit tree printout to depth x (default 3)
"""

from __future__ import annotations

import json
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RawEvent:
    name: str       # event name, e.g. "aten::matmul"
    cat: str        # category, e.g. "cpu_op", "kernel", "gpu_memcpy"
    ts: float       # start timestamp (µs)
    dur: float      # duration (µs)
    tid: int        # thread ID
    pid: int        # process ID
    args: dict      # additional info, e.g. input shapes, memory allocated, etc. 

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
    """A high-level operation reconstructed from nested raw events."""
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
        """End time of this operation (start + duration)."""
        return self.ts + self.dur

    @property
    def self_time_us(self) -> float:
        """Wall time not covered by any direct child."""
        child_time = sum(c.dur for c in self.children)
        return max(0.0, self.dur - child_time)

    @property
    def gpu_time_us(self) -> float:
        """Time spent on GPU operations."""
        return sum(k.dur for k in self.gpu_kernels)

    @property
    def depth(self) -> int:
        """Depth of this node in the operation tree (root==0)."""
        d = 0
        node = self
        while node.parent:
            d += 1
            node = node.parent
        return d

    def is_root(self) -> bool:
        return self.parent is None

    def subtree_memory(self) -> int:
        """Total memory allocated in this op and all its children."""
        return self.memory_bytes + sum(c.subtree_memory() for c in self.children)


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
            if ev is not None:
                self.raw_events.append(ev)

        print(f"Loaded {len(self.raw_events)} raw events from '{self.path}'")
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
        per_thread_roots: List[LogicalOp] = []
        for tid, events in by_thread.items():
            roots = self._build_tree(events)
            per_thread_roots.extend(roots)

        # 4. Collect all operations for kernel attachment
        self.roots = per_thread_roots
        self._all_ops = []
        for root in self.roots:
            self._dfs_collect(root)

        # 5. Attach GPU kernels to the innermost (most specific) enclosing cpu_op
        self._attach_kernels(gpu_events)

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
        # Sort operation by depth (most specific first)
        ops_by_depth = sorted(self._all_ops, key=lambda o: -o.depth)

        for kernel in gpu_events:
            for op in ops_by_depth:
                if op.ts <= kernel.ts and kernel.end <= op.end:
                    op.gpu_kernels.append(kernel)
                    break 

    # ------------------------------------------------------------------
    #  Pretty Printing & CLI FEATURES
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a structured summary of all metrics."""
        result = {"ops": [], "bottlenecks": []}

        # Recursively summarize an op and its subtree
        for root in self.roots:
            result["ops"].append(self._op_summary(root))

        # Bottleneck-analysis: Find the 5 ops with the worst GPU utilization (lowest %)
        # that could indicate CPU overhead or memory bottlenecks
        all_summaries = [s for s in result["ops"]] # simplified: only top-level ops
        all_summaries.sort(key=lambda s: s["gpu_utilization_pct"])
        result["bottlenecks"] = [
            {"name": s["name"], "gpu_utilization_pct": s["gpu_utilization_pct"],
             "wall_time_us": s["wall_time_us"]}
            for s in all_summaries[:5]   
        ]
        return result

    def _op_summary(self, op: LogicalOp) -> Dict:
        """Help-method for creating a metrics dictonary for each operation."""
        gpu_util = (op.gpu_time_us / op.dur * 100) if op.dur > 0 else 0
        return {
            "name": op.name,
            "wall_time_us": round(op.dur, 2),
            "self_time_us": round(op.self_time_us, 2),
            "gpu_time_us": round(op.gpu_time_us, 2),
            "gpu_utilization_pct": round(gpu_util, 1),
            "memory_bytes": op.memory_bytes,
            "subtree_memory_bytes": op.subtree_memory(),
            "num_gpu_kernels": len(op.gpu_kernels),
            "children": [self._op_summary(c) for c in op.children],
        }

    def print_tree(self, max_depth: int = 3):
        """Pretty-print the logical operation tree to the console."""
        print("\n" + "=" * 70)
        print("LOGICAL OPERATION TREE")
        print("=" * 70)
        for root in self.roots:
            self._print_node(root, depth=0, max_depth=max_depth)

    def _print_node(self, op: LogicalOp, depth: int, max_depth: int):
        """Helper method to print a single node and its subtree."""
        if depth > max_depth: return
        indent = "  " * depth
        marker = "▶" if not op.children else "◆"
        gpu_info = (f" | GPU {op.gpu_time_us:.0f}µs ({op.gpu_time_us/op.dur*100:.0f}%)" if op.dur > 0 else "")
        mem_info = (f" | mem {op.memory_bytes/1024:.1f}KB" if op.memory_bytes > 0 else "")
        print(f"{indent}{marker} {op.name:<40} wall={op.dur:>8.1f}µs{gpu_info}{mem_info}")
        for child in op.children:
            self._print_node(child, depth + 1, max_depth)

    def print_summary(self):
        """Print a summary of all operations and bottlenecks."""
        s = self.summary()
        print("\n" + "=" * 70)
        print("PER-OP SUMMARY  (top-level ops)")
        print("=" * 70)
        print(f"{'Op':<35} {'Wall(µs)':>10} {'GPU(µs)':>10} {'GPU%':>6} {'Mem(KB)':>10} {'Kernels':>8}")
        print("-" * 70)
        for op in s["ops"]:
            print(f"{op['name']:<35} {op['wall_time_us']:>10.1f} {op['gpu_time_us']:>10.1f} "
                  f"{op['gpu_utilization_pct']:>5.1f}% {op['subtree_memory_bytes']/1024:>10.1f} {op['num_gpu_kernels']:>8}")

        print("\n" + "=" * 70)
        print("BOTTLENECKS  (lowest GPU utilisation)")
        print("=" * 70)
        for b in s["bottlenecks"]:
            print(f"  {b['name']:<40}  GPU util: {b['gpu_utilization_pct']:>5.1f}%  wall: {b['wall_time_us']:.1f}µs")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Parse a PyTorch profiler trace.")
    parser.add_argument("trace", help="Path to the trace JSON file")
    parser.add_argument("--depth", type=int, default=3, help="Max depth for tree printout")
    parser.add_argument("--json", action="store_true", help="Dump summary to JSON")
    args = parser.parse_args()

    tp = TraceParser(args.trace).load().parse()
    tp.print_tree(max_depth=args.depth)
    tp.print_summary()

    if args.json:
        out_path = args.trace.replace(".json", ".summary.json")
        with open(out_path, "w") as f:
            json.dump(tp.summary(), f, indent=2)
        print(f"\nSummary written to {out_path}")

if __name__ == "__main__":
    # Test-code: Load and parse a trace file
    main()