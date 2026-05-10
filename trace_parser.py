import json
from dataclasses import dataclass
from typing import List, Optional

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
            name=d.get("name", ""), cat=d.get("cat", ""),
            ts=float(d.get("ts", 0)), dur=float(d.get("dur", 0)),
            tid=d.get("tid", 0), pid=d.get("pid", 0), args=d.get("args", {}),
        )

class TraceParser:
    def __init__(self, path: str):
        self.path = path
        self.raw_events: List[RawEvent] = []

    def load(self):
        with open(self.path) as f:
            data = json.load(f)
        raw = data.get("traceEvents", data) if isinstance(data, dict) else data
        for d in raw:
            ev = RawEvent.from_dict(d)
            if ev: self.raw_events.append(ev)
        return self