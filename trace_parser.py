
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

    @property
    def total_time(self) -> float:
        return max(self.cpu_duration, self.gpu_duration)