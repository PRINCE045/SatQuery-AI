from dataclasses import dataclass, field
from typing import List


@dataclass
class QueryPlan:
    task: str
    description: str
    required_inputs: List[str] = field(default_factory=list)
    selected_tool: str = ""
    confidence: float = 0.0