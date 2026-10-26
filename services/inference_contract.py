from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InferenceRequest:
    task: str
    model: str
    image_paths: List[str] = field(default_factory=list)
    question: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    status: str
    task: str
    model: str
    answer: Optional[str] = None
    confidence: float = 0.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None