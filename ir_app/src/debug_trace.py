from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceStep:
    name: str
    payload: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    steps: List[TraceStep] = field(default_factory=list)

    def add(self, name: str, payload: Any = None, **meta: Any) -> None:
        self.steps.append(TraceStep(name=name, payload=payload, meta=meta))

    def to_dict(self) -> List[Dict[str, Any]]:
        return [{"name": s.name, "payload": s.payload, "meta": s.meta} for s in self.steps]


def as_trace(trace: Optional[Trace]) -> Trace:
    return trace if trace is not None else Trace()


def preview_list(xs: List[Any], n: int = 30) -> Dict[str, Any]:
    return {"count": len(xs), "head": xs[:n]}


def preview_dict(d: Dict[Any, Any], n: int = 30) -> Dict[str, Any]:
    items = list(d.items())[:n]
    return {"count": len(d), "head": items}