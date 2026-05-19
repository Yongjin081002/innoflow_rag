from typing import Any, TypedDict


class FaultRuleResult(TypedDict):
    type: str
    id: str
    content: str
    base_fault: dict[str, Any]
    modifiers: list[dict[str, Any]]
    category: str
    source: str
    score: float
    metadata: dict[str, Any]
