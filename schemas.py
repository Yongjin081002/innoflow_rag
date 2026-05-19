from typing import Any
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class FaultRule(BaseModel):
    type: str
    id: str
    content: str
    base_fault: dict[str, Any]
    modifiers: list[dict[str, Any]]
    category: str
    source: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[FaultRule]
