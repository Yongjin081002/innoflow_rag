from fastapi import APIRouter, HTTPException

from application.dto.search import SearchRequest, SearchResponse
from application.usecases.search_fault_rules import SearchFaultRulesUseCase

router = APIRouter()
_search_usecase: SearchFaultRulesUseCase | None = None


def set_search_usecase(usecase: SearchFaultRulesUseCase | None) -> None:
    global _search_usecase
    _search_usecase = usecase


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if _search_usecase is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")
    return SearchResponse(results=_search_usecase.execute(req.query, top_k=req.top_k))
