from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from schemas import SearchRequest, SearchResponse
from search_final import FaultRuleSearcher

_searcher: FaultRuleSearcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher
    _searcher = FaultRuleSearcher()
    yield
    if _searcher:
        _searcher.close()


app = FastAPI(title="RAG 과실기준 검색 API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if _searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")
    results = _searcher.search(req.query, top_k=req.top_k)
    return SearchResponse(results=results)
