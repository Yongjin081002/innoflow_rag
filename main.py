from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.v1.fault_rules import router as fault_rules_router
from api.v1.fault_rules import set_search_usecase
from application.usecases.search_fault_rules import SearchFaultRulesUseCase
from infrastructure.rag.fault_rule_searcher import FaultRuleSearcher

_searcher: FaultRuleSearcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher
    _searcher = FaultRuleSearcher()
    set_search_usecase(SearchFaultRulesUseCase(_searcher))
    yield
    set_search_usecase(None)
    if _searcher:
        _searcher.close()


app = FastAPI(title="RAG 과실기준 검색 API", version="1.0.0", lifespan=lifespan)
app.include_router(fault_rules_router)
