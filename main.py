from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.v1.fault_rules import router as fault_rules_router
from api.v1.fault_rules import set_search_usecase
from api.v1.admin import router as admin_router
from api.v1.admin import set_reload_callback
from application.usecases.search_fault_rules import SearchFaultRulesUseCase
from infrastructure.rag.fault_rule_searcher import FaultRuleSearcher

_searcher: FaultRuleSearcher | None = None


def _reload_searcher() -> None:
    global _searcher
    if _searcher:
        _searcher.close()
    _searcher = FaultRuleSearcher()
    set_search_usecase(SearchFaultRulesUseCase(_searcher))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher
    _searcher = FaultRuleSearcher()
    set_search_usecase(SearchFaultRulesUseCase(_searcher))
    set_reload_callback(_reload_searcher)
    yield
    set_search_usecase(None)
    set_reload_callback(None)
    if _searcher:
        _searcher.close()


app = FastAPI(title="RAG 과실기준 검색 API", version="1.0.0", lifespan=lifespan)
app.include_router(fault_rules_router)
app.include_router(admin_router)
