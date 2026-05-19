from domain.entities.fault_rule import FaultRuleResult
from infrastructure.rag.fault_rule_searcher import FaultRuleSearcher


class SearchFaultRulesUseCase:
    def __init__(self, searcher: FaultRuleSearcher):
        self.searcher = searcher

    def execute(self, query: str, top_k: int = 5) -> list[FaultRuleResult]:
        return self.searcher.search(query, top_k=top_k)
