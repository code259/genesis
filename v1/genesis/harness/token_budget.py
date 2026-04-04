from __future__ import annotations


class TokenBudget:
    DEFAULT_ALLOCATIONS = {
        "system_prompt": 2000,
        "always_in_context": 3000,
        "retrieved_history": 8000,
        "domain_knowledge": 2000,
        "instruction_output": 2000,
        "reserve": 1000,
    }

    def allocate(self, model_context_limit: int) -> dict[str, int]:
        if model_context_limit <= 20000:
            return dict(self.DEFAULT_ALLOCATIONS)
        allocations = dict(self.DEFAULT_ALLOCATIONS)
        non_history_total = sum(
            value for key, value in allocations.items() if key != "retrieved_history"
        )
        target_history = 20000 if model_context_limit >= 100000 else min(12000, model_context_limit - non_history_total)
        allocations["retrieved_history"] = max(allocations["retrieved_history"], target_history)
        total = sum(allocations.values())
        if total > model_context_limit:
            overflow = total - model_context_limit
            allocations["retrieved_history"] = max(1000, allocations["retrieved_history"] - overflow)
        return allocations

    def check_overflow(self, content: str, layer_budget: int) -> bool:
        return self._estimate_tokens(content) > layer_budget

    def trim_to_budget(self, content: str, layer_budget: int) -> str:
        words = content.split()
        while words and self._estimate_tokens(" ".join(words)) > layer_budget:
            words = words[:- max(1, len(words) // 10)]
        return " ".join(words)

    def _estimate_tokens(self, content: str) -> int:
        return max(1, len(content) // 4)
