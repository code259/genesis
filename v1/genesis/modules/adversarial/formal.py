from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Union

from genesis.models import CheckResult, OracleResult


class FormalConsistencyChecker:
    def check_parameter_count(self, code_path: Union[str, Path], claimed_params: int) -> CheckResult:
        payload_path = Path(code_path)
        observed = 0
        if payload_path.exists():
            if payload_path.suffix == ".json":
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                config = payload.get("config", {}) if isinstance(payload, dict) else {}
                observed = int(config.get("model_parameter_count", 0))
            else:
                text = payload_path.read_text(encoding="utf-8")
                observed = text.count("def ") + text.count("class ")
        passed = observed <= claimed_params if claimed_params > 0 else True
        return CheckResult(
            name="parameter_count",
            passed=passed,
            evidence=[f"observed={observed}", f"claimed={claimed_params}"],
            score=float(observed),
        )

    def check_metric_plausibility(self, claimed_metric: float, theoretical_bounds: tuple[float, float]) -> CheckResult:
        lower, upper = theoretical_bounds
        passed = lower <= claimed_metric <= upper
        return CheckResult(
            name="metric_plausibility",
            passed=passed,
            evidence=[f"metric={claimed_metric}", f"bounds={theoretical_bounds}"],
            score=claimed_metric,
        )

    def check_implementation_drift(self, prose_methods: str, code_path: Union[str, Path, list[Union[str, Path]]]) -> CheckResult:
        code = self._load_code_text(code_path)
        prose_tokens = {token for token in prose_methods.lower().split() if len(token) > 3}
        code_tokens = {token for token in code.lower().split() if len(token) > 3}
        overlap = len(prose_tokens & code_tokens)
        token_budget = max(1, min(len(prose_tokens), len(code_tokens)))
        overlap_ratio = overlap / token_budget
        passed = overlap_ratio >= 0.1
        return CheckResult(
            name="implementation_drift",
            passed=passed,
            evidence=[f"token_overlap={overlap}", f"overlap_ratio={overlap_ratio:.3f}"],
            score=float(overlap_ratio),
        )

    def _load_code_text(self, code_path: Union[str, Path, list[Union[str, Path]]]) -> str:
        if isinstance(code_path, list):
            return "\n".join(self._load_code_text(item) for item in code_path)
        path = Path(code_path)
        if not path.exists():
            return ""
        if path.is_dir():
            return "\n".join(
                candidate.read_text(encoding="utf-8", errors="ignore")
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix in {".py", ".md", ".txt", ".json", ".tex", ".yaml", ".yml"}
            )
        return path.read_text(encoding="utf-8", errors="ignore")

    def run_oracle(self, oracle_path: Union[str, Path], outputs_dir: Union[str, Path]) -> OracleResult:
        module_path = Path(oracle_path)
        if not module_path.exists():
            return OracleResult(pass_rate=0.0, failures=["oracle missing"], is_critical_fail=True)
        spec = importlib.util.spec_from_file_location("project_oracle", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        result = module.run_all_checks(str(outputs_dir))
        if isinstance(result, OracleResult):
            return result
        return OracleResult(**result)
