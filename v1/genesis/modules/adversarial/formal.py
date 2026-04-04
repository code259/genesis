from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Union

from genesis.models import CheckResult, OracleResult


class FormalConsistencyChecker:
    def check_parameter_count(self, code_path: Union[str, Path], claimed_params: int) -> CheckResult:
        text = Path(code_path).read_text(encoding="utf-8") if Path(code_path).exists() else ""
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

    def check_implementation_drift(self, prose_methods: str, code_path: Union[str, Path]) -> CheckResult:
        code = Path(code_path).read_text(encoding="utf-8") if Path(code_path).exists() else ""
        overlap = len(set(prose_methods.lower().split()) & set(code.lower().split()))
        passed = overlap > 3
        return CheckResult(
            name="implementation_drift",
            passed=passed,
            evidence=[f"token_overlap={overlap}"],
            score=float(overlap),
        )

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
