from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Union

from genesis.models import OracleResult


class OracleValidator:
    def validate_with_synthetic_data(self, oracle_path: Union[str, Path]) -> dict[str, object]:
        synthetic_dir = Path(oracle_path).parent / "_synthetic_validation"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        (synthetic_dir / "result.json").write_text(
            json.dumps({"primary_metric": 0.7, "secondary_metric": 0.3}, indent=2),
            encoding="utf-8",
        )
        (synthetic_dir / "notes.md").write_text("primary_metric=0.7\nsecondary_metric=0.3\n", encoding="utf-8")
        result = self.run_oracle(oracle_path, synthetic_dir)
        return {
            "name": "synthetic_oracle_validation",
            "passed": not result.is_critical_fail and 0.0 <= result.pass_rate <= 1.0,
            "result": result.to_dict(),
        }

    def run_oracle(self, oracle_path: Union[str, Path], outputs_dir: Union[str, Path]) -> OracleResult:
        module = self._load_module(oracle_path)
        if module is None:
            return OracleResult(pass_rate=0.0, failures=["oracle_load_failed"], warnings=[], is_critical_fail=True)
        run_all_checks = getattr(module, "run_all_checks", None)
        if not callable(run_all_checks):
            return OracleResult(pass_rate=0.0, failures=["run_all_checks_missing"], warnings=[], is_critical_fail=True)
        try:
            result = run_all_checks(str(outputs_dir))
        except Exception as exc:
            return OracleResult(pass_rate=0.0, failures=[f"oracle_runtime_error::{exc}"], warnings=[], is_critical_fail=True)
        return self._coerce_result(result)

    def _load_module(self, oracle_path: Union[str, Path]):
        spec = importlib.util.spec_from_file_location("genesis_project_oracle", oracle_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _coerce_result(self, result: Any) -> OracleResult:
        if isinstance(result, OracleResult):
            return result
        if not isinstance(result, dict):
            return OracleResult(pass_rate=0.0, failures=["oracle_result_not_dict"], warnings=[], is_critical_fail=True)
        return OracleResult(
            pass_rate=float(result.get("pass_rate", 0.0)),
            failures=[str(item) for item in result.get("failures", [])],
            warnings=[str(item) for item in result.get("warnings", [])],
            is_critical_fail=bool(result.get("is_critical_fail", False)),
        )
