from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Union

from genesis.models import OracleResult


class OracleValidator:
    def validate_with_synthetic_data(self, oracle_path: Union[str, Path]) -> dict[str, object]:
        synthetic_dir = Path(oracle_path).parent / "_synthetic_validation"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        (synthetic_dir / "result.json").write_text(
            json.dumps({"primary_metric": 0.7, "secondary_metric": 0.3}, indent=2),
            encoding="utf-8",
        )
        result = self.run_oracle(oracle_path, synthetic_dir)
        return {
            "passed": isinstance(result, OracleResult) and not result.is_critical_fail,
            "result": result.to_dict() if isinstance(result, OracleResult) else {},
        }

    def run_oracle(self, oracle_path: Union[str, Path], outputs_dir: Union[str, Path]) -> OracleResult:
        spec = importlib.util.spec_from_file_location("genesis_project_oracle", oracle_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        result = module.run_all_checks(str(outputs_dir))
        return result if isinstance(result, OracleResult) else OracleResult(**result)
