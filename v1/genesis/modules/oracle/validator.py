from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Union

from genesis.models import OracleResult


class OracleValidator:
    def validate_with_synthetic_data(self, oracle_path: Union[str, Path]) -> dict[str, object]:
        result = self.run_oracle(oracle_path, Path(oracle_path).parent)
        return {"passed": isinstance(result, OracleResult) and not result.is_critical_fail}

    def run_oracle(self, oracle_path: Union[str, Path], outputs_dir: Union[str, Path]) -> OracleResult:
        spec = importlib.util.spec_from_file_location("genesis_project_oracle", oracle_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        result = module.run_all_checks(str(outputs_dir))
        return result if isinstance(result, OracleResult) else OracleResult(**result)
