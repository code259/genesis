from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from genesis.modules.adversarial.formal import FormalConsistencyChecker
from genesis.modules.oracle.validator import OracleValidator


class VerificationPipeline:
    def __init__(self) -> None:
        self.formal = FormalConsistencyChecker()
        self.oracle_validator = OracleValidator()

    def run(
        self,
        outputs_dir: Union[str, Path],
        project_id: str,
        oracle_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        outputs_dir = Path(outputs_dir)
        report: dict[str, Any] = {
            "project_id": project_id,
            "outputs_dir": str(outputs_dir),
            "checks": [],
        }
        results_file = outputs_dir / "result.json"
        if results_file.exists():
            report["checks"].append({"name": "artifact_exists", "passed": True})
        if oracle_path:
            report["checks"].append(self.oracle_validator.run_oracle(oracle_path, outputs_dir).to_dict())
        return report
