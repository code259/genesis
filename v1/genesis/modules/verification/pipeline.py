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
            report["checks"].append(
                self.formal.check_implementation_drift(results_file.read_text(encoding="utf-8"), results_file).to_dict()
            )
        else:
            report["checks"].append({"name": "artifact_exists", "passed": False})
        if oracle_path:
            report["checks"].append(self.oracle_validator.run_oracle(oracle_path, outputs_dir).to_dict())
        report["passed"] = all(check.get("passed", True) or check.get("pass_rate", 0.0) > 0.0 for check in report["checks"])
        return report
