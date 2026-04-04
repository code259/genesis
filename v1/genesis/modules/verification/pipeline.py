from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from genesis.modules.citations.agent import CitationsAgent
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
            result_payload = self._load_result(results_file)
            report["checks"].append({"name": "artifact_exists", "passed": True})
            report["checks"].append(
                self.formal.check_implementation_drift(results_file.read_text(encoding="utf-8"), results_file).to_dict()
            )
            report["checks"].append(self._metric_consistency_check(result_payload))
        else:
            report["checks"].append({"name": "artifact_exists", "passed": False})
        if oracle_path:
            report["checks"].append(self.oracle_validator.run_oracle(oracle_path, outputs_dir).to_dict())
        citation_flags = self._citation_check(outputs_dir)
        if citation_flags:
            report["checks"].append({"name": "citation_verification", "passed": False, "flags": citation_flags})
        else:
            report["checks"].append({"name": "citation_verification", "passed": True})
        report["passed"] = all(check.get("passed", True) or check.get("pass_rate", 0.0) > 0.0 for check in report["checks"])
        return report

    def _load_result(self, path: Path) -> dict[str, Any]:
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def _metric_consistency_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        selected = payload.get("selected_experiment")
        if not isinstance(selected, dict):
            return {"name": "metric_consistency", "passed": True}
        primary_metric = float(payload.get("primary_metric", 0.0))
        selected_metric = float(selected.get("primary_metric", primary_metric))
        passed = abs(primary_metric - selected_metric) < 1e-6
        return {
            "name": "metric_consistency",
            "passed": passed,
            "evidence": [f"primary_metric={primary_metric}", f"selected_metric={selected_metric}"],
        }

    def _citation_check(self, outputs_dir: Path) -> list[dict[str, Any]]:
        paper_dir = outputs_dir.parent / "paper"
        tex_path = paper_dir / "main.tex"
        bib_path = paper_dir / "references.bib"
        if not tex_path.exists() or not bib_path.exists():
            return []
        agent = CitationsAgent(outputs_dir.parent.parent / "knowledge" / "citations_cache.json")
        return agent.verify_all_in_latex(
            tex_path.read_text(encoding="utf-8"),
            bib_path.read_text(encoding="utf-8"),
        )
