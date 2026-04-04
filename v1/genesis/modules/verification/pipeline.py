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
            oracle_result = self.oracle_validator.run_oracle(oracle_path, outputs_dir).to_dict()
            oracle_result["name"] = "oracle_execution"
            oracle_result["passed"] = not oracle_result.get("is_critical_fail", False)
            report["checks"].append(oracle_result)
            synthetic_result = self.oracle_validator.validate_with_synthetic_data(oracle_path)
            synthetic_result["name"] = "oracle_synthetic_validation"
            report["checks"].append(synthetic_result)
        citation_flags = self._citation_check(outputs_dir)
        if citation_flags:
            report["checks"].append({"name": "citation_verification", "passed": False, "flags": citation_flags})
        else:
            report["checks"].append({"name": "citation_verification", "passed": True})
        paper_artifact_check = self._paper_artifact_check(outputs_dir)
        if paper_artifact_check is not None:
            report["checks"].append(paper_artifact_check)
        report["passed"] = all(self._is_check_passing(check) for check in report["checks"])
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

    def _paper_artifact_check(self, outputs_dir: Path) -> dict[str, Any] | None:
        paper_dir = outputs_dir.parent / "paper"
        if not paper_dir.exists():
            return None
        required = {
            "latex": paper_dir / "main.tex",
            "report": paper_dir / "synthesis_report.json",
        }
        evidence = [f"{name}={path.exists()}" for name, path in required.items()]
        if (paper_dir / "main.pdf").exists():
            evidence.append("pdf=True")
        if (paper_dir / "figures").exists():
            metadata_files = list((paper_dir / "figures").glob("**/*.metadata.json"))
            evidence.append(f"figure_metadata_count={len(metadata_files)}")
        return {
            "name": "paper_artifacts",
            "passed": all(path.exists() for path in required.values()),
            "evidence": evidence,
        }

    def _is_check_passing(self, check: dict[str, Any]) -> bool:
        if "passed" in check:
            return bool(check["passed"])
        if "pass_rate" in check:
            return float(check["pass_rate"]) > 0.0
        return True
