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
        task_kind: str = "",
        expected_artifacts: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        outputs_dir = Path(outputs_dir)
        report: dict[str, Any] = {
            "project_id": project_id,
            "outputs_dir": str(outputs_dir),
            "task_kind": task_kind,
            "checks": [],
        }
        results_file = self._result_path(outputs_dir)
        if results_file.exists():
            result_payload = self._load_result(results_file)
            report["checks"].append({"name": "artifact_exists", "passed": True})
            report["checks"].append(self._expected_artifact_check(result_payload, expected_artifacts or []))
            report["checks"].append(self._substantive_artifact_check(result_payload))
            if task_kind not in {"survey", "paper"}:
                drift_targets = self._implementation_targets(outputs_dir, result_payload)
                report["checks"].append(
                    self.formal.check_implementation_drift(
                        str(result_payload.get("summary", "")),
                        drift_targets,
                    ).to_dict()
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
        if task_kind in {"", "survey", "paper"}:
            if citation_flags:
                report["checks"].append({"name": "citation_verification", "passed": False, "flags": citation_flags})
            else:
                report["checks"].append({"name": "citation_verification", "passed": True})
        if task_kind in {"", "paper"}:
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

    def _substantive_artifact_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        generated_artifacts = payload.get("generated_artifacts", [])
        artifact_records = payload.get("artifact_records", [])
        executed_commands = payload.get("executed_commands", [])
        command_results = payload.get("command_results", [])
        selected_experiment = payload.get("selected_experiment")
        code_path = str(payload.get("code_path", "")).strip()
        has_generated_artifacts = isinstance(generated_artifacts, list) and bool(generated_artifacts)
        has_nonempty_artifacts = (
            isinstance(artifact_records, list)
            and any(
                isinstance(item, dict) and bool(item.get("substantive")) and int(item.get("size_bytes", 0)) > 0
                for item in artifact_records
            )
        )
        has_successful_commands = (
            isinstance(command_results, list)
            and any(
                int(item.get("returncode", 1)) == 0 and not self._is_setup_command(str(item.get("command", "")))
                for item in command_results
                if isinstance(item, dict)
            )
        )
        has_executed_commands = isinstance(executed_commands, list) and bool(executed_commands)
        has_selected_experiment = isinstance(selected_experiment, dict) and bool(selected_experiment)
        has_code_path = bool(code_path)
        passed = has_nonempty_artifacts or has_successful_commands or has_selected_experiment or (has_code_path and has_generated_artifacts)
        return {
            "name": "substantive_artifacts",
            "passed": passed,
            "evidence": [
                f"generated_artifacts={len(generated_artifacts) if isinstance(generated_artifacts, list) else 0}",
                f"nonempty_artifacts={sum(1 for item in artifact_records if isinstance(item, dict) and bool(item.get('substantive')) and int(item.get('size_bytes', 0)) > 0) if isinstance(artifact_records, list) else 0}",
                f"executed_commands={len(executed_commands) if isinstance(executed_commands, list) else 0}",
                f"successful_commands={sum(1 for item in command_results if isinstance(item, dict) and int(item.get('returncode', 1)) == 0) if isinstance(command_results, list) else 0}",
                f"task_relevant_successful_commands={sum(1 for item in command_results if isinstance(item, dict) and int(item.get('returncode', 1)) == 0 and not self._is_setup_command(str(item.get('command', '')))) if isinstance(command_results, list) else 0}",
                f"has_selected_experiment={has_selected_experiment}",
                f"has_code_path={has_code_path}",
            ],
        }

    def _expected_artifact_check(self, payload: dict[str, Any], expected_artifacts: list[str]) -> dict[str, Any]:
        if not expected_artifacts:
            return {"name": "expected_artifacts", "passed": True}
        records = payload.get("artifact_records", [])
        present = {
            Path(str(item.get("path", ""))).name
            for item in records
            if isinstance(item, dict) and bool(item.get("substantive"))
        }
        missing = [artifact for artifact in expected_artifacts if artifact not in present]
        return {
            "name": "expected_artifacts",
            "passed": not missing,
            "evidence": [f"missing={missing}" if missing else "all_expected_present"],
        }

    def _implementation_targets(self, outputs_dir: Path, payload: dict[str, Any]) -> list[Path]:
        candidates: list[Path] = []
        generated_artifacts = payload.get("generated_artifacts", [])
        if isinstance(generated_artifacts, list):
            for item in generated_artifacts:
                if isinstance(item, str) and item.strip():
                    candidates.append(Path(item))
        code_path = str(payload.get("code_path", "")).strip()
        if code_path:
            candidates.append(Path(code_path))
        if not candidates:
            candidates.append(outputs_dir)
        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            signature = str(candidate)
            if signature not in seen:
                deduped.append(candidate)
                seen.add(signature)
        return deduped

    def _is_setup_command(self, command: str) -> bool:
        lowered = command.strip().lower()
        return lowered.startswith("pip install") or lowered.startswith("python -m pip install")

    def _citation_check(self, outputs_dir: Path) -> list[dict[str, Any]]:
        project_dir = self._project_dir(outputs_dir)
        paper_dir = project_dir / "outputs" / "paper"
        if not paper_dir.exists() and (project_dir / "paper").exists():
            paper_dir = project_dir / "paper"
        tex_path = paper_dir / "main.tex"
        bib_path = paper_dir / "references.bib"
        if not tex_path.exists() or not bib_path.exists():
            return []
        agent = CitationsAgent(project_dir / "knowledge" / "citations_cache.json")
        return agent.verify_all_in_latex(
            tex_path.read_text(encoding="utf-8"),
            bib_path.read_text(encoding="utf-8"),
        )

    def _paper_artifact_check(self, outputs_dir: Path) -> dict[str, Any] | None:
        project_dir = self._project_dir(outputs_dir)
        paper_dir = project_dir / "outputs" / "paper"
        if not paper_dir.exists() and (project_dir / "paper").exists():
            paper_dir = project_dir / "paper"
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

    def _result_path(self, outputs_dir: Path) -> Path:
        direct = outputs_dir / "result.json"
        if direct.exists():
            return direct
        sibling = outputs_dir.parent / "result.json"
        if sibling.exists():
            return sibling
        return direct

    def _project_dir(self, outputs_dir: Path) -> Path:
        if (outputs_dir.parent / "paper").exists():
            return outputs_dir.parent
        if outputs_dir.name == "artifacts" and outputs_dir.parent.parent.name == "runs":
            return outputs_dir.parents[2]
        if outputs_dir.parent.name == "paper" and outputs_dir.parent.parent.name == "outputs":
            return outputs_dir.parents[2]
        if outputs_dir.parent.name == "code" and outputs_dir.parent.parent.name == "outputs":
            return outputs_dir.parents[2]
        return outputs_dir.parents[1]
