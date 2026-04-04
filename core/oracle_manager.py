from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol

from core.artifact_runner import task_artifact_dir
from oracle.astro import catalog_checks, photometry_checks, physical_checks, run_oracle, spectral_checks, statistical_checks


@dataclass
class OracleResult:
    configured: bool
    applicable: bool
    oracle_pass: bool | None
    summary: dict
    failures: list[dict]
    warnings: list[dict]
    notes: list[str]
    report_path: str | None = None


class OracleAdapter(Protocol):
    def applicable(self, task_spec: dict, project_path: Path) -> bool:
        ...

    def run(self, task_spec: dict, task_output: str, artifacts: list[str], project_path: Path) -> OracleResult:
        ...


def select_oracle(task_spec: dict, project_path: Path) -> OracleAdapter | None:
    config_path = project_path / "project_config.json"
    if not config_path.exists():
        return None

    config = json.loads(config_path.read_text())
    oracle_module = config.get("oracle_module")
    if oracle_module == "astro":
        return AstroOracleAdapter()
    return None


def run_oracle_checks(
    task_spec: dict,
    output_path: Path,
    artifact_paths: list[str],
    project_path: Path,
) -> OracleResult:
    adapter = select_oracle(task_spec, project_path)
    if adapter is None:
        return OracleResult(
            configured=False,
            applicable=False,
            oracle_pass=None,
            summary={},
            failures=[],
            warnings=[],
            notes=["No oracle configured for this project"],
        )

    task_output = output_path.read_text() if output_path.exists() else ""
    if not adapter.applicable(task_spec, project_path):
        return OracleResult(
            configured=True,
            applicable=False,
            oracle_pass=None,
            summary={},
            failures=[],
            warnings=[],
            notes=["Configured oracle is not applicable for this task"],
        )

    return adapter.run(task_spec, task_output, artifact_paths, project_path)


class AstroOracleAdapter:
    CHECKS = {
        "velocity_physical": physical_checks.check_velocity_physical,
        "luminosity_physical": physical_checks.check_luminosity_physical,
        "eddington_limit": physical_checks.check_eddington_limit,
        "distance_modulus_consistent": physical_checks.check_distance_modulus_consistent,
        "stefan_boltzmann_consistent": physical_checks.check_stefan_boltzmann_consistent,
        "benchmark_star_recovery": catalog_checks.check_benchmark_star_recovery,
        "benchmark_galaxy_recovery": catalog_checks.check_benchmark_galaxy_recovery,
        "uncertainty_propagation": statistical_checks.check_uncertainty_propagation,
        "chi_squared_fit": statistical_checks.check_chi_squared_fit,
        "redshift_distance_consistency": statistical_checks.check_redshift_distance_consistency,
        "photon_count_statistics": statistical_checks.check_photon_count_statistics,
        "redshift_from_lines": spectral_checks.check_redshift_from_lines,
        "line_ratio_physical": spectral_checks.check_line_ratio_physical,
        "color_physical": photometry_checks.check_color_physical,
        "flux_conservation": photometry_checks.check_flux_conservation,
        "magnitude_system_consistent": photometry_checks.check_magnitude_system_consistent,
    }

    def applicable(self, task_spec: dict, project_path: Path) -> bool:
        return True

    def run(self, task_spec: dict, task_output: str, artifacts: list[str], project_path: Path) -> OracleResult:
        results_path = task_artifact_dir(project_path, task_spec["id"]) / "results.json"
        if not results_path.exists():
            return OracleResult(
                configured=True,
                applicable=False,
                oracle_pass=None,
                summary={},
                failures=[],
                warnings=[],
                notes=["No results.json found for oracle inputs"],
            )

        payload = json.loads(results_path.read_text())
        instructions = payload.get("oracle_inputs", [])
        if not instructions:
            return OracleResult(
                configured=True,
                applicable=False,
                oracle_pass=None,
                summary={},
                failures=[],
                warnings=[],
                notes=["No oracle_inputs found in results.json"],
            )

        checks = []
        for instruction in instructions:
            name = instruction.get("check")
            kwargs = instruction.get("kwargs", {})
            fn = self.CHECKS.get(name)
            if fn is None:
                checks.append(
                    {
                        "check": f"unknown oracle check: {name}",
                        "pass": None,
                        "warning": False,
                        "interpretation": f"SKIP: unknown oracle check '{name}'",
                    }
                )
                continue
            checks.append(fn(**kwargs))

        report = run_oracle.run_all_checks(checks)
        report_path = run_oracle.write_oracle_report(project_path, task_spec["id"], report)
        return OracleResult(
            configured=True,
            applicable=True,
            oracle_pass=report["oracle_pass"],
            summary=report["summary"],
            failures=report["failures"],
            warnings=report["warnings"],
            notes=[],
            report_path=str(report_path.relative_to(project_path)),
        )


def oracle_result_to_dict(result: OracleResult) -> dict:
    if isinstance(result, dict):
        return result
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    return {
        "configured": getattr(result, "configured"),
        "applicable": getattr(result, "applicable"),
        "oracle_pass": getattr(result, "oracle_pass"),
        "summary": getattr(result, "summary"),
        "failures": getattr(result, "failures"),
        "warnings": getattr(result, "warnings"),
        "notes": getattr(result, "notes"),
        "report_path": getattr(result, "report_path"),
    }
