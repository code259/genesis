from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import click

from genesis.config import load_project_config
from genesis.harness.loop import MetaHarnessLoop
from genesis.storage.filesystem import ProjectFilesystem


INTERVENTION_TYPES = ["REDIRECT", "APPROVE", "REJECT", "STOP"]


def _echo_json(payload: dict[str, object]) -> None:
    click.echo(json.dumps(payload, indent=2))


@click.group()
def main() -> None:
    """Genesis v1 CLI."""


@main.command("init")
@click.option("--spec", "spec_path", type=click.Path(exists=True, path_type=Path))
@click.option("--interactive", is_flag=True, default=False)
@click.option("--project-id", required=False)
def init_project(spec_path: Optional[Path], interactive: bool, project_id: Optional[str]) -> None:
    if interactive:
        payload = {
            "research_question": click.prompt("Research question"),
            "domain": click.prompt("Domain", default="general"),
            "success_criteria": [],
            "oracle_hints": [],
            "compute_budget": click.prompt("Compute budget", default="local_cpu"),
            "time_budget_hours": click.prompt("Time budget", default=2, type=int),
            "domain_knowledge_model": click.prompt("Domain knowledge model", default="none"),
            "output_dir": "projects",
        }
        project_id = project_id or uuid.uuid4().hex[:8]
        project_dir = Path(payload["output_dir"]) / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "spec.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(str(project_dir))
        return
    if not spec_path:
        raise click.UsageError("Provide --spec or --interactive")
    config = load_project_config(spec_path)
    project_id = project_id or uuid.uuid4().hex[:8]
    project_dir = Path(config.output_dir) / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "spec.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    click.echo(str(project_dir))


@main.command("run")
@click.option("--project-id", required=False)
@click.option("--spec", "spec_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--max-runs", default=50, show_default=True, type=int)
@click.option(
    "--runtime-config",
    type=click.Path(exists=True, path_type=Path),
    default=Path(__file__).resolve().parents[2] / "configs" / "runtime_omo.jsonc",
    show_default=True,
)
def run_project(project_id: Optional[str], spec_path: Path, max_runs: int, runtime_config: Path) -> None:
    config = load_project_config(spec_path)
    project_id = project_id or uuid.uuid4().hex[:8]
    output_root = Path(config.output_dir)
    loop = MetaHarnessLoop(
        projects_root=output_root,
        taste_root=output_root.parent / "taste_db",
        runtime_config_path=runtime_config,
    )
    result = loop.run(project_id, config, max_runs=max_runs)
    _echo_json(result.to_dict())


@main.command("doctor")
@click.option(
    "--runtime-config",
    type=click.Path(exists=True, path_type=Path),
    default=Path(__file__).resolve().parents[2] / "configs" / "runtime_omo.jsonc",
    show_default=True,
)
@click.option("--probe-models", is_flag=True, default=False, help="Run a minimal probe against configured model routes.")
def doctor(runtime_config: Path, probe_models: bool) -> None:
    runtime = MetaHarnessLoop(
        projects_root=Path("projects"),
        taste_root=Path("taste_db"),
        runtime_config_path=runtime_config,
    ).agent_runtime
    _echo_json(runtime.check_health(probe_models=probe_models))


@main.command("status")
@click.option("--project-id", required=True)
@click.option("--root", "root_dir", type=click.Path(path_type=Path), default=Path("projects"))
def status_project(project_id: str, root_dir: Path) -> None:
    project_dir = root_dir / project_id
    if not project_dir.exists():
        click.echo("missing")
        return
    filesystem = ProjectFilesystem(root_dir)
    state = filesystem.read_project_state(project_id)
    payload = {
        "project_id": project_id,
        "runs": len(list((project_dir / "runs").glob("*"))),
        "has_halt": (project_dir / "HALT.json").exists(),
        "has_latex": (project_dir / "outputs" / "paper" / "main.tex").exists(),
        "has_pdf": (project_dir / "outputs" / "paper" / "main.pdf").exists(),
        "log_path": str(project_dir / "genesis.log"),
        "state": state,
    }
    _echo_json(payload)


@main.command("intervene")
@click.option("--project-id", required=True)
@click.option("--type", "intervention_type", type=click.Choice(INTERVENTION_TYPES), required=True)
@click.option("--root", "root_dir", type=click.Path(path_type=Path), default=Path("projects"))
def intervene(project_id: str, intervention_type: str, root_dir: Path) -> None:
    intervention_path = root_dir / project_id / "human_intervention.json"
    intervention_path.parent.mkdir(parents=True, exist_ok=True)
    intervention_path.write_text(json.dumps({"type": intervention_type}, indent=2), encoding="utf-8")
    click.echo(str(intervention_path))


@main.command("results")
@click.option("--project-id", required=True)
@click.option("--root", "root_dir", type=click.Path(path_type=Path), default=Path("projects"))
def results(project_id: str, root_dir: Path) -> None:
    paper_dir = root_dir / project_id / "outputs" / "paper"
    payload = {
        "paper_dir": str(paper_dir),
        "code_dir": str(root_dir / project_id / "outputs" / "code"),
        "latex_path": str(paper_dir / "main.tex"),
        "pdf_path": str(paper_dir / "main.pdf"),
        "run_index": json.loads((paper_dir / "run_index.json").read_text(encoding="utf-8")) if (paper_dir / "run_index.json").exists() else [],
        "citation_flags": json.loads((paper_dir / "citation_flags.json").read_text(encoding="utf-8")) if (paper_dir / "citation_flags.json").exists() else [],
        "synthesis_report": json.loads((paper_dir / "synthesis_report.json").read_text(encoding="utf-8")) if (paper_dir / "synthesis_report.json").exists() else {},
    }
    _echo_json(payload)


@main.command("build-manifold")
@click.option("--domain", required=True)
def build_manifold(domain: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["python3", "scripts/build_manifold.py", "--domain", domain],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise click.ClickException(result.stderr.strip() or "build_manifold failed")
    click.echo(result.stdout.strip() or f"build manifold for {domain}")


@main.command("init-taste")
def init_taste() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["python3", "scripts/init_taste.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise click.ClickException(result.stderr.strip() or "init_taste failed")
    click.echo(result.stdout.strip() or "taste_db initialized")


if __name__ == "__main__":
    main()
