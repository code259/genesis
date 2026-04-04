from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

import click

from genesis.config import ProjectConfig, load_project_config
from genesis.harness.loop import MetaHarnessLoop


@click.group()
def main() -> None:
    """Genesis v1 CLI."""


@main.command("init")
@click.option("--spec", "spec_path", type=click.Path(exists=True, path_type=Path))
@click.option("--interactive", is_flag=True, default=False)
def init_project(spec_path: Optional[Path], interactive: bool) -> None:
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
        click.echo(json.dumps(payload, indent=2))
        return
    if not spec_path:
        raise click.UsageError("Provide --spec or --interactive")
    config = load_project_config(spec_path)
    click.echo(json.dumps(config.to_dict(), indent=2))


@main.command("run")
@click.option("--project-id", required=False)
@click.option("--spec", "spec_path", type=click.Path(exists=True, path_type=Path), required=True)
def run_project(project_id: Optional[str], spec_path: Path) -> None:
    config = load_project_config(spec_path)
    project_id = project_id or uuid.uuid4().hex[:8]
    loop = MetaHarnessLoop(projects_root=Path(config.output_dir), taste_root=Path("taste_db"))
    result = loop.run(project_id, config)
    click.echo(json.dumps(result.to_dict(), indent=2))


@main.command("status")
@click.option("--project-id", required=True)
def status_project(project_id: str) -> None:
    project_dir = Path("projects") / project_id
    click.echo("exists" if project_dir.exists() else "missing")


@main.command("intervene")
@click.option("--project-id", required=True)
@click.option("--type", "intervention_type", required=True)
def intervene(project_id: str, intervention_type: str) -> None:
    intervention_path = Path("projects") / project_id / "human_intervention.json"
    intervention_path.parent.mkdir(parents=True, exist_ok=True)
    intervention_path.write_text(json.dumps({"type": intervention_type}, indent=2), encoding="utf-8")
    click.echo(str(intervention_path))


@main.command("results")
@click.option("--project-id", required=True)
def results(project_id: str) -> None:
    project_dir = Path("projects") / project_id / "outputs"
    click.echo(str(project_dir))


@main.command("build-manifold")
@click.option("--domain", required=True)
def build_manifold(domain: str) -> None:
    click.echo(f"build manifold for {domain}")


@main.command("init-taste")
def init_taste() -> None:
    Path("taste_db").mkdir(exist_ok=True)
    click.echo("taste_db initialized")


if __name__ == "__main__":
    main()
