import json
from pathlib import Path
import shutil

from core.artifact_runner import task_artifact_dir


def build_stage_summary(project_path: Path, stage: int) -> Path:
    stage_dir = project_path / "stages" / f"stage_{stage}"
    lines = [f"# Stage {stage} Summary", ""]

    for verify_path in sorted(stage_dir.glob("*_verify.json")):
        verification = json.loads(verify_path.read_text())
        if verification.get("status") != "ACCEPT":
            continue

        task_id = verify_path.name.replace("_verify.json", "")
        output_path = stage_dir / f"{task_id}.md"
        lines.append(f"## {task_id}")
        lines.append("")
        lines.append(output_path.read_text().strip() if output_path.exists() else "Output missing.")
        lines.append("")

    summary_path = stage_dir / "summary.md"
    summary_path.write_text("\n".join(lines).strip() + "\n")
    return summary_path


def build_paper_package(project_path: Path) -> Path:
    tasks_path = project_path / "tasks.json"
    tasks = json.loads(tasks_path.read_text()) if tasks_path.exists() else []

    paper_dir = project_path / "paper"
    sections_dir = paper_dir / "sections"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    sections_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    section_names = []
    manifest = {"sections": [], "figures": []}

    for task in sorted(tasks, key=lambda item: (item["stage"], item["id"])):
        stage_dir = project_path / "stages" / f"stage_{task['stage']}"
        verify_path = stage_dir / f"{task['id']}_verify.json"
        output_path = stage_dir / f"{task['id']}.md"
        if not verify_path.exists() or not output_path.exists():
            continue

        verification = json.loads(verify_path.read_text())
        if verification.get("status") != "ACCEPT":
            continue

        section_name = f"{task['id']}.tex"
        section_path = sections_dir / section_name
        section_path.write_text(_task_to_latex(task["id"], output_path.read_text()))
        section_names.append(section_name)
        manifest["sections"].append({"task_id": task["id"], "file": f"sections/{section_name}"})

        for png_path in sorted(task_artifact_dir(project_path, task["id"]).glob("*.png")):
            dest = figures_dir / png_path.name
            shutil.copyfile(png_path, dest)
            manifest["figures"].append({"task_id": task["id"], "file": f"figures/{png_path.name}"})

    refs_path = paper_dir / "refs.bib"
    refs_path.write_text("% Add BibTeX entries manually.\n")

    readme_path = paper_dir / "README.md"
    readme_path.write_text(
        "# Paper Package\n\n"
        "- `main.tex`: top-level manuscript entrypoint\n"
        "- `sections/`: task-derived section files\n"
        "- `figures/`: copied verified PNG artifacts\n"
        "- `tables/`: reserved for generated tables\n"
        "- `refs.bib`: manual citation placeholder scaffold\n"
    )

    manifest_path = paper_dir / "paper_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    main_tex = ["\\documentclass{article}", "\\usepackage{graphicx}", "\\begin{document}", "\\tableofcontents"]
    for section_name in section_names:
        main_tex.append(f"\\input{{sections/{section_name}}}")
    if manifest["figures"]:
        main_tex.append("\\section*{Figures}")
        for figure in manifest["figures"]:
            main_tex.extend(
                [
                    "\\begin{figure}[h]",
                    "\\centering",
                    f"\\includegraphics[width=0.8\\textwidth]{{{figure['file']}}}",
                    f"\\caption{{Artifact from {figure['task_id']}}}",
                    "\\end{figure}",
                ]
            )
    main_tex.append("\\bibliographystyle{plain}")
    main_tex.append("\\bibliography{refs}")
    main_tex.append("\\end{document}")
    (paper_dir / "main.tex").write_text("\n".join(main_tex) + "\n")

    return paper_dir


def _task_to_latex(task_id: str, content: str) -> str:
    escaped = content.replace("\\", "\\textbackslash{}")
    return f"\\section{{{task_id}}}\n\\begin{{verbatim}}\n{escaped}\n\\end{{verbatim}}\n"
