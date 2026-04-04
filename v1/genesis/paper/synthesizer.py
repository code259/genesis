from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Union

from genesis.agents.runtime import CodingAgentRuntime, ProviderRuntimeError
from genesis.models import FigureSpec
from genesis.modules.citations.agent import CitationsAgent
from genesis.modules.plotting.module import PlottingModule


class PaperSynthesizer:
    def __init__(self, project_root: Union[str, Path], runtime: CodingAgentRuntime | None = None):
        self.project_root = Path(project_root)
        self.runtime = runtime

    def synthesize(self, project_id: str) -> dict[str, str]:
        project_dir = self.project_root / project_id
        paper_dir = project_dir / "outputs" / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        references_path = paper_dir / "references.bib"
        tex_path = paper_dir / "main.tex"
        report_path = paper_dir / "synthesis_report.json"
        citations = CitationsAgent(project_dir / "knowledge" / "citations_cache.json")

        sections = self._collect_sections(project_dir)
        reference_metadata = self._collect_reference_metadata(project_dir, sections["runs"], citations)
        references_path.write_text("".join(citations.format_bibtex(metadata) for metadata in reference_metadata), encoding="utf-8")

        template = (Path(__file__).parent / "templates" / "main.tex").read_text(encoding="utf-8")
        tex = (
            template.replace("{{TITLE}}", f"Genesis results for {project_id}")
            .replace("{{ABSTRACT}}", sections["abstract"])
            .replace("{{INTRODUCTION}}", sections["introduction"])
            .replace("{{METHODS}}", sections["methods"])
            .replace("{{RESULTS}}", sections["results_text"])
            .replace("{{DISCUSSION}}", sections["discussion"])
            .replace("{{FIGURE_BLOCK}}", sections["figure_block"])
        )
        tex_path.write_text(tex, encoding="utf-8")

        citation_flags = citations.verify_all_in_latex(tex, references_path.read_text(encoding="utf-8"))
        if citation_flags:
            (paper_dir / "citation_flags.json").write_text(json.dumps(citation_flags, indent=2), encoding="utf-8")

        pdf_path = paper_dir / "main.pdf"
        compile_backend = "fallback_pdf"
        if subprocess.run(["which", "pdflatex"], capture_output=True, text=True).returncode == 0:
            compile_backend = "pdflatex"
            compile_log = self._compile_latex(tex_path)
            (paper_dir / "compile.log").write_text(compile_log, encoding="utf-8")
            if not pdf_path.exists():
                self._write_minimal_pdf(pdf_path, [f"Genesis results for {project_id}", sections["abstract"]])
                compile_backend = "fallback_pdf"
        else:
            self._write_minimal_pdf(pdf_path, [f"Genesis results for {project_id}", sections["abstract"]])

        report = {
            "project_id": project_id,
            "compile_backend": compile_backend,
            "verified_run_count": sections["verified_run_count"],
            "total_run_count": sections["total_run_count"],
            "reference_count": len(reference_metadata),
            "pdf_path": str(pdf_path),
            "latex_path": str(tex_path),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {"pdf_path": str(pdf_path), "latex_path": str(tex_path)}

    def _collect_sections(self, project_dir: Path) -> dict[str, Any]:
        runs = []
        for run_dir in sorted((project_dir / "runs").glob("*")):
            result_path = run_dir / "result.json"
            verification_path = run_dir / "verification_report.json"
            if not result_path.exists():
                continue
            runs.append(
                {
                    "result": json.loads(result_path.read_text(encoding="utf-8")),
                    "verification": json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else {},
                }
            )
        if not runs:
            return {
                "abstract": "No verified results were available at synthesis time.",
                "introduction": "The project did not produce runnable outputs before synthesis.",
                "methods": "No method artifacts were captured.",
                "results_text": "No results were available.",
                "discussion": "Further execution is required before a final paper can be synthesized.",
                "figure_block": "",
                "verified_run_count": 0,
                "total_run_count": 0,
                "runs": [],
            }

        verified_runs = [run for run in runs if run["verification"].get("passed")]
        source_runs = verified_runs or runs
        top_metric = max(run["result"].get("primary_metric", 0.0) for run in source_runs)
        abstract = (
            f"Genesis completed {len(runs)} research iterations, with {len(verified_runs)} verification-passing runs. "
            f"The best observed primary metric was {top_metric:.4f}."
        )
        introduction = "This report summarizes the current Genesis v1 project state and emphasizes verification-passing artifacts."
        methods = "\n".join(
            f"Run {index}: task={run['result'].get('task_id', 'unknown')}, provider={run['result'].get('agent_runtime', {}).get('provider', 'n/a')}, model={run['result'].get('agent_runtime', {}).get('model', 'n/a')}."
            for index, run in enumerate(source_runs, start=1)
        )
        results_text = "\n\n".join(
            f"\\textbf{{Run {index}}}: {run['result'].get('summary', 'n/a')} Primary metric={run['result'].get('primary_metric', 0.0)}. Verification passed={run['verification'].get('passed', False)}."
            for index, run in enumerate(source_runs, start=1)
        )
        discussion = (
            "No runs passed verification; follow-up work should prioritize fixing verification failures."
            if not verified_runs
            else f"{len(verified_runs)} of {len(runs)} runs passed verification. Future work should expand the verified result set."
        )

        if self.runtime is not None:
            try:
                payload = self.runtime.generate_task(
                    category="genesis-paper",
                    instruction="Draft a concise scientific paper summary from verified Genesis run artifacts.",
                    context={"project_id": project_dir.name, "results": [run["result"] for run in source_runs]},
                    budget={"sections": ["abstract", "results"]},
                )
                abstract = str(payload.get("summary") or abstract)
                generated_body = str(payload.get("paper_body") or "").strip()
                if generated_body:
                    results_text = generated_body
            except ProviderRuntimeError:
                pass

        return {
            "abstract": abstract,
            "introduction": introduction,
            "methods": methods,
            "results_text": results_text,
            "discussion": discussion,
            "figure_block": self._build_figure_block(project_dir, source_runs),
            "verified_run_count": len(verified_runs),
            "total_run_count": len(runs),
            "runs": source_runs,
        }

    def _collect_reference_metadata(self, project_dir: Path, runs: list[dict[str, Any]], citations: CitationsAgent) -> list[dict[str, object]]:
        references: list[dict[str, object]] = []
        seen: set[str] = set()

        spec_path = project_dir / "spec.json"
        if spec_path.exists():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            query = spec.get("research_question", "")
            if query:
                references.extend(citations.search_semantic_scholar(query)[:2])
                references.extend(citations.search_crossref(query, limit=1))

        for run in runs:
            for citation in run["result"].get("citations", []):
                if isinstance(citation, dict):
                    references.append(citation)

        deduped = []
        for metadata in references:
            title = str(metadata.get("title", "")).strip().lower()
            doi = str(metadata.get("doi") or metadata.get("DOI") or metadata.get("externalIds", {}).get("DOI", "")).strip().lower()
            key = doi or title
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(metadata)
        if not deduped:
            deduped.append({"title": "Genesis v1", "year": 2026, "authors": [{"name": "Genesis"}]})
        return deduped

    def _build_figure_block(self, project_dir: Path, runs: list[dict[str, Any]]) -> str:
        trajectory = None
        for run in runs:
            selected = run["result"].get("selected_experiment")
            if isinstance(selected, dict) and selected.get("trajectory"):
                trajectory = selected["trajectory"]
                break
        if not trajectory:
            metrics = [run["result"].get("primary_metric", 0.0) for run in runs]
            if len(metrics) > 1:
                trajectory = metrics
        if not trajectory:
            return ""

        plotting = PlottingModule(project_dir / "outputs" / "paper" / "figures")
        figure = plotting.generate_figure(
            FigureSpec(
                figure_type="line",
                data_source={"y": trajectory},
                axis_labels=["Iteration", "Metric"],
                title="trajectory_overview",
                style="publication",
            )
        )
        relative_pdf = Path(figure.pdf_path).relative_to(project_dir / "outputs" / "paper")
        return (
            "\\section{Figures}\n"
            "\\begin{figure}[h]\n\\centering\n"
            f"\\includegraphics[width=0.8\\linewidth]{{{relative_pdf.as_posix()}}}\n"
            "\\caption{Metric trajectory across the best available results.}\n"
            "\\end{figure}\n"
        )

    def _compile_latex(self, tex_path: Path) -> str:
        log_parts = []
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_path.name],
                cwd=tex_path.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            log_parts.extend([result.stdout, result.stderr])
            if result.returncode != 0:
                break
        return "\n".join(part for part in log_parts if part)

    def _write_minimal_pdf(self, pdf_path: Path, lines: list[str]) -> None:
        escaped_lines = []
        for index, line in enumerate(lines):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            escaped_lines.append(f"BT /F1 12 Tf 72 {720 - index * 18} Td ({escaped}) Tj ET")
        stream = "\n".join(escaped_lines).encode("latin-1", errors="replace")
        objects = [
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
            f"4 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n",
            b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        ]
        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for obj in objects:
            offsets.append(len(pdf))
            pdf.extend(obj)
        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
        pdf.extend(("trailer << /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".format(size=len(offsets), xref=xref_offset)).encode("latin-1"))
        pdf_path.write_bytes(bytes(pdf))
