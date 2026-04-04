from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Union

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
        references = paper_dir / "references.bib"
        run_index = paper_dir / "run_index.json"
        citations = CitationsAgent(project_dir / "knowledge" / "citations_cache.json")
        sections = self._collect_sections(project_dir)
        reference_metadata = self._collect_reference_metadata(project_dir, citations)
        references.write_text(
            "".join(citations.format_bibtex(metadata) for metadata in reference_metadata),
            encoding="utf-8",
        )
        run_index.write_text(
            json.dumps(sections["run_index"], indent=2),
            encoding="utf-8",
        )
        template = (Path(__file__).parent / "templates" / "main.tex").read_text(encoding="utf-8")
        tex = template.replace("{{TITLE}}", f"Genesis results for {project_id}")
        tex = tex.replace("{{ABSTRACT}}", sections["abstract"])
        tex = tex.replace("{{BODY}}", sections["body"])
        tex = tex.replace("{{FIGURE_BLOCK}}", sections["figure_block"])
        tex_path = paper_dir / "main.tex"
        tex_path.write_text(tex, encoding="utf-8")
        citation_flags = citations.verify_all_in_latex(tex, references.read_text(encoding="utf-8"))
        if citation_flags:
            (paper_dir / "citation_flags.json").write_text(json.dumps(citation_flags, indent=2), encoding="utf-8")
        pdf_path = paper_dir / "main.pdf"
        if subprocess.run(["which", "pdflatex"], capture_output=True, text=True).returncode == 0:
            compile_result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_path.name],
                cwd=paper_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if compile_result.returncode != 0:
                self._write_minimal_pdf(
                    pdf_path,
                    [
                        f"Genesis results for {project_id}",
                        "pdflatex compilation failed; embedded fallback summary generated.",
                        sections["abstract"],
                    ],
                )
        else:
            self._write_minimal_pdf(pdf_path, [f"Genesis results for {project_id}", sections["abstract"]])
        return {"pdf_path": str(pdf_path), "latex_path": str(tex_path)}

    def _collect_sections(self, project_dir: Path) -> dict[str, str]:
        run_dirs = sorted((project_dir / "runs").glob("*"))
        results = []
        for run_dir in run_dirs:
            result_path = run_dir / "result.json"
            adversarial_path = run_dir / "adversarial_report.json"
            verification_path = run_dir / "verification_report.json"
            if result_path.exists():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                results.append(
                    {
                        "result": payload,
                        "adversarial": json.loads(adversarial_path.read_text(encoding="utf-8")) if adversarial_path.exists() else {},
                        "verification": json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else {},
                    }
                )
        if not results:
            return {
                "abstract": "No verified results were available at synthesis time.",
                "body": "The project did not produce runnable outputs before synthesis.",
                "figure_block": "",
                "run_index": [],
            }
        top_metric = max(item["result"].get("primary_metric", 0.0) for item in results)
        verified_runs = sum(1 for item in results if item["verification"].get("passed"))
        abstract = (
            f"Genesis completed {len(results)} research iterations with {verified_runs} verified runs. "
            f"The best observed primary metric was {top_metric:.4f}."
        )
        body_lines = []
        for index, item in enumerate(results, start=1):
            result = item["result"]
            body_lines.append(
                "\\section*{Run "
                + str(index)
                + "}\n"
                + f"Task: {result.get('task_id', 'unknown')}\\\\\n"
                + f"Summary: {result.get('summary', 'n/a')}\\\\\n"
                + f"Primary metric: {result.get('primary_metric', 0.0)}\\\\\n"
                + f"Verification passed: {item['verification'].get('passed', False)}\n"
            )
        body = "\n".join(body_lines)
        if self.runtime is not None:
            try:
                payload = self.runtime.generate_task(
                    category="genesis-paper",
                    instruction="Draft the abstract and body for the final paper from verified run artifacts.",
                    context={
                        "project_id": project_dir.name,
                        "results": [item["result"] for item in results],
                        "verification": [item["verification"] for item in results],
                    },
                    budget={"sections": ["abstract", "body"]},
                )
                abstract = str(payload.get("summary") or abstract)
                body = str(payload.get("paper_body") or body)
            except ProviderRuntimeError:
                pass
        figure_block = self._build_figure_block(project_dir, results)
        return {
            "abstract": abstract,
            "body": body,
            "figure_block": figure_block,
            "run_index": [
                {
                    "task_id": item["result"].get("task_id", "unknown"),
                    "primary_metric": item["result"].get("primary_metric", 0.0),
                    "verification_passed": item["verification"].get("passed", False),
                }
                for item in results
            ],
        }

    def _collect_reference_metadata(self, project_dir: Path, citations: CitationsAgent) -> list[dict[str, object]]:
        spec_path = project_dir / "spec.json"
        references: list[dict[str, object]] = []
        if spec_path.exists():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            query = spec.get("research_question", "")
            if query:
                references.extend(citations.search_semantic_scholar(query)[:2])
                references.extend(citations.search_crossref(query, limit=2)[:1])
                references.extend(citations.search_arxiv(query, limit=1)[:1])
        if not references:
            references.append({"title": "Genesis v1", "year": 2026, "authors": [{"name": "Genesis"}]})
        return references

    def _build_figure_block(self, project_dir: Path, results: list[dict[str, object]]) -> str:
        trajectory = None
        for item in results:
            selected = item["result"].get("selected_experiment") if isinstance(item.get("result"), dict) else None
            if isinstance(selected, dict) and selected.get("trajectory"):
                trajectory = selected["trajectory"]
                break
        if not trajectory:
            metrics = [item["result"].get("primary_metric", 0.0) for item in results if isinstance(item.get("result"), dict)]
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
        pdf.extend(
            (
                "trailer << /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".format(
                    size=len(offsets), xref=xref_offset
                )
            ).encode("latin-1")
        )
        pdf_path.write_bytes(bytes(pdf))
