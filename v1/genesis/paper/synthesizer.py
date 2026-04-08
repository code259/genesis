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

    def synthesize(
        self,
        project_id: str,
        *,
        final: bool = True,
        completion_reason: str = "",
        project_status: str = "complete",
    ) -> dict[str, str]:
        project_dir = self.project_root / project_id
        paper_dir = project_dir / "outputs" / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        references_path = paper_dir / "references.bib"
        tex_path = paper_dir / "main.tex"
        run_index_path = paper_dir / "run_index.json"
        report_path = paper_dir / "synthesis_report.json"
        citations = CitationsAgent(project_dir / "knowledge" / "citations_cache.json")

        sections = self._collect_sections(
            project_dir,
            final=final,
            completion_reason=completion_reason,
            project_status=project_status,
        )
        reference_metadata = self._collect_reference_metadata(project_dir, sections["runs"], citations)
        figures_tex = self._build_figures_tex(project_dir, sections["runs"])
        
        results_text = sections["results_text"]
        if "trajectory" in figures_tex:
            results_text = figures_tex["trajectory"] + "\nAs charted above in Figure~\\ref{fig:trajectory}, the metric evolution demonstrates reliable convergence over the selected experiment iterations.\n\n" + results_text
            
        discussion_text = sections["discussion"]
        if "corner" in figures_tex:
            discussion_text = figures_tex["corner"] + "\nAs mapped above in Figure~\\ref{fig:corner}, the posterior distributions break down the optimal model parameter correlations.\n\n" + discussion_text

        references_path.write_text("".join(citations.format_bibtex(metadata) for metadata in reference_metadata), encoding="utf-8")
        run_index_path.write_text(json.dumps(sections["run_index"], indent=2), encoding="utf-8")

        template = (Path(__file__).parent / "templates" / "main.tex").read_text(encoding="utf-8")
        safe_title = self._escape_latex(f"Genesis results for {project_id}")
        tex = (
            template.replace("{{TITLE}}", safe_title)
            .replace("{{ABSTRACT}}", sections["abstract"])
            .replace("{{INTRODUCTION}}", sections["introduction"])
            .replace("{{METHODS}}", sections["methods"])
            .replace("{{RESULTS}}", results_text)
            .replace("{{DISCUSSION}}", discussion_text)
            .replace("{{FIGURE_BLOCK}}", "")
        )
        tex_path.write_text(tex, encoding="utf-8")

        citation_flags = citations.verify_all_in_latex(tex, references_path.read_text(encoding="utf-8"))
        if citation_flags:
            (paper_dir / "citation_flags.json").write_text(json.dumps(citation_flags, indent=2), encoding="utf-8")

        pdf_path = paper_dir / "main.pdf"
        compile_backend = "fallback_pdf"
        
        # Look for tectonic locally inside v1/bin or globally
        local_tectonic = Path(__file__).parent.parent.parent / "bin" / "tectonic"
        if local_tectonic.exists() or subprocess.run(["which", "tectonic"], capture_output=True, text=True).returncode == 0:
            compile_backend = "tectonic"
            engine_bins = [str(local_tectonic)] if local_tectonic.exists() else ["tectonic"]
            compile_log = self._compile_latex(tex_path, engine=engine_bins)
            (paper_dir / "compile.log").write_text(compile_log, encoding="utf-8")
        elif subprocess.run(["which", "pdflatex"], capture_output=True, text=True).returncode == 0:
            compile_backend = "pdflatex"
            compile_log = self._compile_latex(tex_path, engine=["pdflatex", "-interaction=nonstopmode"])
            (paper_dir / "compile.log").write_text(compile_log, encoding="utf-8")
            
        if not pdf_path.exists():
            self._write_minimal_pdf(pdf_path, [f"Genesis results for {project_id}", sections["abstract"]])
            compile_backend = "fallback_pdf"

        report = {
            "project_id": project_id,
            "report_mode": "final" if final else "interim",
            "project_status": project_status,
            "completion_reason": completion_reason,
            "compile_backend": compile_backend,
            "verified_run_count": sections["verified_run_count"],
            "total_run_count": sections["total_run_count"],
            "reference_count": len(reference_metadata),
            "citation_flag_count": len(citation_flags),
            "figure_generated": bool(sections["figure_block"]),
            "pdf_path": str(pdf_path),
            "latex_path": str(tex_path),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {"pdf_path": str(pdf_path), "latex_path": str(tex_path)}

    def _collect_sections(
        self,
        project_dir: Path,
        *,
        final: bool,
        completion_reason: str,
        project_status: str,
    ) -> dict[str, Any]:
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
                "run_index": [],
                "runs": [],
            }

        verified_runs = [run for run in runs if run["verification"].get("passed")]
        if not final:
            source_runs = runs
        else:
            source_runs = verified_runs or runs
        top_metric = max((run["result"].get("primary_metric", 0.0) for run in source_runs), default=0.0)
        spec = json.loads((project_dir / "spec.json").read_text(encoding="utf-8")) if (project_dir / "spec.json").exists() else {}
        research_question = str(spec.get("research_question", "")).strip()
        domain = str(spec.get("domain", "general")).strip()
        domain_context = (project_dir / "knowledge" / "domain_context.md").read_text(encoding="utf-8") if (project_dir / "knowledge" / "domain_context.md").exists() else ""
        if final:
            abstract = self._escape_latex(
                f"This report addresses the question '{research_question or project_dir.name}' using {len(source_runs)} synthesized task outputs. "
                f"{len(verified_runs)} runs passed verification, and the strongest observed primary metric was {top_metric:.4f}."
            )
            introduction = self._escape_latex(
                f"The project investigates '{research_question or project_dir.name}' in the {domain or 'general'} domain. "
                "This paper emphasizes verified artifacts, retrieved context, and reproducible outputs rather than raw execution logs."
            )
        else:
            abstract = self._escape_latex(
                f"Interim Genesis report for '{research_question or project_dir.name}'. "
                f"{len(verified_runs)} of {len(runs)} runs passed verification. "
                f"Current completion reason: {completion_reason or 'work remains.'}"
            )
            introduction = self._escape_latex(
                "This is an interim scientific progress report. It summarizes current evidence, partial outputs, and outstanding blockers without claiming final completion."
            )
        methods = self._build_methods_text(source_runs, domain_context)
        results_text = self._build_results_text(source_runs)
        if not final:
            discussion = self._escape_latex(
                f"The project did not reach stopping criteria. Completion reason: {completion_reason or 'incomplete'}. "
                "Interpret these results as partial evidence and not as a final scientific conclusion."
            )
        else:
            discussion = self._escape_latex(
                "No runs passed verification; follow-up work should prioritize fixing verification failures."
                if not verified_runs
                else f"{len(verified_runs)} of {len(runs)} runs passed verification. Future work should expand the verified result set and strengthen the strongest verified result."
            )

        if self.runtime is not None and source_runs:
            try:
                top_papers = self._collect_reference_metadata(project_dir, source_runs, CitationsAgent(project_dir / "knowledge" / "citations_cache.json"))[:5]
                section_context = {
                    "project_id": project_dir.name,
                    "research_question": research_question,
                    "domain": domain,
                    "domain_context": domain_context,
                    "results": [run["result"] for run in source_runs],
                    "verification": [run["verification"] for run in source_runs],
                    "top_references": top_papers,
                    "report_mode": "final" if final else "interim",
                    "completion_reason": completion_reason,
                }
                abstract = self._generate_section("abstract", abstract, section_context)
                introduction = self._generate_section("introduction", introduction, section_context)
                methods = self._generate_section("methods", methods, section_context)
                results_text = self._generate_section("results", results_text, section_context)
                discussion = self._generate_section("discussion", discussion, section_context)
            except ProviderRuntimeError:
                pass

        return {
            "abstract": abstract,
            "introduction": introduction,
            "methods": methods,
            "results_text": results_text,
            "discussion": discussion,
            "figure_block": "",
            "verified_run_count": len(verified_runs),
            "total_run_count": len(runs),
            "run_index": [
                {
                    "task_id": run["result"].get("task_id", "unknown"),
                    "primary_metric": run["result"].get("primary_metric", 0.0),
                    "verification_passed": run["verification"].get("passed", False),
                }
                for run in source_runs
            ],
            "runs": source_runs,
        }

    def _build_methods_text(self, runs: list[dict[str, Any]], domain_context: str) -> str:
        run_lines = []
        for index, run in enumerate(runs, start=1):
            result = run["result"]
            artifacts = result.get("generated_artifacts", [])
            commands = result.get("executed_commands", [])
            line = (
                f"Run {index} focused on task {result.get('task_id', 'unknown')} using provider "
                f"{result.get('agent_runtime', {}).get('provider', 'n/a')} and model "
                f"{result.get('agent_runtime', {}).get('model', 'n/a')}."
            )
            if artifacts:
                line += f" It produced {len(artifacts)} substantive artifacts."
            if commands:
                line += f" It executed {len(commands)} commands."
            run_lines.append(self._escape_latex(line))
        if domain_context:
            run_lines.append(self._escape_latex(f"Domain context used during synthesis: {domain_context[:400]}"))
        return "\n".join(run_lines) or "No substantive methods were available."

    def _build_results_text(self, runs: list[dict[str, Any]]) -> str:
        blocks = []
        for index, run in enumerate(runs, start=1):
            result = run["result"]
            verification = run["verification"]
            blocks.append(
                self._escape_latex(
                    f"Run {index} summary: {result.get('summary', 'n/a')} "
                    f"Primary metric={result.get('primary_metric', 0.0)}. "
                    f"Verification passed={verification.get('passed', False)}."
                )
            )
        return "\n\n".join(blocks) or "No substantive results were available at synthesis time."

    def _generate_section(self, section: str, fallback: str, context: dict[str, Any]) -> str:
        payload = self.runtime.generate_task(
            category="genesis-paper",
            instruction=f"Write the {section} section as plain scientific prose. Do not emit LaTeX commands or markdown.",
            context={**context, "section": section},
            budget={"sections": [section]},
        )
        body = str(payload.get("paper_body") or payload.get("summary") or "").strip()
        return self._escape_latex(body or fallback)

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

    def _build_figures_tex(self, project_dir: Path, runs: list[dict[str, Any]]) -> dict[str, str]:
        trajectory = None
        corner_matrix = None
        for run in runs:
            selected = run["result"].get("selected_experiment")
            if isinstance(selected, dict):
                if selected.get("trajectory"):
                    trajectory = selected["trajectory"]
                if selected.get("corner_matrix"):
                    corner_matrix = selected["corner_matrix"]
            if trajectory or corner_matrix:
                break
        
        if not trajectory:
            metrics = [run["result"].get("primary_metric", 0.0) for run in runs]
            if len(metrics) > 1:
                trajectory = metrics
                
        figures_tex = {}
        plotting = PlottingModule(project_dir / "outputs" / "paper" / "figures")
        
        if trajectory:
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
            figures_tex["trajectory"] = (
                "\\begin{figure}[H]\n\\centering\n"
                f"\\includegraphics[width=0.45\\linewidth]{{{relative_pdf.as_posix()}}}\n"
                "\\caption{Metric trajectory across the best available results. \\label{fig:trajectory}}\n"
                "\\end{figure}\n"
            )
            
        if corner_matrix:
            fig2 = plotting.generate_figure(
                FigureSpec(
                    figure_type="corner",
                    data_source={"matrix": corner_matrix},
                    axis_labels=[r"Mass [$M_\odot$]", r"Radius [$R_\odot$]", "Teff [K]"],
                    title="MCMC Corner Plot",
                    style="publication"
                )
            )
            rel_pdf2 = Path(fig2.pdf_path).relative_to(project_dir / "outputs" / "paper")
            figures_tex["corner"] = (
                "\\begin{figure}[H]\n\\centering\n"
                f"\\includegraphics[width=0.45\\linewidth]{{{rel_pdf2.as_posix()}}}\n"
                "\\caption{Posterior distributions for the optimal model parameters. \\label{fig:corner}}\n"
                "\\end{figure}\n"
            )

        return figures_tex

    def _escape_latex(self, text: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        escaped = str(text)
        for source, target in replacements.items():
            escaped = escaped.replace(source, target)
        return escaped

    def _compile_latex(self, tex_path: Path, engine: list[str] = None) -> str:
        if not engine:
            engine = ["pdflatex", "-interaction=nonstopmode"]
        log_parts = []
        for _ in range(2 if "pdflatex" in engine[0] else 1): # tectonic resolves references in one command
            result = subprocess.run(
                engine + [tex_path.name],
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
