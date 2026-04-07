import json
from pathlib import Path

from genesis.config import ProjectConfig
from genesis.domain_knowledge.registry import DomainKnowledgeRegistry
from genesis.models import FigureSpec
from genesis.modules.citations.agent import CitationsAgent
from genesis.modules.oracle.generator import DomainOracleGenerator
from genesis.modules.oracle.validator import OracleValidator
from genesis.modules.plotting.module import PlottingModule
from genesis.modules.verification.pipeline import VerificationPipeline
from genesis.paper.synthesizer import PaperSynthesizer


def test_oracle_generation_and_validation(tmp_path):
    config = ProjectConfig(
        research_question="Check oracle generation",
        domain="general",
        compute_budget="local_cpu",
        time_budget_hours=1,
        domain_knowledge_model="none",
        output_dir=str(tmp_path / "projects"),
        oracle_hints=["verify sample consistency", "metric consistency"],
    )
    source = DomainOracleGenerator().generate(config)
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(source, encoding="utf-8")
    validator = OracleValidator()
    assert DomainOracleGenerator().validate_oracle(oracle_path)
    result = validator.run_oracle(oracle_path, tmp_path)
    assert result.pass_rate >= 0.0
    synthetic = validator.validate_with_synthetic_data(oracle_path)
    assert synthetic["passed"] is True


def test_citations_plotting_and_verification(tmp_path):
    citations = CitationsAgent(tmp_path / "cache" / "citations.json")
    bibtex = citations.format_bibtex(
        {
            "title": "A test paper",
            "year": 2026,
            "authors": [{"name": "Test Author"}],
            "doi": "10.1000/test-doi",
        }
    )
    assert "@article" in bibtex
    citations.verify_citation = lambda citation: {"verified": True, "citation": citation, "evidence": []}  # type: ignore[method-assign]
    assert citations.verify_all_in_latex("\\cite{test_author_2026_a_test_paper}", bibtex) == []

    plotting = PlottingModule(tmp_path / "figures")
    figure = plotting.generate_figure(
        FigureSpec(
            figure_type="line",
            data_source=[0.1, 0.2, 0.3],
            axis_labels=["x", "y"],
            title="Demo Figure",
            style="publication",
        )
    )
    assert Path(figure.pdf_path).exists()
    assert Path(figure.pdf_path).with_name("demo_figure.metadata.json").exists()

    verification = VerificationPipeline()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "result.json").write_text("{}", encoding="utf-8")
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text("\\cite{test_author_2026_a_test_paper}", encoding="utf-8")
    (paper_dir / "references.bib").write_text(bibtex, encoding="utf-8")
    (paper_dir / "synthesis_report.json").write_text("{}", encoding="utf-8")
    report = verification.run(output_dir, "project-demo")
    assert report["checks"]
    assert isinstance(report["passed"], bool)
    assert any(check["name"] == "paper_artifacts" for check in report["checks"])


def test_paper_synthesizer_and_domain_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("GENESIS_CACHE_ROOT", str(tmp_path / "cache"))
    project_root = tmp_path / "projects"
    run_dir = project_root / "demo" / "runs" / "1"
    paper_dir = project_root / "demo" / "outputs" / "paper"
    run_dir.mkdir(parents=True)
    paper_dir.mkdir(parents=True)
    (project_root / "demo" / "spec.json").write_text(json.dumps({"research_question": "How does this behave?"}), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"task_id": "task-1", "summary": "summary", "primary_metric": 0.9, "citations": []}), encoding="utf-8")
    (run_dir / "verification_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    result = PaperSynthesizer(project_root).synthesize("demo")
    assert Path(result["latex_path"]).exists()
    assert (paper_dir / "synthesis_report.json").exists()
    assert (paper_dir / "run_index.json").exists()

    provider = DomainKnowledgeRegistry(cache_root=tmp_path / "cache").get_provider("astrophysics")
    summary = provider.initialize({"research_question": "How does this behave?"})
    assert summary


def test_paper_synthesizer_escapes_special_characters_and_uses_richer_fallback(tmp_path):
    project_root = tmp_path / "projects"
    run_dir = project_root / "demo" / "runs" / "1"
    paper_dir = project_root / "demo" / "outputs" / "paper"
    knowledge_dir = project_root / "demo" / "knowledge"
    run_dir.mkdir(parents=True)
    paper_dir.mkdir(parents=True)
    knowledge_dir.mkdir(parents=True)
    (project_root / "demo" / "spec.json").write_text(
        json.dumps({"research_question": "Effect of x & y on z_%", "domain": "general"}),
        encoding="utf-8",
    )
    (knowledge_dir / "domain_context.md").write_text("context with 100% detail & constraints", encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "summary": "Measured x & y with z_% outcome.",
                "primary_metric": 0.9,
                "citations": [],
                "generated_artifacts": ["artifact.txt"],
                "agent_runtime": {"provider": "test", "model": "fake"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "verification_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    result = PaperSynthesizer(project_root).synthesize("demo")
    tex = Path(result["latex_path"]).read_text(encoding="utf-8")
    assert r"\&" in tex
    assert r"\%" in tex
    assert "Domain context used during synthesis" in tex


def test_paper_synthesizer_uses_section_runtime(tmp_path):
    class _Runtime:
        def __init__(self):
            self.calls = []

        def generate_task(self, **kwargs):
            self.calls.append(kwargs["context"]["section"])
            return {"summary": f"{kwargs['context']['section']} section", "paper_body": f"{kwargs['context']['section']} body"}

    project_root = tmp_path / "projects"
    run_dir = project_root / "demo" / "runs" / "1"
    paper_dir = project_root / "demo" / "outputs" / "paper"
    run_dir.mkdir(parents=True)
    paper_dir.mkdir(parents=True)
    (project_root / "demo" / "spec.json").write_text(json.dumps({"research_question": "How does this behave?", "domain": "general"}), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"task_id": "task-1", "summary": "summary", "primary_metric": 0.9, "citations": []}), encoding="utf-8")
    (run_dir / "verification_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    runtime = _Runtime()
    PaperSynthesizer(project_root, runtime=runtime).synthesize("demo")
    assert runtime.calls == ["abstract", "introduction", "methods", "results", "discussion"]
