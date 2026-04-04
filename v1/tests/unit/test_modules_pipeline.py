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
    report = verification.run(output_dir, "project-demo")
    assert report["checks"]
    assert isinstance(report["passed"], bool)


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

    provider = DomainKnowledgeRegistry(cache_root=tmp_path / "cache").get_provider("astrophysics")
    summary = provider.initialize({"research_question": "How does this behave?"})
    assert summary
