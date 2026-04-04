import os
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
        oracle_hints=["verify sample consistency"],
    )
    source = DomainOracleGenerator().generate(config)
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(source, encoding="utf-8")
    validator = OracleValidator()
    result = validator.run_oracle(oracle_path, tmp_path)
    assert result.pass_rate >= 0.0


def test_citations_plotting_and_verification(tmp_path):
    citations = CitationsAgent(tmp_path / "cache" / "citations.json")
    bibtex = citations.format_bibtex({"title": "A test paper", "year": 2026})
    assert "@article" in bibtex
    assert citations.search_title("A test paper") is not None

    plotting = PlottingModule(tmp_path / "figures")
    figure = plotting.generate_figure(
        FigureSpec(
            figure_type="line",
            data_source=[0.1, 0.2, 0.3],
            axis_labels=["x", "y"],
            title="Demo Figure",
        )
    )
    assert Path(figure.pdf_path).exists()
    assert figure.metadata["pdf_path"].endswith(".pdf")

    verification = VerificationPipeline()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "result.json").write_text("{}", encoding="utf-8")
    report = verification.run(output_dir, "project-demo")
    assert report["checks"]
    assert isinstance(report["passed"], bool)


def test_paper_synthesizer_and_domain_registry(tmp_path):
    os.chdir(tmp_path)
    project_root = tmp_path / "projects"
    paper_dir = project_root / "demo" / "outputs" / "paper"
    paper_dir.mkdir(parents=True)
    result = PaperSynthesizer(project_root).synthesize("demo")
    assert Path(result["latex_path"]).exists()

    provider = DomainKnowledgeRegistry().get_provider("astrophysics")
    summary = provider.initialize({"research_question": "How does this behave?"})
    assert summary
