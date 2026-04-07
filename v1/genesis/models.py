from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TaskNode:
    task_id: str
    description: str
    acceptance_criteria: list[str]
    oracle_checks: list[str]
    estimated_compute_budget: str
    dependencies: list[str] = field(default_factory=list)
    success_metric: str = ""
    requires_ml_optimizer: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskTree:
    root_id: str
    tasks: list[TaskNode]

    def to_dict(self) -> dict[str, Any]:
        return {"root_id": self.root_id, "tasks": [task.to_dict() for task in self.tasks]}


@dataclass
class OracleResult:
    pass_rate: float
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_critical_fail: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckResult:
    name: str
    passed: bool
    evidence: list[str] = field(default_factory=list)
    score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriteriaFinding:
    criterion: str
    passed: bool
    severity: str = "medium"
    rationale: str = ""
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimFinding:
    claim: str
    classification: str
    rationale: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    why_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiteratureFinding:
    claim: str
    contradicted: bool
    rationale: str = ""
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StoppingDecision:
    should_stop: bool
    reasons: list[str]
    critical_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdversarialReport:
    generated_criteria: list[str] = field(default_factory=list)
    criteria_findings: list[CriteriaFinding] = field(default_factory=list)
    claim_findings: list[ClaimFinding] = field(default_factory=list)
    literature_findings: list[LiteratureFinding] = field(default_factory=list)
    claim_flags: list[str] = field(default_factory=list)
    literature_flags: list[str] = field(default_factory=list)
    formal_checks: list[CheckResult] = field(default_factory=list)
    acceptance_ratio: float = 0.0
    grounded_claims: int = 0
    total_claims: int = 0
    critical_blockers: list[str] = field(default_factory=list)
    iteration_count: int = 0
    task_id: str = ""
    stage: str = ""
    stopping_decision: StoppingDecision = field(
        default_factory=lambda: StoppingDecision(False, ["not evaluated"])
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["criteria_findings"] = [finding.to_dict() for finding in self.criteria_findings]
        payload["claim_findings"] = [finding.to_dict() for finding in self.claim_findings]
        payload["literature_findings"] = [finding.to_dict() for finding in self.literature_findings]
        payload["formal_checks"] = [check.to_dict() for check in self.formal_checks]
        payload["stopping_decision"] = self.stopping_decision.to_dict()
        return payload


@dataclass
class ManifoldHealth:
    status: str
    paper_count: int
    has_embeddings: bool
    has_latent_vectors: bool
    has_density_scores: bool
    citation_edge_count: int
    ready_modes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    experiment_id: str
    task_id: str
    primary_metric: float
    secondary_metrics: dict[str, float]
    trajectory: list[float]
    peak_memory: float
    runtime_seconds: float
    status: str
    code_hash: str
    artifact_path: str
    trajectory_path: str = ""
    anomaly_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentProposal:
    description: str
    code_diff: str
    expected_metric: float
    expected_trajectory: list[float]
    compute_budget: str
    model_parameter_count: int = 0
    command: Optional[Union[str, list[str]]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Idea:
    title: str
    summary: str
    source: str
    landing_point: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IdeaScore:
    novelty: float
    tractability: float
    connection_quality: float
    taste_prediction: float
    composite_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredIdea:
    idea: Idea
    score: IdeaScore

    def to_dict(self) -> dict[str, Any]:
        return {"idea": self.idea.to_dict(), "score": self.score.to_dict()}


@dataclass
class IdeationResult:
    status: str
    health: ManifoldHealth
    ideas: list[ScoredIdea] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "health": self.health.to_dict(),
            "ideas": [idea.to_dict() for idea in self.ideas],
            "reasons": list(self.reasons),
        }


@dataclass
class FigureSpec:
    figure_type: str
    data_source: Union[str, List[float], Dict[str, Any]]
    axis_labels: List[str]
    title: str
    style: str = "publication"


@dataclass
class FigureResult:
    pdf_path: str
    png_path: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectResult:
    project_id: str
    status: str
    paper_path: Optional[str] = None
    run_count: int = 0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
