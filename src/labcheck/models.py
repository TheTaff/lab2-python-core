from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Submission:
    submission_id: str
    assignment_id: str
    student_id: str
    attempt: int
    family_id: str
    code: str
    is_template: bool = False


@dataclass(frozen=True)
class EvidenceSpan:
    start_line: int
    end_line: int
    kind: str  # "token_3gram" | "ast_node" | "template_overlap"
    detail: str


@dataclass
class CodeFeatures:
    submission_id: str
    assignment_id: str
    student_id: str
    attempt: int
    token_ngrams: set[tuple[str, ...]]
    template_ngrams: set[tuple[str, ...]]
    token_line_spans: dict[tuple[str, ...], list[int]]
    ast_ngrams: set[tuple[str, ...]] | None
    ast_error: str | None = None
    fallback_used: bool = False
    is_empty: bool = False


@dataclass
class PairComparison:
    assignment_id: str
    left_id: str
    right_id: str
    left_student: str
    right_student: str
    same_student_reupload: bool
    method: str
    score: float | None
    reason: str | None = None
    template_only_overlap: bool = False
    fallback_used: bool = False
    review_required: bool = False
    evidence: list[EvidenceSpan] = field(default_factory=list)
