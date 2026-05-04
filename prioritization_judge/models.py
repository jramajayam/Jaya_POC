"""
Data models for the Prioritization Judge.

ExposureItem         - one exposure in the prioritized list.
PrioritizationInput  - the full ordered list to evaluate.
RankingIssue         - one specific ordering problem found by the LLM.
PrioritizationJudgment - the LLM's overall evaluation of the ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExposureItem:
    """A single exposure in the prioritization list."""

    rank: int
    exposure_id: str
    exposure_type: str
    exploit_feasibility: float
    business_impact: float
    risk_score: int
    risk_band: str
    asset_domain: str
    description: str = ""

    def to_summary(self) -> str:
        """One-line summary for the LLM prompt."""
        return (
            f"#{self.rank} | {self.exposure_id} | {self.exposure_type} | "
            f"EF={self.exploit_feasibility:.0f} | BI={self.business_impact:.0f} | "
            f"Risk={self.risk_score} ({self.risk_band}) | {self.asset_domain}"
        )


@dataclass
class RankingIssue:
    """A specific ordering problem identified by the LLM."""

    exposure_id: str
    current_rank: int
    suggested_rank: int
    severity: str  # "major" | "minor" | "nitpick"
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "exposure_id": self.exposure_id,
            "current_rank": self.current_rank,
            "suggested_rank": self.suggested_rank,
            "severity": self.severity,
            "reasoning": self.reasoning,
        }


@dataclass
class PrioritizationInput:
    """The full prioritized list to evaluate."""

    items: List[ExposureItem]
    formula_used: str = "geometric_mean"
    context: str = ""  # optional additional context for the LLM

    @property
    def count(self) -> int:
        return len(self.items)

    def to_table(self) -> str:
        """Format items as a readable table for the LLM."""
        header = (
            f"{'Rank':<5} {'ID':<18} {'Type':<20} {'EF':>4} {'BI':>4} "
            f"{'Risk':>5} {'Band':<10} {'Domain'}"
        )
        lines = [header, "-" * 90]
        for item in self.items:
            lines.append(
                f"{item.rank:<5} {item.exposure_id:<18} {item.exposure_type:<20} "
                f"{item.exploit_feasibility:>4.0f} {item.business_impact:>4.0f} "
                f"{item.risk_score:>5} {item.risk_band:<10} {item.asset_domain}"
            )
        return "\n".join(lines)


@dataclass
class PrioritizationJudgment:
    """The LLM's evaluation of the prioritization ordering."""

    verdict: str  # "CORRECT" | "MOSTLY_CORRECT" | "NEEDS_REORDERING"
    overall_score: int  # 0-100 quality score for the ordering
    summary: str  # one-paragraph assessment
    issues: List[RankingIssue] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    error: Optional[str] = None

    @property
    def is_correct(self) -> bool:
        return self.verdict == "CORRECT"

    @property
    def is_error(self) -> bool:
        return self.verdict == "ERROR"

    @property
    def major_issues(self) -> List[RankingIssue]:
        return [i for i in self.issues if i.severity == "major"]

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "issues_count": len(self.issues),
            "major_issues": len(self.major_issues),
            "strengths": "; ".join(self.strengths),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "error": self.error or "",
        }

    @staticmethod
    def error_result(error_msg: str) -> "PrioritizationJudgment":
        return PrioritizationJudgment(
            verdict="ERROR",
            overall_score=0,
            summary=f"Evaluation failed: {error_msg}",
            confidence=0.0,
            reasoning="",
            error=error_msg,
        )
