"""
Data models for the LLM Judge.

AssetInput  — the 5 asset fields sent to the LLM + MLP predictions for comparison.
JudgmentResult — structured output (LLM classification + comparison with MLP).
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AssetInput:
    """Input to the LLM judge.

    The 5 asset fields (domain, ports, technologies, banner, dns_count) are
    sent to the LLM for independent classification.

    ``mlp_categories`` come from a separate MLP run and are used ONLY in
    Python-side comparison — they are never sent to the LLM.
    """

    asset_id: str
    domain: str
    ports: List[str]
    technologies: List[str]
    banner: str
    dns_count: int
    mlp_categories: List[str]      # from separate MLP run (NOT sent to LLM)
    valid_categories: List[str]


@dataclass
class JudgmentResult:
    """Result of LLM judgment on a single asset."""

    asset_id: str
    domain: str
    llm_categories: List[str]
    judgment: str  # CORRECT | PARTIALLY_CORRECT | INCORRECT
    correct_predictions: List[str]
    missing_categories: List[str]
    wrong_categories: List[str]
    confidence: float
    confidence_reasoning: str
    reasoning: str
    error: Optional[str] = None

    # ── convenience helpers ────────────────────────────────────────────

    @property
    def is_correct(self) -> bool:
        return self.judgment == "CORRECT"

    @property
    def is_error(self) -> bool:
        return self.judgment == "ERROR"

    def to_dict(self) -> dict:
        """Serialise to a flat dictionary (e.g. for CSV / DataFrame)."""
        return {
            "asset_id": self.asset_id,
            "domain": self.domain,
            "llm_categories": ", ".join(self.llm_categories),
            "judgment": self.judgment,
            "correct_predictions": ", ".join(self.correct_predictions),
            "missing_categories": ", ".join(self.missing_categories),
            "wrong_categories": ", ".join(self.wrong_categories),
            "confidence": self.confidence,
            "confidence_reasoning": self.confidence_reasoning,
            "reasoning": self.reasoning,
            "error": self.error or "",
        }

    @staticmethod
    def error_result(
        asset_id: str,
        domain: str,
        error_msg: str,
    ) -> "JudgmentResult":
        """Factory for an error-state result."""
        return JudgmentResult(
            asset_id=asset_id,
            domain=domain,
            llm_categories=[],
            judgment="ERROR",
            correct_predictions=[],
            missing_categories=[],
            wrong_categories=[],
            confidence=0.0,
            confidence_reasoning="Error occurred during LLM call",
            reasoning="",
            error=error_msg,
        )
