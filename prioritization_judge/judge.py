"""
PrioritizationJudge - core class that evaluates a risk-score-ordered list.

Flow:
  1. Format the prioritized list into a readable table
  2. Send to the LLM with evaluation prompt
  3. Parse the LLM's judgment (verdict, issues, strengths)
  4. Return PrioritizationJudgment

Uses:
  - prioritization_judge.prompts    -> SYSTEM_PROMPT, EVALUATE_PROMPT_TEMPLATE
  - openai_model_manager            -> OpenAI.get_openai_chat()
"""

from __future__ import annotations

import time
from typing import List, Optional

from prioritization_judge.prompts import SYSTEM_PROMPT, EVALUATE_PROMPT_TEMPLATE
from prioritization_judge.models import (
    PrioritizationInput,
    ExposureItem,
    PrioritizationJudgment,
    RankingIssue,
)
from prioritization_judge.json_parser import extract_json
from openai_model_manager import OpenAI


class PrioritizationJudge:
    """Evaluates a risk-score prioritized list using an LLM.

    The LLM judges whether the ordering is sensible - it does NOT
    recompute scores, just evaluates if the list makes sense for
    a security team to act on.

    Parameters
    ----------
    max_retries : int
        How many times to retry the LLM call on failure (default 3).
    backoff_factor : int
        Seconds between retries (default 2).
    """

    def __init__(self, max_retries: int = 3, backoff_factor: int = 2) -> None:
        self._client = OpenAI()
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor

    # -- public API ---------------------------------------------------------

    def evaluate(self, inp: PrioritizationInput) -> PrioritizationJudgment:
        """Evaluate the prioritization ordering via LLM.

        Parameters
        ----------
        inp : PrioritizationInput
            The ordered list of exposures to evaluate.

        Returns
        -------
        PrioritizationJudgment
            The LLM's evaluation with verdict, score, issues, and strengths.
        """
        prompt = self._build_prompt(inp)
        messages = self._build_messages(prompt)

        for attempt in range(self._max_retries):
            try:
                content, _raw, _usage = self._client.get_openai_chat(messages)
                if content is None:
                    raise RuntimeError("LLM returned None - check API key / network")

                parsed = extract_json(content)
                return self._parse_judgment(parsed)

            except Exception as exc:
                if attempt < self._max_retries - 1:
                    time.sleep(self._backoff_factor * (attempt + 1))
                    continue
                return PrioritizationJudgment.error_result(str(exc))

        return PrioritizationJudgment.error_result("Exhausted retries")

    def evaluate_batch(
        self, inputs: List[PrioritizationInput]
    ) -> List[PrioritizationJudgment]:
        """Evaluate multiple prioritization lists."""
        return [self.evaluate(inp) for inp in inputs]

    # -- prompt building ----------------------------------------------------

    @staticmethod
    def _build_prompt(inp: PrioritizationInput) -> str:
        """Format the evaluation prompt with the prioritized list."""
        context_section = ""
        if inp.context:
            context_section = f"Additional context: {inp.context}"

        return EVALUATE_PROMPT_TEMPLATE.format(
            formula=inp.formula_used,
            context=context_section,
            table=inp.to_table(),
        )

    @staticmethod
    def _build_messages(prompt: str) -> list:
        """Build the messages array for the LLM API call."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    # -- response parsing ---------------------------------------------------

    @staticmethod
    def _parse_judgment(parsed: dict) -> PrioritizationJudgment:
        """Parse the JSON response into a PrioritizationJudgment."""
        issues = []
        for issue_data in parsed.get("issues", []):
            issues.append(
                RankingIssue(
                    exposure_id=issue_data.get("exposure_id", "unknown"),
                    current_rank=int(issue_data.get("current_rank", 0)),
                    suggested_rank=int(issue_data.get("suggested_rank", 0)),
                    severity=issue_data.get("severity", "minor"),
                    reasoning=issue_data.get("reasoning", ""),
                )
            )

        return PrioritizationJudgment(
            verdict=parsed.get("verdict", "NEEDS_REORDERING"),
            overall_score=int(parsed.get("overall_score", 0)),
            summary=parsed.get("summary", ""),
            issues=issues,
            strengths=parsed.get("strengths", []),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=parsed.get("reasoning", ""),
        )
