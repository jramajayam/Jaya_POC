"""
LLMJudge — core class that evaluates one asset.

Flow:
  1. Send only 5 asset fields to the LLM  → LLM classifies independently
  2. Compare LLM categories vs MLP categories in Python
  3. Return JudgmentResult

Uses:
  • llm_judge.prompts            → CLASSIFY_SYSTEM_PROMPT, CLASSIFY_PROMPT_TEMPLATE
  • llm_judge_prompts            → WEB_SERVER_SUBSUMES
  • openai_model_manager         → OpenAI.get_openai_chat()
"""

from __future__ import annotations

import time
from typing import List, Set

from llm_judge.prompts import CLASSIFY_SYSTEM_PROMPT, CLASSIFY_PROMPT_TEMPLATE, WEB_SERVER_SUBSUMES
from openai_model_manager import OpenAI

from llm_judge.models import AssetInput, JudgmentResult
from llm_judge.json_parser import extract_json


_WEB_SUBSUMES_SET: Set[str] = set(WEB_SERVER_SUBSUMES)


class LLMJudge:
    """Sends asset details to the LLM, gets independent classification,
    then compares with MLP predictions in Python.

    Parameters
    ----------
    max_retries : int
        How many times to retry the LLM call on failure (default 3).
    backoff_factor : int
        Seconds multiplier between retries (default 2).
    """

    def __init__(self, max_retries: int = 3, backoff_factor: int = 2) -> None:
        self._client = OpenAI()
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor

    # ── public API ────────────────────────────────────────────────────

    def judge(self, asset: AssetInput) -> JudgmentResult:
        """Classify asset via LLM, compare with MLP, return judgment."""
        prompt = self._build_prompt(asset)
        messages = self._build_messages(prompt)

        for attempt in range(self._max_retries):
            try:
                content, _raw, _usage = self._client.get_openai_chat(messages)
                if content is None:
                    raise RuntimeError("LLM returned None — check API key / network")

                parsed = extract_json(content)
                llm_categories = [
                    c.lower().strip() for c in parsed.get("your_categories", [])
                ]
                confidence = float(parsed.get("confidence", 0.5))
                confidence_reasoning = parsed.get("confidence_reasoning", "")
                reasoning = parsed.get("reasoning", "")

                return self._compare(
                    asset=asset,
                    llm_categories=llm_categories,
                    confidence=confidence,
                    confidence_reasoning=confidence_reasoning,
                    reasoning=reasoning,
                )

            except Exception as exc:
                if attempt < self._max_retries - 1:
                    time.sleep(self._backoff_factor * (attempt + 1))
                    continue
                return JudgmentResult.error_result(
                    asset_id=asset.asset_id,
                    domain=asset.domain,
                    error_msg=str(exc),
                )

        # safety net
        return JudgmentResult.error_result(
            asset_id=asset.asset_id,
            domain=asset.domain,
            error_msg="Exhausted retries",
        )

    # ── prompt building (only 5 asset fields) ─────────────────────────

    @staticmethod
    def _build_prompt(asset: AssetInput) -> str:
        """Format classification prompt with the 5 asset fields only."""
        return CLASSIFY_PROMPT_TEMPLATE.format(
            domain=asset.domain,
            ports=", ".join(asset.ports[:15]) if asset.ports else "none",
            technologies=(
                ", ".join(asset.technologies[:10])
                if asset.technologies
                else "none"
            ),
            banner=asset.banner[:300] if asset.banner else "none",
            dns_count=asset.dns_count,
            valid_categories=", ".join(asset.valid_categories),
        )

    @staticmethod
    def _build_messages(prompt: str) -> list[dict]:
        """Combine system + user prompt into a single message (o1-compatible)."""
        combined = f"{CLASSIFY_SYSTEM_PROMPT}\n\n---\n\n{prompt}"
        return [{"role": "user", "content": combined}]

    # ── comparison logic (Python, not LLM) ────────────────────────────

    @staticmethod
    def _compare(
        asset: AssetInput,
        llm_categories: List[str],
        confidence: float,
        confidence_reasoning: str,
        reasoning: str,
    ) -> JudgmentResult:
        """Compare MLP predictions against LLM classification.

        Subsumption rule: if MLP predicted "web server" and LLM says the
        asset is a specialisation (e.g. "cdn"), that counts as correct —
        "web server" subsumes those categories.
        """
        mlp_set = {c.lower().strip() for c in asset.mlp_categories}
        llm_set = set(llm_categories)

        # ── correct: MLP categories the LLM agrees with ──
        correct: List[str] = []
        for cat in asset.mlp_categories:
            cl = cat.lower().strip()
            if cl in llm_set:
                correct.append(cl)
            # subsumption: MLP said "web server", LLM said a specialisation
            elif cl == "web server" and llm_set & _WEB_SUBSUMES_SET:
                correct.append(cl)

        # ── wrong: MLP predicted but LLM does not agree ──
        wrong = [c for c in mlp_set if c not in {x.lower() for x in correct}]

        # ── missing: LLM categories that MLP missed ──
        missing: List[str] = []
        for cat in llm_categories:
            if cat in mlp_set:
                continue
            # subsumption: LLM said specialisation, MLP said "web server"
            if cat in _WEB_SUBSUMES_SET and "web server" in mlp_set:
                continue
            missing.append(cat)

        # ── judgment ──
        if not wrong and not missing:
            judgment = "CORRECT"
        elif correct and (wrong or missing):
            judgment = "PARTIALLY_CORRECT"
        else:
            judgment = "INCORRECT"

        return JudgmentResult(
            asset_id=asset.asset_id,
            domain=asset.domain,
            llm_categories=llm_categories,
            judgment=judgment,
            correct_predictions=correct,
            missing_categories=missing,
            wrong_categories=wrong,
            confidence=confidence,
            confidence_reasoning=confidence_reasoning,
            reasoning=reasoning,
        )
