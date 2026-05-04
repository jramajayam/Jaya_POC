"""
Tests for the Prioritization Judge.

Covers:
  - models: ExposureItem, PrioritizationInput, RankingIssue, PrioritizationJudgment
  - json_parser: extract_json from various formats
  - judge: prompt building, response parsing (no live LLM calls)

Total: 24 tests
"""

import pytest

from prioritization_judge.models import (
    ExposureItem,
    PrioritizationInput,
    PrioritizationJudgment,
    RankingIssue,
)
from prioritization_judge.json_parser import extract_json
from prioritization_judge.judge import PrioritizationJudge


# -- helpers -----------------------------------------------------------------

def _make_item(rank=1, **overrides) -> ExposureItem:
    defaults = dict(
        rank=rank,
        exposure_id=f"EXP-{rank:03d}",
        exposure_type="CVE",
        exploit_feasibility=80.0,
        business_impact=80.0,
        risk_score=80,
        risk_band="HIGH",
        asset_domain="app.example.com",
    )
    defaults.update(overrides)
    return ExposureItem(**defaults)


def _make_input(n=5) -> PrioritizationInput:
    items = [_make_item(rank=i, risk_score=100 - i * 10) for i in range(1, n + 1)]
    return PrioritizationInput(items=items, formula_used="geometric_mean")


# ===========================================================================
#  ExposureItem tests
# ===========================================================================

class TestExposureItem:

    def test_to_summary(self):
        item = _make_item(rank=1, exposure_id="CVE-001", risk_score=90)
        s = item.to_summary()
        assert "#1" in s
        assert "CVE-001" in s
        assert "Risk=90" in s

    def test_fields(self):
        item = _make_item(rank=3, exposure_type="Weak SPF", risk_band="MEDIUM")
        assert item.rank == 3
        assert item.exposure_type == "Weak SPF"
        assert item.risk_band == "MEDIUM"


# ===========================================================================
#  PrioritizationInput tests
# ===========================================================================

class TestPrioritizationInput:

    def test_count(self):
        inp = _make_input(5)
        assert inp.count == 5

    def test_to_table_has_header(self):
        inp = _make_input(3)
        table = inp.to_table()
        assert "Rank" in table
        assert "ID" in table
        assert "Risk" in table

    def test_to_table_has_all_items(self):
        inp = _make_input(4)
        table = inp.to_table()
        for item in inp.items:
            assert item.exposure_id in table

    def test_empty_context(self):
        inp = PrioritizationInput(items=[], formula_used="gm")
        assert inp.context == ""


# ===========================================================================
#  RankingIssue tests
# ===========================================================================

class TestRankingIssue:

    def test_to_dict(self):
        issue = RankingIssue(
            exposure_id="CVE-001",
            current_rank=5,
            suggested_rank=2,
            severity="major",
            reasoning="Critical CVE should be higher",
        )
        d = issue.to_dict()
        assert d["exposure_id"] == "CVE-001"
        assert d["current_rank"] == 5
        assert d["suggested_rank"] == 2
        assert d["severity"] == "major"


# ===========================================================================
#  PrioritizationJudgment tests
# ===========================================================================

class TestPrioritizationJudgment:

    def test_is_correct(self):
        j = PrioritizationJudgment(
            verdict="CORRECT", overall_score=92, summary="Good",
        )
        assert j.is_correct is True
        assert j.is_error is False

    def test_is_error(self):
        j = PrioritizationJudgment.error_result("timeout")
        assert j.is_error is True
        assert j.is_correct is False
        assert j.overall_score == 0
        assert "timeout" in j.error

    def test_major_issues_filter(self):
        issues = [
            RankingIssue("A", 1, 3, "major", "reason"),
            RankingIssue("B", 2, 4, "minor", "reason"),
            RankingIssue("C", 3, 1, "major", "reason"),
        ]
        j = PrioritizationJudgment(
            verdict="NEEDS_REORDERING", overall_score=40,
            summary="Bad", issues=issues,
        )
        assert len(j.major_issues) == 2

    def test_to_dict(self):
        j = PrioritizationJudgment(
            verdict="MOSTLY_CORRECT", overall_score=75,
            summary="OK", strengths=["good CVE ordering"],
            confidence=0.8, reasoning="logic",
        )
        d = j.to_dict()
        assert d["verdict"] == "MOSTLY_CORRECT"
        assert d["overall_score"] == 75
        assert "good CVE ordering" in d["strengths"]


# ===========================================================================
#  JSON parser tests
# ===========================================================================

class TestJsonParser:

    def test_pure_json(self):
        text = '{"verdict": "CORRECT", "overall_score": 90}'
        result = extract_json(text)
        assert result["verdict"] == "CORRECT"

    def test_code_block(self):
        text = '```json\n{"verdict": "MOSTLY_CORRECT"}\n```'
        result = extract_json(text)
        assert result["verdict"] == "MOSTLY_CORRECT"

    def test_embedded_json(self):
        text = 'Here is my evaluation:\n{"verdict": "CORRECT", "overall_score": 85}\nDone.'
        result = extract_json(text)
        assert result["verdict"] == "CORRECT"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            extract_json("")

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            extract_json("This has no JSON at all")


# ===========================================================================
#  Judge prompt building & parsing tests (no live LLM)
# ===========================================================================

class TestJudgeLogic:

    def test_build_prompt_contains_table(self):
        inp = _make_input(3)
        prompt = PrioritizationJudge._build_prompt(inp)
        assert "EXP-001" in prompt
        assert "geometric_mean" in prompt

    def test_build_prompt_with_context(self):
        inp = PrioritizationInput(
            items=[_make_item()],
            formula_used="gm",
            context="Test environment",
        )
        prompt = PrioritizationJudge._build_prompt(inp)
        assert "Test environment" in prompt

    def test_build_messages_structure(self):
        inp = _make_input(2)
        prompt = PrioritizationJudge._build_prompt(inp)
        messages = PrioritizationJudge._build_messages(prompt)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_parse_judgment_full(self):
        parsed = {
            "verdict": "MOSTLY_CORRECT",
            "overall_score": 78,
            "summary": "Good ordering with minor issues",
            "strengths": ["CVEs ranked first", "Weights applied correctly"],
            "issues": [
                {
                    "exposure_id": "SPF-001",
                    "current_rank": 3,
                    "suggested_rank": 5,
                    "severity": "minor",
                    "reasoning": "SPF should be lower than DMARC",
                }
            ],
            "confidence": 0.85,
            "reasoning": "Overall the list follows expected patterns",
        }
        j = PrioritizationJudge._parse_judgment(parsed)
        assert j.verdict == "MOSTLY_CORRECT"
        assert j.overall_score == 78
        assert len(j.issues) == 1
        assert j.issues[0].severity == "minor"
        assert len(j.strengths) == 2
        assert j.confidence == 0.85

    def test_parse_judgment_minimal(self):
        parsed = {"verdict": "CORRECT", "overall_score": 95}
        j = PrioritizationJudge._parse_judgment(parsed)
        assert j.verdict == "CORRECT"
        assert j.issues == []
        assert j.strengths == []
