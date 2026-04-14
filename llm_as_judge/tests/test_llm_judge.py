"""
Tests for the llm_judge package.

Covers:
  • json_parser.extract_json — edge cases
  • models — AssetInput, JudgmentResult, error_result, to_dict
  • judge._compare — deterministic comparison logic (no LLM)
  • judge.judge — end-to-end with mocked OpenAI client
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from llm_judge.json_parser import extract_json
from llm_judge.models import AssetInput, JudgmentResult
from llm_judge.judge import LLMJudge


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_asset(**overrides) -> AssetInput:
    """Build a default AssetInput, overriding specific fields."""
    defaults = dict(
        asset_id="test-001",
        domain="api.example.com",
        ports=["80", "443"],
        technologies=["nginx"],
        banner="HTTP/1.1 200 OK\r\nServer: nginx",
        dns_count=100,
        mlp_categories=["web server"],
        valid_categories=["web server", "api endpoint", "mail server",
                          "dns server", "content delivery network"],
    )
    defaults.update(overrides)
    return AssetInput(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
#  json_parser tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractJson:
    """Tests for extract_json."""

    def test_clean_json(self):
        raw = '{"your_categories": ["web server"], "confidence": 0.9}'
        result = extract_json(raw)
        assert result["your_categories"] == ["web server"]
        assert result["confidence"] == 0.9

    def test_markdown_fenced(self):
        raw = '```json\n{"your_categories": ["mail server"]}\n```'
        result = extract_json(raw)
        assert result["your_categories"] == ["mail server"]

    def test_reasoning_before_json(self):
        raw = (
            "Let me think about this asset.\n\n"
            '{"your_categories": ["dns server"], "confidence": 0.7, '
            '"confidence_reasoning": "port 53", "reasoning": "DNS"}'
        )
        result = extract_json(raw)
        assert result["your_categories"] == ["dns server"]

    def test_trailing_comma_fix(self):
        raw = '{"your_categories": ["web server",], "confidence": 0.8,}'
        result = extract_json(raw)
        assert result["your_categories"] == ["web server"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty response"):
            extract_json("")

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("This is just plain text with no braces")


# ═══════════════════════════════════════════════════════════════════════════════
#  models tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModels:
    """Tests for AssetInput and JudgmentResult."""

    def test_asset_input_fields(self):
        asset = _make_asset()
        assert asset.domain == "api.example.com"
        assert asset.ports == ["80", "443"]
        assert asset.mlp_categories == ["web server"]

    def test_judgment_result_is_correct(self):
        jr = JudgmentResult(
            asset_id="1", domain="x.com", llm_categories=["web server"],
            judgment="CORRECT", correct_predictions=["web server"],
            missing_categories=[], wrong_categories=[],
            confidence=0.9, confidence_reasoning="clear",
            reasoning="ok",
        )
        assert jr.is_correct is True
        assert jr.is_error is False

    def test_judgment_result_to_dict(self):
        jr = JudgmentResult(
            asset_id="1", domain="x.com", llm_categories=["web server"],
            judgment="CORRECT", correct_predictions=["web server"],
            missing_categories=[], wrong_categories=[],
            confidence=0.9, confidence_reasoning="clear",
            reasoning="ok",
        )
        d = jr.to_dict()
        assert d["judgment"] == "CORRECT"
        assert d["llm_categories"] == "web server"
        assert d["error"] == ""

    def test_error_result(self):
        jr = JudgmentResult.error_result("1", "x.com", "timeout")
        assert jr.is_error is True
        assert jr.judgment == "ERROR"
        assert jr.error == "timeout"
        assert jr.llm_categories == []


# ═══════════════════════════════════════════════════════════════════════════════
#  comparison logic tests (no LLM needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompare:
    """Tests for LLMJudge._compare (deterministic Python logic)."""

    def test_exact_match_correct(self):
        """MLP and LLM agree exactly → CORRECT."""
        asset = _make_asset(mlp_categories=["web server"])
        result = LLMJudge._compare(
            asset=asset,
            llm_categories=["web server"],
            confidence=0.9,
            confidence_reasoning="clear",
            reasoning="match",
        )
        assert result.judgment == "CORRECT"
        assert result.correct_predictions == ["web server"]
        assert result.missing_categories == []
        assert result.wrong_categories == []

    def test_subsumption_mlp_webserver_llm_cdn(self):
        """MLP: web server, LLM: content delivery network → CORRECT.
        (web server subsumes CDN)."""
        asset = _make_asset(mlp_categories=["web server"])
        result = LLMJudge._compare(
            asset=asset,
            llm_categories=["content delivery network"],
            confidence=0.85,
            confidence_reasoning="CDN",
            reasoning="CDN node",
        )
        assert result.judgment == "CORRECT"
        assert "web server" in result.correct_predictions

    def test_subsumption_mlp_webserver_llm_api(self):
        """MLP: web server, LLM: api endpoint → CORRECT (subsumed)."""
        asset = _make_asset(mlp_categories=["web server"])
        result = LLMJudge._compare(
            asset=asset,
            llm_categories=["api endpoint"],
            confidence=0.8,
            confidence_reasoning="api",
            reasoning="API host",
        )
        assert result.judgment == "CORRECT"

    def test_missing_different_attack_surface(self):
        """MLP: web server only, LLM: web server + mail server →
        PARTIALLY_CORRECT (mail is a fundamentally different surface)."""
        asset = _make_asset(mlp_categories=["web server"])
        result = LLMJudge._compare(
            asset=asset,
            llm_categories=["web server", "mail server"],
            confidence=0.8,
            confidence_reasoning="multi-service",
            reasoning="both HTTP and SMTP",
        )
        assert result.judgment == "PARTIALLY_CORRECT"
        assert "mail server" in result.missing_categories
        assert "web server" in result.correct_predictions

    def test_completely_wrong(self):
        """MLP: dns server, LLM: mail server → INCORRECT."""
        asset = _make_asset(mlp_categories=["dns server"])
        result = LLMJudge._compare(
            asset=asset,
            llm_categories=["mail server"],
            confidence=0.7,
            confidence_reasoning="SMTP",
            reasoning="port 25",
        )
        assert result.judgment == "INCORRECT"
        assert "dns server" in result.wrong_categories
        assert "mail server" in result.missing_categories

    def test_multi_category_partial(self):
        """MLP: web server + dns server, LLM: web server + mail server →
        PARTIALLY_CORRECT."""
        asset = _make_asset(mlp_categories=["web server", "dns server"])
        result = LLMJudge._compare(
            asset=asset,
            llm_categories=["web server", "mail server"],
            confidence=0.75,
            confidence_reasoning="mixed",
            reasoning="complex",
        )
        assert result.judgment == "PARTIALLY_CORRECT"
        assert "web server" in result.correct_predictions
        assert "dns server" in result.wrong_categories
        assert "mail server" in result.missing_categories

    def test_multi_category_all_correct(self):
        """MLP and LLM agree on multiple categories → CORRECT."""
        asset = _make_asset(
            mlp_categories=["web server", "mail server"]
        )
        result = LLMJudge._compare(
            asset=asset,
            llm_categories=["web server", "mail server"],
            confidence=0.95,
            confidence_reasoning="clear",
            reasoning="both",
        )
        assert result.judgment == "CORRECT"
        assert set(result.correct_predictions) == {"web server", "mail server"}


# ═══════════════════════════════════════════════════════════════════════════════
#  end-to-end judge test (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestJudgeEndToEnd:
    """Tests LLMJudge.judge with a mocked OpenAI client."""

    @patch("llm_judge.judge.OpenAI")
    def test_judge_correct(self, MockOpenAI):
        """Happy path: LLM returns valid JSON, categories match."""
        mock_client = MagicMock()
        mock_client.get_openai_chat.return_value = (
            json.dumps({
                "your_categories": ["web server"],
                "confidence": 0.92,
                "confidence_reasoning": "HTTP on 80/443",
                "reasoning": "serves HTTP content",
            }),
            {},  # raw response
            {},  # usage
        )
        MockOpenAI.return_value = mock_client

        judge = LLMJudge()
        asset = _make_asset(mlp_categories=["web server"])
        result = judge.judge(asset)

        assert result.judgment == "CORRECT"
        assert result.llm_categories == ["web server"]
        assert result.confidence == 0.92

        # Verify only 5 asset fields were in the prompt (no mlp_categories)
        call_args = mock_client.get_openai_chat.call_args[0][0]
        prompt_text = call_args[0]["content"]
        assert "api.example.com" in prompt_text       # domain ✓
        assert "80, 443" in prompt_text                # ports ✓
        assert "nginx" in prompt_text                  # technologies ✓
        assert "mlp" not in prompt_text.lower().split("---")[-1]  # no MLP in classify prompt

    @patch("llm_judge.judge.OpenAI")
    def test_judge_llm_returns_none(self, MockOpenAI):
        """LLM returns None → error result."""
        mock_client = MagicMock()
        mock_client.get_openai_chat.return_value = (None, None, None)
        MockOpenAI.return_value = mock_client

        judge = LLMJudge(max_retries=1)
        result = judge.judge(_make_asset())

        assert result.is_error
        assert "None" in result.error

    @patch("llm_judge.judge.OpenAI")
    def test_judge_llm_bad_json(self, MockOpenAI):
        """LLM returns garbage → error result after retries."""
        mock_client = MagicMock()
        mock_client.get_openai_chat.return_value = ("not json at all", {}, {})
        MockOpenAI.return_value = mock_client

        judge = LLMJudge(max_retries=1, backoff_factor=0)
        result = judge.judge(_make_asset())

        assert result.is_error

    @patch("llm_judge.judge.OpenAI")
    def test_judge_partially_correct(self, MockOpenAI):
        """LLM adds a category MLP missed → PARTIALLY_CORRECT."""
        mock_client = MagicMock()
        mock_client.get_openai_chat.return_value = (
            json.dumps({
                "your_categories": ["web server", "mail server"],
                "confidence": 0.8,
                "confidence_reasoning": "HTTP + SMTP",
                "reasoning": "both services",
            }),
            {},
            {},
        )
        MockOpenAI.return_value = mock_client

        judge = LLMJudge()
        asset = _make_asset(mlp_categories=["web server"])
        result = judge.judge(asset)

        assert result.judgment == "PARTIALLY_CORRECT"
        assert "mail server" in result.missing_categories


# ═══════════════════════════════════════════════════════════════════════════════
#  prompt building tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptBuilding:
    """Verify the prompt only contains the 5 asset fields."""

    def test_prompt_has_five_fields(self):
        asset = _make_asset()
        prompt = LLMJudge._build_prompt(asset)

        assert "api.example.com" in prompt         # domain
        assert "80, 443" in prompt                  # ports
        assert "nginx" in prompt                    # technologies
        assert "HTTP/1.1 200 OK" in prompt          # banner
        assert "100" in prompt                      # dns_count

    def test_prompt_excludes_mlp_categories(self):
        asset = _make_asset(mlp_categories=["web server", "api endpoint"])
        prompt = LLMJudge._build_prompt(asset)

        # The classification prompt must NOT contain MLP predictions
        assert "ML MODEL PREDICTED" not in prompt
        assert "mlp" not in prompt.lower()

    def test_prompt_empty_fields(self):
        asset = _make_asset(ports=[], technologies=[], banner="")
        prompt = LLMJudge._build_prompt(asset)

        assert "none" in prompt  # empty fields → "none"
