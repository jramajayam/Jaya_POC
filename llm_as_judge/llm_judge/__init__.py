"""
LLM-as-Judge: Asset Category Classifier Evaluation
====================================================

Sends 5 asset fields to the LLM for independent classification,
then compares with MLP predictions in Python.

Input to LLM:  domain, ports, technologies, banner, dns_count
Output:         JudgmentResult (llm_categories, judgment, confidence, …)

Usage:
    from llm_judge import AssetInput, JudgmentResult, LLMJudge

    judge = LLMJudge()
    result = judge.judge(AssetInput(
        asset_id="123",
        domain="api.example.com",
        ports=["80", "443"],
        technologies=["nginx"],
        banner="HTTP/1.1 200 OK",
        dns_count=150,
        mlp_categories=["web server", "api endpoint"],   # from separate MLP
        valid_categories=["web server", "api endpoint", "mail server"],
    ))
    print(result.judgment)   # CORRECT / PARTIALLY_CORRECT / INCORRECT
"""

from llm_judge.models import AssetInput, JudgmentResult
from llm_judge.judge import LLMJudge
from llm_judge.json_parser import extract_json

__all__ = ["AssetInput", "JudgmentResult", "LLMJudge", "extract_json"]
