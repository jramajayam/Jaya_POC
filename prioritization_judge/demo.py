"""
Demo: Compute risk scores, then have the LLM judge the prioritization order.

Usage:
    python -m prioritization_judge.demo

Requires AZURE_OPENAI_API_KEY in .env
"""

from risk_score import compute_risk_batch, RiskInput
from prioritization_judge import (
    PrioritizationJudge,
    PrioritizationInput,
    ExposureItem,
)


def main():
    # -- Step 1: Compute risk scores for a realistic set of exposures --
    inputs = [
        RiskInput("CVE-2024-1234",   "CVE",               90, 90, "payments.acme.com"),
        RiskInput("CVE-2024-5678",   "CVE",               70, 70, "blog.acme.com"),
        RiskInput("CNAME-dangling",  "Dangling CNAME",    90, 90, "old-app.acme.com"),
        RiskInput("LAME-001",        "Lame Delegation",   85, 85, "acme.com"),
        RiskInput("AXFR-leak",       "Open Zone Transfer", 80, 80, "ns1.acme.com"),
        RiskInput("DMARC-none",      "Weak DMARC",        85, 75, "acme.com"),
        RiskInput("SPF-permissive",  "Weak SPF",          80, 80, "acme.com"),
        RiskInput("EXP-domain",      "Expiring Domain",   90, 90, "legacy.acme.com"),
        RiskInput("CVE-2024-9999",   "CVE",               90, 20, "internal.acme.com"),
        RiskInput("CVE-2024-0001",   "CVE",               20, 90, "payments.acme.com"),
    ]

    results = compute_risk_batch(inputs, formula="geometric_mean")

    # -- Step 2: Convert to ExposureItems for the judge --
    items = []
    for rank, r in enumerate(results, 1):
        items.append(
            ExposureItem(
                rank=rank,
                exposure_id=r.exposure_id,
                exposure_type=r.exposure_type,
                exploit_feasibility=r.exploit_feasibility,
                business_impact=r.business_impact,
                risk_score=r.risk_score,
                risk_band=r.risk_band,
                asset_domain=r.asset_domain,
            )
        )

    prioritization = PrioritizationInput(
        items=items,
        formula_used="geometric_mean: sqrt(EF x exposure_weight x BI)",
        context="10 exposures across a corporate attack surface. Mix of critical CVEs, DNS issues, and email security gaps.",
    )

    # -- Step 3: Show the list --
    print("=" * 95)
    print("PRIORITIZED LIST (by risk score - geometric mean)".center(95))
    print("=" * 95)
    print(prioritization.to_table())
    print("=" * 95)
    print()

    # -- Step 4: Ask the LLM to judge --
    print("Sending to LLM for evaluation...")
    print()

    judge = PrioritizationJudge()
    judgment = judge.evaluate(prioritization)

    # -- Step 5: Print results --
    print("=" * 95)
    print("LLM PRIORITIZATION JUDGMENT".center(95))
    print("=" * 95)
    print(f"  Verdict:    {judgment.verdict}")
    print(f"  Score:      {judgment.overall_score}/100")
    print(f"  Confidence: {judgment.confidence:.0%}")
    print()
    print(f"  Summary: {judgment.summary}")
    print()

    if judgment.strengths:
        print("  Strengths:")
        for s in judgment.strengths:
            print(f"    + {s}")
        print()

    if judgment.issues:
        print(f"  Issues ({len(judgment.issues)}):")
        for issue in judgment.issues:
            print(f"    [{issue.severity.upper()}] {issue.exposure_id}: "
                  f"rank #{issue.current_rank} -> suggested #{issue.suggested_rank}")
            print(f"      {issue.reasoning}")
        print()

    if judgment.reasoning:
        print(f"  Reasoning: {judgment.reasoning}")

    print("=" * 95)


if __name__ == "__main__":
    main()
