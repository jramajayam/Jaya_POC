"""
Prompts for the Prioritization Judge.

The LLM evaluates a risk-score-ordered list of exposures and judges
whether the ordering makes sense from a security practitioner's perspective.
"""

SYSTEM_PROMPT = """You are a senior security analyst evaluating a prioritized list of attack surface exposures.

Your job is to judge whether the ORDERING of the list makes sense - i.e., whether the most critical findings are at the top and least critical at the bottom.

You understand:
- Exposure types: CVE, Dangling CNAME, Lame Delegation, Open Zone Transfer, Weak DMARC, Weak SPF, Expiring Domain
- Risk scoring: combines Exploit Feasibility (EF) and Business Impact (BI) using geometric mean with exposure-type weights
- A CVE with high EF and high BI should rank above a Weak SPF with the same raw scores (because CVE = direct exploitation)
- Asymmetric cases: high EF + low BI should rank lower than balanced high scores

You are NOT recomputing scores. You are evaluating whether the prioritization ORDER is sensible for a security team to act on."""


EVALUATE_PROMPT_TEMPLATE = """Evaluate the following prioritized exposure list.

Formula used: {formula}
Formula explanation: Risk Score = sqrt(Exploit_Feasibility x Exposure_Weight x Business_Impact)

Exposure weights by type:
- CVE: 1.0 (direct exploitation)
- Lame Delegation: 1.0 (full zone takeover)
- Dangling CNAME: 0.90 (subdomain takeover)
- Open Zone Transfer: 0.50 (info disclosure)
- Weak DMARC: 0.40 (spoofing enabler)
- Weak SPF: 0.30 (spoofing enabler)
- Expiring Domain: 0.20 (risk indicator)

{context}

PRIORITIZED LIST (highest risk first):
{table}

---

Evaluate this ordering and respond in EXACTLY this JSON format:
{{
    "verdict": "CORRECT" | "MOSTLY_CORRECT" | "NEEDS_REORDERING",
    "overall_score": <0-100 quality score for the ordering>,
    "summary": "<one paragraph assessment of the ordering quality>",
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "issues": [
        {{
            "exposure_id": "<id of mis-ranked item>",
            "current_rank": <current position>,
            "suggested_rank": <where it should be>,
            "severity": "major" | "minor" | "nitpick",
            "reasoning": "<why this is mis-ranked>"
        }}
    ],
    "confidence": <0.0-1.0 how confident you are in this evaluation>,
    "reasoning": "<your overall reasoning process>"
}}

Rules:
- "CORRECT" = ordering is sensible, no major issues (score >= 85)
- "MOSTLY_CORRECT" = mostly good, 1-2 minor issues (score 60-84)
- "NEEDS_REORDERING" = significant problems (score < 60)
- Only flag issues where the ordering would mislead a security team
- Focus on whether critical items are prioritized over less critical ones
- Consider that the weights already encode exposure-type severity
"""
