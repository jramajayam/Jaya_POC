"""
Robust JSON extraction from LLM responses.

LLM output may contain markdown fences, reasoning text before/after,
or minor formatting issues.  This module handles all of that.
"""

import json
import re


def extract_json(text: str) -> dict:
    """Extract the first valid JSON object from *text*.

    Strategy (tried in order):
    1. Direct ``json.loads``
    2. Strip markdown ``` fences and retry
    3. Find first ``{ … }`` block with brace-depth matching
    4. Fix trailing-comma issues and retry

    Raises
    ------
    ValueError
        If no valid JSON object can be found.
    """
    if not text or not text.strip():
        raise ValueError("Empty response from LLM")

    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    if "```" in text:
        for part in text.split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # 3. Brace-depth matching
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # 4. Fix trailing commas
                        fixed = re.sub(r",\s*}", "}", candidate)
                        fixed = re.sub(r",\s*]", "]", fixed)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            pass
                    break

    raise ValueError(
        f"No valid JSON found in response "
        f"(length={len(text)}, preview={text[:150]})"
    )
