"""
JSON parser for LLM responses - extracts JSON from potentially messy output.
"""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict:
    """Extract JSON object from LLM response text.

    Handles:
      - Pure JSON responses
      - JSON wrapped in ```json ... ``` code blocks
      - JSON embedded in surrounding text
    """
    if not text:
        raise ValueError("Empty response from LLM")

    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")
