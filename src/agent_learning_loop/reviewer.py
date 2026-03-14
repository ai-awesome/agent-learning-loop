"""ReviewEngine — LLM-powered post-session review for AI agents."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are reviewing a series of actions taken by an AI agent.
Analyze the outcomes, identify patterns, and provide actionable lessons.

Respond with EXACTLY one JSON object:
{
  "summary": "1-2 sentence overview of the session",
  "what_worked": ["list of things that went well"],
  "what_failed": ["list of mistakes or missed opportunities"],
  "lessons": ["actionable lessons for next time"],
  "grade": "A/B/C/D/F with optional +/-",
  "next_focus": "one sentence about what to focus on next"
}"""

DEFAULT_PROMPT_TEMPLATE = """SESSION REVIEW

EXECUTION LOG:
{traces_text}

{extra_context}
Review this session. What worked? What didn't? What should be done differently next time?"""


class ReviewEngine:
    """Generate structured post-session reviews using an LLM.

    The engine is LLM-agnostic: provide any async function that takes
    (prompt, system_prompt) and returns a string response.
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str], Awaitable[str]],
        review_prompt_template: str | None = None,
        system_prompt: str | None = None,
    ):
        self.llm_fn = llm_fn
        self.prompt_template = review_prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    async def review(
        self, traces: list[dict], extra_context: str = ""
    ) -> dict:
        """Generate a structured review from execution traces.

        Args:
            traces: List of dicts with fields like action, outcome, reasoning,
                    timestamp, metadata.
            extra_context: Additional context to include in the prompt.

        Returns:
            Review dict with summary, what_worked, what_failed, lessons,
            grade, next_focus.
        """
        traces_text = json.dumps(traces, indent=2, default=str)

        prompt = self.prompt_template.format(
            traces_text=traces_text,
            extra_context=extra_context,
        )

        try:
            response = await self.llm_fn(prompt, self.system_prompt)
            review = _parse_json(response)

            # Ensure required fields
            review.setdefault("summary", "Review generated.")
            review.setdefault("what_worked", [])
            review.setdefault("what_failed", [])
            review.setdefault("lessons", [])
            review.setdefault("grade", "N/A")
            review.setdefault("next_focus", "")

            return review

        except json.JSONDecodeError:
            logger.error("ReviewEngine: LLM returned invalid JSON")
            return _fallback(traces, "LLM returned invalid JSON")
        except Exception:
            logger.exception("ReviewEngine: review generation failed")
            return _fallback(traces, "Review generation failed")


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from LLM response text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No JSON found in response", text, 0)


def _fallback(traces: list[dict], error: str) -> dict:
    return {
        "summary": f"Auto-review failed. {len(traces)} actions recorded.",
        "what_worked": [],
        "what_failed": [],
        "lessons": [],
        "grade": "N/A",
        "next_focus": "",
        "error": error,
    }
