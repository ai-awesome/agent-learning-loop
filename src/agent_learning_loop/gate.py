"""ValidationGate — validates lessons against historical outcomes.

Before persisting new lessons, this module checks whether each lesson
reinforces successful or unsuccessful patterns. Lessons that align with
failure patterns are rejected to prevent bad habits from accumulating.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "and", "but", "or", "nor", "not", "no", "so", "yet", "both",
    "for", "with", "from", "into", "onto", "upon", "about", "above",
    "after", "before", "during", "until", "while", "of", "at", "by",
    "in", "on", "to", "up", "out", "off", "over", "under", "through",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "we", "us", "our", "you", "your", "i", "me", "my", "he", "she",
    "if", "then", "than", "when", "where", "what", "which", "who",
    "how", "all", "each", "every", "any", "some", "such", "own",
    "even", "also", "just", "only", "very", "too", "more", "most",
    "other", "another", "as", "like",
})


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, filtering stop words."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) >= 3}


class ValidationGate:
    """Validates lessons against historical outcomes using keyword matching.

    Lessons that predominantly match failure outcomes are rejected.
    No LLM call needed — pure heuristic.
    """

    def __init__(self, min_keyword_overlap: int = 2):
        self.min_keyword_overlap = min_keyword_overlap

    async def validate(
        self,
        lesson: str,
        historical_outcomes: list[dict],
        min_outcomes: int = 5,
    ) -> dict:
        """Validate a single lesson against historical outcomes.

        Args:
            lesson: The lesson text to validate.
            historical_outcomes: List of dicts with fields:
                - action (str): what was done
                - outcome (str): "success" or "failure"
                - reasoning (str): why this action was taken
            min_outcomes: Minimum outcomes needed for validation.

        Returns:
            ValidationResult dict.
        """
        if len(historical_outcomes) < min_outcomes:
            return {
                "accepted": True,
                "baseline_success_rate": 0.0,
                "projected_success_rate": 0.0,
                "matching_outcomes": 0,
                "reason": "insufficient historical data",
            }

        successes = sum(
            1 for o in historical_outcomes if o.get("outcome") == "success"
        )
        baseline = (successes / len(historical_outcomes) * 100) if historical_outcomes else 0.0

        lesson_keywords = _extract_keywords(lesson)
        if not lesson_keywords:
            return {
                "accepted": True,
                "baseline_success_rate": baseline,
                "projected_success_rate": baseline,
                "matching_outcomes": 0,
                "reason": "no extractable keywords in lesson",
            }

        matching_success: list[dict] = []
        matching_failure: list[dict] = []

        for outcome in historical_outcomes:
            reasoning = outcome.get("reasoning", "")
            action = outcome.get("action", "")
            outcome_keywords = _extract_keywords(f"{action} {reasoning}")

            overlap = lesson_keywords & outcome_keywords
            if len(overlap) >= self.min_keyword_overlap:
                if outcome.get("outcome") == "success":
                    matching_success.append(outcome)
                else:
                    matching_failure.append(outcome)

        total_matching = len(matching_success) + len(matching_failure)

        if total_matching == 0:
            return {
                "accepted": True,
                "baseline_success_rate": baseline,
                "projected_success_rate": baseline,
                "matching_outcomes": 0,
                "reason": "lesson does not match any historical outcomes",
            }

        match_success_rate = len(matching_success) / total_matching * 100

        match_weight = min(total_matching / len(historical_outcomes), 0.5)
        projected = round(
            baseline * (1 - match_weight) + match_success_rate * match_weight, 2
        )

        accepted = projected >= baseline

        reason = (
            f"lesson matches {total_matching} outcomes "
            f"({len(matching_success)} successful, {len(matching_failure)} failed); "
            f"projected success rate {projected}% vs baseline {baseline:.1f}%"
        )

        return {
            "accepted": accepted,
            "baseline_success_rate": baseline,
            "projected_success_rate": projected,
            "matching_outcomes": total_matching,
            "reason": reason,
        }

    async def validate_batch(
        self,
        lessons: list[str],
        historical_outcomes: list[dict],
        date: str = "",
    ) -> dict:
        """Validate a batch of lessons.

        Returns:
            GateReport dict with accepted, rejected, and report.
        """
        accepted: list[str] = []
        rejected: list[str] = []
        details: list[str] = []

        for lesson in lessons:
            result = await self.validate(lesson, historical_outcomes)
            if result["accepted"]:
                accepted.append(lesson)
                details.append(f"  PASS: {lesson[:60]}... — {result['reason']}")
            else:
                rejected.append(lesson)
                details.append(f"  FAIL: {lesson[:60]}... — {result['reason']}")

        report = (
            f"Validation gate ({date or 'undated'}): "
            f"{len(accepted)} accepted, {len(rejected)} rejected\n"
            + "\n".join(details)
        )

        logger.info(report)

        return {
            "accepted": accepted,
            "rejected": rejected,
            "report": report,
        }
