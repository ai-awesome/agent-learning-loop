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

    Supports synonym expansion and category/entity mapping for higher
    matching precision. Matching confidence is tiered:
    exact keyword (3) > synonym (2) > category (1).

    Lessons that predominantly match failure outcomes are rejected.
    No LLM call needed — pure heuristic.
    """

    def __init__(
        self,
        min_keyword_overlap: int = 2,
        synonyms: dict[str, list[str]] | None = None,
        entity_categories: dict[str, list[str]] | None = None,
    ):
        self.min_keyword_overlap = min_keyword_overlap
        # synonyms: {"momentum": ["trend", "breakout", "rally"]}
        self._synonym_map: dict[str, str] = {}
        if synonyms:
            for canonical, alts in synonyms.items():
                canonical_l = canonical.lower()
                for alt in alts:
                    self._synonym_map[alt.lower()] = canonical_l
                self._synonym_map[canonical_l] = canonical_l
        # entity_categories: {"energy": ["XOM", "CVX"], "frontend": ["React", "Vue"]}
        self._category_map: dict[str, str] = {}
        if entity_categories:
            for category, entities in entity_categories.items():
                cat_l = category.lower()
                for entity in entities:
                    self._category_map[entity.lower()] = cat_l

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
                "match_confidence": 0.0,
                "reason": "no extractable keywords in lesson",
            }

        matching_success: list[dict] = []
        matching_failure: list[dict] = []
        total_confidence = 0.0
        match_count = 0

        for outcome in historical_outcomes:
            reasoning = outcome.get("reasoning", "")
            action = outcome.get("action", "")
            outcome_keywords = _extract_keywords(f"{action} {reasoning}")

            confidence = self._compute_match_confidence(
                lesson_keywords, outcome_keywords
            )
            if confidence > 0:
                total_confidence += confidence
                match_count += 1
                if outcome.get("outcome") == "success":
                    matching_success.append(outcome)
                else:
                    matching_failure.append(outcome)

        total_matching = len(matching_success) + len(matching_failure)
        avg_confidence = (total_confidence / match_count) if match_count else 0.0

        if total_matching == 0:
            return {
                "accepted": True,
                "baseline_success_rate": baseline,
                "projected_success_rate": baseline,
                "matching_outcomes": 0,
                "match_confidence": 0.0,
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
            "match_confidence": round(avg_confidence, 2),
            "reason": reason,
        }

    def _compute_match_confidence(
        self, lesson_kw: set[str], outcome_kw: set[str]
    ) -> float:
        """Compute tiered match confidence between keyword sets.

        Scoring: exact keyword overlap (3) > synonym match (2) > category match (1).
        Returns 0.0 if total score < min_keyword_overlap threshold.
        """
        score = 0.0

        # Exact overlap
        exact = lesson_kw & outcome_kw
        score += len(exact) * 3.0

        # Synonym matching
        if self._synonym_map:
            lesson_canonical = {
                self._synonym_map.get(w, w) for w in lesson_kw
            }
            outcome_canonical = {
                self._synonym_map.get(w, w) for w in outcome_kw
            }
            synonym_matches = (lesson_canonical & outcome_canonical) - exact
            score += len(synonym_matches) * 2.0

        # Category matching
        if self._category_map:
            lesson_cats = {
                self._category_map[w] for w in lesson_kw if w in self._category_map
            }
            outcome_cats = {
                self._category_map[w] for w in outcome_kw if w in self._category_map
            }
            category_matches = lesson_cats & outcome_cats
            score += len(category_matches) * 1.0

        # Threshold: need at least min_keyword_overlap worth of score
        # (exact match = 3 per keyword, so 1 exact match exceeds overlap=2)
        if score < self.min_keyword_overlap:
            return 0.0

        # Normalize to 0-1 range (cap at 1.0)
        max_possible = max(len(lesson_kw), 1) * 3.0
        return min(score / max_possible, 1.0)

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
