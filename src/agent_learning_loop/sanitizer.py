"""Sanitizer — filters adversarial or dangerous lessons to prevent prompt injection."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Domain-agnostic patterns that indicate adversarial lesson content
DEFAULT_SUSPICIOUS_PATTERNS: list[str] = [
    r"ignore.*(?:rule|limit|constraint|check|guard|policy)",
    r"override",
    r"bypass",
    r"disable.*(?:safety|check|limit|guard|filter|validation)",
    r"no.*(?:limit|restriction|constraint)",
    r"unlimited",
    r"always(?:accept|approve|allow|execute)",
    r"never.*(?:reject|deny|refuse|check|validate)",
    r"skip.*(?:validation|check|review|filter)",
]

_DEFAULT_COMPILED = [re.compile(p, re.IGNORECASE) for p in DEFAULT_SUSPICIOUS_PATTERNS]


def _compile_extra(extra_patterns: list[str] | None) -> list[re.Pattern[str]]:
    if not extra_patterns:
        return []
    return [re.compile(p, re.IGNORECASE) for p in extra_patterns]


def is_suspicious(lesson: str, extra_patterns: list[str] | None = None) -> bool:
    """Check if a lesson matches any suspicious pattern."""
    patterns = _DEFAULT_COMPILED + _compile_extra(extra_patterns)
    return any(pat.search(lesson) for pat in patterns)


def sanitize_lessons(
    lessons: list[str], extra_patterns: list[str] | None = None
) -> list[str]:
    """Filter out lessons that match suspicious patterns.

    Returns only the safe lessons. Logs warnings for filtered ones.
    """
    patterns = _DEFAULT_COMPILED + _compile_extra(extra_patterns)
    safe: list[str] = []
    for lesson in lessons:
        if not lesson or not lesson.strip():
            continue
        if any(pat.search(lesson) for pat in patterns):
            logger.warning("Suspicious lesson filtered: %s", lesson[:80])
            continue
        safe.append(lesson)
    return safe
