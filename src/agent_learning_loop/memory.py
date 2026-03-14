"""LessonMemory — persistent storage and weighted retrieval of agent lessons."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from agent_learning_loop.sanitizer import sanitize_lessons

logger = logging.getLogger(__name__)


class LessonMemory:
    """Persistent lesson store with time-decay weighted retrieval.

    Lessons are stored as JSON on disk. Each entry is a dict with fields:
        text (str), date (str YYYY-MM-DD), context_tags (dict), confidence (float)

    Retrieval supports recency decay, context-tag boosting, and confidence weighting.
    When max_lessons is set, the oldest/lowest-scored lessons are evicted
    automatically to stay within capacity.
    """

    def __init__(self, path: str = "lessons.json", max_lessons: int = 0):
        self.path = path
        self.max_lessons = max_lessons  # 0 = unlimited
        self._lessons: list[dict] = self._load()

    # -- Persistence --

    def _load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load lessons from %s", self.path)
            return []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._lessons, f, indent=2)

    def _evict(self) -> None:
        """Evict lowest-scored lessons if over capacity."""
        if self.max_lessons <= 0 or len(self._lessons) <= self.max_lessons:
            return
        now = datetime.now().strftime("%Y-%m-%d")
        ref = datetime.strptime(now, "%Y-%m-%d")

        def score(entry: dict) -> float:
            days_ago = max(_safe_days_between(ref, entry.get("date", "")), 0)
            recency = 0.5 ** (days_ago / 30.0)
            return recency * entry.get("confidence", 0.8)

        self._lessons.sort(key=score, reverse=True)
        removed = len(self._lessons) - self.max_lessons
        self._lessons = self._lessons[: self.max_lessons]
        logger.info("Evicted %d lowest-scored lessons (capacity: %d)", removed, self.max_lessons)

    # -- Write operations --

    def add_lessons(
        self,
        lessons: list[str],
        date: str,
        context_tags: dict | None = None,
        confidence: float = 0.8,
    ) -> None:
        """Add lessons with deduplication. Same text -> update date."""
        safe = sanitize_lessons([ls for ls in lessons if ls and ls.strip()])
        existing = {entry["text"]: entry for entry in self._lessons}

        for lesson in safe:
            text = lesson.strip()
            if text in existing:
                if date >= existing[text].get("date", ""):
                    existing[text]["date"] = date
                    existing[text]["confidence"] = confidence
                    if context_tags is not None:
                        existing[text]["context_tags"] = context_tags
            else:
                entry: dict = {
                    "text": text,
                    "date": date,
                    "confidence": confidence,
                    "context_tags": context_tags or {},
                }
                self._lessons.append(entry)
                existing[text] = entry
        self._evict()
        self._save()

    def initialize_from_seed(self, seed_lessons: list[dict]) -> None:
        """Load seed lessons for cold start. Only works when memory is empty."""
        if not self.is_empty():
            return
        today = datetime.now().strftime("%Y-%m-%d")
        for item in seed_lessons:
            self._lessons.append({
                "text": item.get("text", ""),
                "date": item.get("date", today),
                "confidence": item.get("confidence", 0.8),
                "context_tags": item.get("context_tags", {}),
            })
        self._save()
        logger.info("Memory initialized with %d seed lessons", len(seed_lessons))

    def cleanup(self, as_of: str, keep_days: int = 30) -> None:
        """Remove lessons older than keep_days, then deduplicate."""
        cutoff = datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=keep_days)
        before = len(self._lessons)

        self._lessons = [
            e for e in self._lessons
            if _parse_date(e.get("date", "")) >= cutoff
        ]

        # Deduplicate: keep the entry with the most recent date
        seen: dict[str, dict] = {}
        for entry in self._lessons:
            text = entry.get("text", "")
            if text in seen:
                if entry.get("date", "") > seen[text].get("date", ""):
                    seen[text] = entry
            else:
                seen[text] = entry
        self._lessons = list(seen.values())

        removed = before - len(self._lessons)
        if removed > 0:
            self._save()
            logger.info("Cleanup: removed %d old/duplicate lessons", removed)

    # -- Read operations --

    def get_all(self) -> list[dict]:
        return list(self._lessons)

    def get_recent(
        self, as_of: str, days: int = 7, max_lessons: int = 15
    ) -> list[dict]:
        """Get lessons from the last N days, most recent first."""
        cutoff = datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=days)
        recent = [
            e for e in self._lessons
            if _parse_date(e.get("date", "")) >= cutoff
        ]
        recent.sort(key=lambda x: x.get("date", ""), reverse=True)
        return recent[:max_lessons]

    def retrieve_weighted(
        self,
        context_tags: dict | None = None,
        top_k: int = 5,
        reference_date: str | None = None,
    ) -> list[str]:
        """Retrieve lessons scored by recency, context match, and confidence.

        Score = recency (30-day half-life) * context_boost (1.5x) * confidence.
        """
        if not self._lessons:
            return []

        if reference_date is None:
            reference_date = datetime.now().strftime("%Y-%m-%d")
        ref = datetime.strptime(reference_date, "%Y-%m-%d")

        scored: list[tuple[float, str]] = []
        for entry in self._lessons:
            text = entry.get("text", "")

            # Recency: 30-day half-life exponential decay
            days_ago = max((_safe_days_between(ref, entry.get("date", ""))), 0)
            recency = 0.5 ** (days_ago / 30.0)

            # Context boost
            context_boost = 1.0
            if context_tags and entry.get("context_tags"):
                entry_tags = entry["context_tags"]
                if any(
                    entry_tags.get(k) == v for k, v in context_tags.items()
                ):
                    context_boost = 1.5

            # Confidence
            confidence = entry.get("confidence", 0.8)

            score = recency * context_boost * confidence
            scored.append((score, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    def format_for_prompt(
        self, as_of: str | None = None, days: int = 7, max_lessons: int = 15
    ) -> str:
        """Format recent lessons for injection into an LLM prompt."""
        if as_of is None:
            as_of = datetime.now().strftime("%Y-%m-%d")

        recent = self.get_recent(as_of=as_of, days=days, max_lessons=max_lessons)
        if not recent:
            return "No accumulated lessons yet."

        lines: list[str] = []
        current_date = None
        for entry in recent:
            date = entry.get("date", "")
            if date != current_date:
                lines.append(f"\n[{date}]")
                current_date = date
            lines.append(f"- {entry['text']}")

        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self._lessons) == 0


def _parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return datetime.min


def _safe_days_between(ref: datetime, date_str: str) -> int:
    try:
        entry_date = datetime.strptime(date_str, "%Y-%m-%d")
        return (ref - entry_date).days
    except (ValueError, TypeError):
        return 999
