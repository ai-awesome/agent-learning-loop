"""Data structures for agent-learning-loop."""

from __future__ import annotations

from typing import TypedDict


class Lesson(TypedDict, total=False):
    """A single lesson learned from execution."""

    lesson: str  # Required: the lesson text
    date: str  # Required: ISO date string (YYYY-MM-DD)
    confidence: float  # Default 0.8
    context_tags: dict[str, str]  # Optional: e.g. {"regime": "high-load"}
    source: str  # Optional: e.g. "seed", "review", "manual"


class Review(TypedDict, total=False):
    """Structured output from a post-session review."""

    summary: str
    what_worked: list[str]
    what_failed: list[str]
    lessons: list[str]
    grade: str
    next_focus: str
    date: str
    error: str


class ValidationResult(TypedDict):
    """Result of validating a single lesson."""

    accepted: bool
    baseline_success_rate: float
    projected_success_rate: float
    matching_outcomes: int
    reason: str


class GateReport(TypedDict):
    """Result of validating a batch of lessons."""

    accepted: list[str]
    rejected: list[str]
    report: str
