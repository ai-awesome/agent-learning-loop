"""Shared fixtures for agent-learning-loop tests."""

import pytest


@pytest.fixture
def lessons_path(tmp_path):
    """Path for a temporary lessons JSON file."""
    return str(tmp_path / "lessons.json")


@pytest.fixture
def sample_lessons():
    """Generic lesson strings."""
    return [
        "Always validate input before processing",
        "Retry transient errors with exponential backoff",
        "Log context alongside error messages for debugging",
    ]


@pytest.fixture
def sample_traces():
    """Generic execution traces."""
    return [
        {
            "action": "deployed service",
            "outcome": "success",
            "reasoning": "health checks passed after deployment",
            "timestamp": "2026-03-14T10:00:00",
        },
        {
            "action": "rolled back migration",
            "outcome": "failure",
            "reasoning": "migration had missing column reference",
            "timestamp": "2026-03-14T11:00:00",
        },
        {
            "action": "restarted worker pool",
            "outcome": "success",
            "reasoning": "workers were deadlocked due to connection leak",
            "timestamp": "2026-03-14T12:00:00",
        },
    ]


@pytest.fixture
def sample_outcomes():
    """Historical outcomes for validation gate tests."""
    return [
        {"action": "deploy with canary", "outcome": "success", "reasoning": "canary deployment caught errors early"},
        {"action": "deploy with canary", "outcome": "success", "reasoning": "canary rollout was smooth and stable"},
        {"action": "deploy without canary", "outcome": "failure", "reasoning": "direct deployment caused outage"},
        {"action": "deploy without canary", "outcome": "failure", "reasoning": "skipped canary and hit production bug"},
        {"action": "retry failed job", "outcome": "success", "reasoning": "transient error resolved on retry"},
        {"action": "retry failed job", "outcome": "success", "reasoning": "retry with backoff succeeded"},
        {"action": "ignore flaky test", "outcome": "failure", "reasoning": "ignored test hid real regression"},
    ]
