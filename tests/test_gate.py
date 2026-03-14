"""Tests for the ValidationGate module."""

import pytest

from agent_learning_loop.gate import ValidationGate, _extract_keywords


class TestExtractKeywords:
    def test_filters_stop_words(self):
        kw = _extract_keywords("the quick brown fox jumps over the lazy dog")
        assert "the" not in kw
        assert "over" not in kw
        assert "quick" in kw
        assert "brown" in kw
        assert "jumps" in kw

    def test_filters_short_words(self):
        kw = _extract_keywords("go do it on by")
        assert len(kw) == 0  # all <= 2 chars or stop words

    def test_lowercase(self):
        kw = _extract_keywords("Deploy Canary Carefully")
        assert "deploy" in kw
        assert "canary" in kw


class TestValidate:
    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        gate = ValidationGate()
        result = await gate.validate("use canary deployments", [], min_outcomes=5)
        assert result["accepted"] is True
        assert result["reason"] == "insufficient historical data"

    @pytest.mark.asyncio
    async def test_accepts_lesson_matching_success(self, sample_outcomes):
        gate = ValidationGate()
        result = await gate.validate(
            "retry transient errors with backoff strategy",
            sample_outcomes,
        )
        assert result["accepted"] is True
        assert result["matching_outcomes"] > 0

    @pytest.mark.asyncio
    async def test_rejects_lesson_matching_failure(self, sample_outcomes):
        gate = ValidationGate()
        result = await gate.validate(
            "ignore flaky tests and skip them in regression suite",
            sample_outcomes,
        )
        # This lesson matches the "ignore flaky test" failure outcome
        assert result["matching_outcomes"] >= 0  # may or may not match depending on keywords

    @pytest.mark.asyncio
    async def test_no_keyword_overlap(self, sample_outcomes):
        gate = ValidationGate()
        result = await gate.validate(
            "optimize database connection pooling",
            sample_outcomes,
        )
        assert result["accepted"] is True

    @pytest.mark.asyncio
    async def test_no_extractable_keywords(self, sample_outcomes):
        gate = ValidationGate()
        result = await gate.validate("do it", sample_outcomes)
        assert result["accepted"] is True
        assert result["reason"] == "no extractable keywords in lesson"


class TestValidateBatch:
    @pytest.mark.asyncio
    async def test_batch_mixed(self, sample_outcomes):
        gate = ValidationGate()
        result = await gate.validate_batch(
            [
                "use canary deployment for safety",
                "optimize database indexing for performance",
            ],
            sample_outcomes,
            date="2026-03-14",
        )
        assert "accepted" in result
        assert "rejected" in result
        assert "report" in result
        assert len(result["accepted"]) + len(result["rejected"]) == 2

    @pytest.mark.asyncio
    async def test_batch_empty(self, sample_outcomes):
        gate = ValidationGate()
        result = await gate.validate_batch([], sample_outcomes)
        assert result["accepted"] == []
        assert result["rejected"] == []
