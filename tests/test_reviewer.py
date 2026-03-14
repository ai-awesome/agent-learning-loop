"""Tests for the ReviewEngine module."""

import json

import pytest

from agent_learning_loop.gate import ValidationGate
from agent_learning_loop.memory import LessonMemory
from agent_learning_loop.reviewer import ReviewEngine


async def mock_llm_valid(prompt: str, system_prompt: str) -> str:
    """Mock LLM that returns valid JSON."""
    return json.dumps({
        "summary": "Session went well overall.",
        "what_worked": ["Deployment was smooth"],
        "what_failed": ["Migration had issues"],
        "lessons": ["Always run migrations in staging first"],
        "grade": "B+",
        "next_focus": "Improve migration testing",
    })


async def mock_llm_invalid(prompt: str, system_prompt: str) -> str:
    """Mock LLM that returns invalid JSON."""
    return "This is not JSON at all."


async def mock_llm_markdown_json(prompt: str, system_prompt: str) -> str:
    """Mock LLM that returns JSON inside markdown code fence."""
    return '''Here's the review:

```json
{
  "summary": "Extracted from markdown",
  "what_worked": [],
  "what_failed": [],
  "lessons": ["Test lesson"],
  "grade": "A",
  "next_focus": "Keep going"
}
```
'''


async def mock_llm_error(prompt: str, system_prompt: str) -> str:
    """Mock LLM that raises an exception."""
    raise RuntimeError("LLM service unavailable")


class TestReview:
    @pytest.mark.asyncio
    async def test_valid_response(self, sample_traces):
        engine = ReviewEngine(llm_fn=mock_llm_valid)
        result = await engine.review(sample_traces)
        assert result["summary"] == "Session went well overall."
        assert result["grade"] == "B+"
        assert len(result["lessons"]) == 1

    @pytest.mark.asyncio
    async def test_all_required_fields(self, sample_traces):
        engine = ReviewEngine(llm_fn=mock_llm_valid)
        result = await engine.review(sample_traces)
        for field in ["summary", "what_worked", "what_failed", "lessons", "grade", "next_focus"]:
            assert field in result

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self, sample_traces):
        engine = ReviewEngine(llm_fn=mock_llm_invalid)
        result = await engine.review(sample_traces)
        assert "error" in result
        assert result["grade"] == "N/A"

    @pytest.mark.asyncio
    async def test_exception_fallback(self, sample_traces):
        engine = ReviewEngine(llm_fn=mock_llm_error)
        result = await engine.review(sample_traces)
        assert "error" in result
        assert "3 actions recorded" in result["summary"]

    @pytest.mark.asyncio
    async def test_markdown_json_extraction(self, sample_traces):
        engine = ReviewEngine(llm_fn=mock_llm_markdown_json)
        result = await engine.review(sample_traces)
        assert result["summary"] == "Extracted from markdown"

    @pytest.mark.asyncio
    async def test_custom_prompt_template(self, sample_traces):
        captured = {}

        async def capture_llm(prompt, system_prompt):
            captured["prompt"] = prompt
            return await mock_llm_valid(prompt, system_prompt)

        engine = ReviewEngine(
            llm_fn=capture_llm,
            review_prompt_template="Custom template: {traces_text} {extra_context}",
        )
        await engine.review(sample_traces)
        assert captured["prompt"].startswith("Custom template:")

    @pytest.mark.asyncio
    async def test_extra_context(self, sample_traces):
        captured = {}

        async def capture_llm(prompt, system_prompt):
            captured["prompt"] = prompt
            return await mock_llm_valid(prompt, system_prompt)

        engine = ReviewEngine(llm_fn=capture_llm)
        await engine.review(sample_traces, extra_context="High CPU during session")
        assert "High CPU during session" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_empty_traces(self):
        engine = ReviewEngine(llm_fn=mock_llm_valid)
        result = await engine.review([])
        assert "summary" in result


class TestReviewAndLearn:
    @pytest.mark.asyncio
    async def test_stores_lessons_in_memory(self, lessons_path, sample_traces):
        engine = ReviewEngine(llm_fn=mock_llm_valid)
        memory = LessonMemory(lessons_path)
        result = await engine.review_and_learn(
            sample_traces, memory=memory, date="2026-03-14"
        )
        assert len(result["stored_lessons"]) == 1
        assert result["stored_lessons"][0] == "Always run migrations in staging first"
        assert not memory.is_empty()

    @pytest.mark.asyncio
    async def test_with_gate_validation(self, lessons_path, sample_traces):
        engine = ReviewEngine(llm_fn=mock_llm_valid)
        memory = LessonMemory(lessons_path)
        gate = ValidationGate()
        result = await engine.review_and_learn(
            sample_traces, memory=memory, gate=gate, date="2026-03-14"
        )
        assert "stored_lessons" in result
        assert "rejected_lessons" in result
        total = len(result["stored_lessons"]) + len(result["rejected_lessons"])
        assert total == 1  # one lesson from mock_llm_valid

    @pytest.mark.asyncio
    async def test_no_lessons_stores_nothing(self, lessons_path, sample_traces):
        async def llm_no_lessons(prompt, system_prompt):
            return json.dumps({
                "summary": "Nothing happened.",
                "what_worked": [],
                "what_failed": [],
                "lessons": [],
                "grade": "N/A",
                "next_focus": "",
            })

        engine = ReviewEngine(llm_fn=llm_no_lessons)
        memory = LessonMemory(lessons_path)
        result = await engine.review_and_learn(
            sample_traces, memory=memory, date="2026-03-14"
        )
        assert result["stored_lessons"] == []
        assert memory.is_empty()

    @pytest.mark.asyncio
    async def test_context_tags_passed_to_memory(self, lessons_path, sample_traces):
        engine = ReviewEngine(llm_fn=mock_llm_valid)
        memory = LessonMemory(lessons_path)
        await engine.review_and_learn(
            sample_traces,
            memory=memory,
            date="2026-03-14",
            context_tags={"env": "staging"},
        )
        stored = memory.get_all()
        assert len(stored) == 1
        assert stored[0]["context_tags"] == {"env": "staging"}
