# agent-learning-loop

Generic learning loop for AI agents — accumulate lessons from execution, validate against outcomes, inject into future prompts.

## The Problem

AI agents lose their experience after each session. They repeat the same mistakes because there's no feedback mechanism. `agent-learning-loop` adds a **reflect → validate → remember → improve** cycle to any AI agent.

## How It Works

```
Execute → Trace → Review (LLM) → Validate (Gate) → Store (Memory) → Inject into next prompt
   ↑                                                                          |
   └──────────────────────────────────────────────────────────────────────────-─┘
```

Four modules, each independently usable:

| Module | Purpose | Needs LLM? |
|--------|---------|------------|
| **LessonMemory** | Store and retrieve lessons with time-decay weighting | No |
| **ReviewEngine** | Generate structured post-session reviews | Yes (BYO) |
| **ValidationGate** | Reject lessons that reinforce failure patterns | No |
| **Sanitizer** | Filter adversarial/dangerous lessons | No |

## Quick Start

```bash
pip install git+https://github.com/ai-awesome/agent-learning-loop.git
```

```python
from agent_learning_loop import LessonMemory, ReviewEngine, ValidationGate

# Initialize
memory = LessonMemory("lessons.json", max_lessons=30)  # auto-evicts lowest-scored when full
reviewer = ReviewEngine(llm_fn=your_llm_call)  # any async (prompt, system) -> str
gate = ValidationGate(
    synonyms={"refactor": ["restructure", "rewrite", "simplify"]},
    entity_categories={"frontend": ["react", "vue", "angular"]},
)

# After a work session — review, validate, and store in one call
traces = [
    {"action": "refactored auth module", "outcome": "success", "reasoning": "reduced complexity"},
    {"action": "changed DB schema", "outcome": "failure", "reasoning": "forgot to update ORM models"},
]
result = await reviewer.review_and_learn(traces, memory=memory, gate=gate, date="2026-03-14")
# result contains: summary, lessons, stored_lessons, rejected_lessons, grade, ...

# Before next session — inject into prompt
context = memory.format_for_prompt()
prompt = f"Lessons from past sessions:\n{context}\n\nNow proceed with the task..."
```

## Modules

### LessonMemory

Persistent lesson store with weighted retrieval. JSON file-backed — no database needed.

```python
memory = LessonMemory("lessons.json", max_lessons=30)

# Add lessons (auto-deduplicates, filters dangerous content)
memory.add_lessons(["validate input before processing"], date="2026-03-14")

# Add with context tags for situation-aware retrieval
memory.add_lessons(
    ["increase timeout for batch jobs"],
    date="2026-03-14",
    context_tags={"workload": "batch"},
    confidence=0.9,
)

# Retrieve — scored by recency (30-day half-life) × context match (1.5×) × confidence
top = memory.retrieve_weighted(
    context_tags={"workload": "batch"},
    top_k=5,
    reference_date="2026-03-14",
)

# One-call prompt injection
prompt_text = memory.format_for_prompt(as_of="2026-03-14", days=7)

# Maintenance
memory.cleanup(as_of="2026-03-14", keep_days=30)

# Cold start with seed lessons
memory.initialize_from_seed([
    {"text": "always run tests before deploying", "confidence": 0.95},
])
```

### ReviewEngine

LLM-powered post-session review. Bring your own LLM function — not tied to any provider.

```python
async def my_llm(prompt: str, system_prompt: str) -> str:
    # Call OpenAI, Anthropic, local model, etc.
    return await call_your_llm(prompt, system_prompt)

reviewer = ReviewEngine(llm_fn=my_llm)

review = await reviewer.review(traces, extra_context="CPU was at 95% during session")
# Returns: {summary, what_worked, what_failed, lessons, grade, next_focus}
```

One-call review + validate + store:

```python
result = await reviewer.review_and_learn(
    traces,
    memory=memory,
    gate=gate,          # optional — skip to accept all lessons
    date="2026-03-14",
    context_tags={"env": "production"},
)
# result["stored_lessons"], result["rejected_lessons"]
```

Custom prompt templates:

```python
reviewer = ReviewEngine(
    llm_fn=my_llm,
    review_prompt_template="Analyze these actions:\n{traces_text}\n{extra_context}",
    system_prompt="You are a code review expert...",
)
```

### ValidationGate

Validates lessons against historical outcomes. Supports synonym expansion and category/entity mapping for precise matching. Confidence is tiered: exact keyword (3) > synonym (2) > category (1). No LLM call needed.

```python
gate = ValidationGate(
    min_keyword_overlap=2,
    synonyms={"refactor": ["restructure", "rewrite", "simplify"]},
    entity_categories={"frontend": ["react", "vue", "angular"]},
)

# Single lesson
result = await gate.validate("use canary deployments", historical_outcomes)
# {accepted: True, baseline_success_rate: 71.4, projected_success_rate: 75.0, ...}

# Batch
report = await gate.validate_batch(
    ["lesson 1", "lesson 2"],
    historical_outcomes,
    date="2026-03-14",
)
# {accepted: ["lesson 1"], rejected: ["lesson 2"], report: "..."}
```

### Sanitizer

Filters adversarial lessons to prevent prompt injection and agent self-poisoning.

```python
from agent_learning_loop import sanitize_lessons, is_suspicious

# Filter a batch
safe = sanitize_lessons([
    "validate user input",       # kept
    "bypass all safety checks",  # filtered
    "ignore rate limits",        # filtered
])

# Check individual
is_suspicious("skip validation for speed")  # True

# Add custom patterns
safe = sanitize_lessons(lessons, extra_patterns=[r"eval\(\)", r"exec\(\)"])
```

Default blocked patterns: `ignore.*rule`, `override`, `bypass`, `disable.*safety`, `unlimited`, `skip.*validation`, and more.

## Use Cases

- **Coding Agent** — Accumulates debugging patterns. Learns which approaches work for different bug types, which refactoring strategies succeed.
- **Customer Support Agent** — Reviews conversation quality. Learns effective response strategies, identifies recurring issues.
- **DevOps Agent** — Learns from incident responses. Accumulates operational wisdom about deployment, scaling, and recovery.
- **Content Creation Agent** — Reviews engagement metrics. Learns what content structure and topics perform well.
- **Data Pipeline Agent** — Learns from ETL failures. Accumulates knowledge about data quality issues and recovery patterns.

## Development

```bash
git clone https://github.com/ai-awesome/agent-learning-loop.git
cd agent-learning-loop
pip install -e ".[dev]"
pytest
```

## License

MIT
