# ADR 0002: LangGraph workflow, decisions in code, engineer review as a typed interrupt

## Context

The output of this system goes into an inspection record that someone may have to defend.
Three things follow:

1. The priority and the due date cannot depend on how a language model phrases things.
2. Some findings must be signed off by an engineer, and that person answers hours later,
   from another machine.
3. Every note has to be traceable: which numbers, which guidance, which model, how many
   tries, who approved.

## Decision

- Use LangGraph for the workflow: assess, exposure, policy, retrieve, draft, validate,
  fallback, human review, publish. Each step is a node with a typed state.
- The policy node is plain Python with a configurable table. It sets priority, due days and
  whether review is required. The language model never decides any of these.
- The draft node only writes a short note. The validate node checks it in code: every
  citation must be one of the retrieved pages, every number must be one of the measured
  facts, the worst state, priority code and due days must appear, and nothing worse than
  the measured state may be mentioned.
- A failed check goes back to the model with the list of problems, up to three tries. After
  that a deterministic template note is used and the photo is sent to review.
- Review is a LangGraph `interrupt()` with a Pydantic `response_schema` (added in LangGraph
  1.2.12, September 2026). The client gets a JSON schema to render a form, and a malformed
  answer is rejected by validation instead of being written to the record.
- The graph is checkpointed in SQLite. `bridge-triage review <thread>` resumes a paused run
  from a new process.
- The publish node upserts by thread id into DuckDB, so a replayed step never writes twice.

## Alternatives considered

- A single prompt that returns the full finding as JSON: simpler, but puts the decision in
  the model and makes partial failures invisible.
- A free form tool calling agent: more flexible, and exactly the flexibility that is not
  wanted in a regulated record.
- A queue plus a database table for pending reviews: works, but rebuilds what the
  checkpointer already gives, including replay and the full state at the moment of pause.

## Consequences

- The language model is replaceable without touching the decision: template, a local Qwen
  model on CPU, or any OpenAI compatible endpoint, selected by configuration.
- Small local models fail the checks often. That is measured per check, and the fallback
  means a failure costs review time, never a wrong record.
- SQLite is fine for one inspector or a small team. A shared deployment would use the
  Postgres checkpointer with the same graph.
