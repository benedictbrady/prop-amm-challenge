# Agent Framework Research (As Of Feb 19, 2026)

## Problem fit

This challenge is a repeated code-improvement loop with fast objective feedback (`prop-amm run`) and strict final gate (`1000`-sim out-of-sample). The best setup is not a single monolithic agent. It is:

1. A strong coding agent runtime.
2. A deterministic outer harness for evaluation, budget, and stopping rules.
3. A restart/diversification policy to avoid local minima.

## What people use for this style (Ralph Wiggum and related)

- Ralph-style loops: Geoffrey Huntley describes a pattern where an agent is run repeatedly in fresh sessions with a fixed objective and test gate until success. ([post](https://ghuntley.com/agentic-loop/), [project](https://github.com/ghuntley/ralph-wiggum))
- OpenAI also published a similar harness pattern for autonomous coding workflows: small, reliable loop; machine-checkable gates; restart when stuck. ([Harness Engineering for Agents](https://openai.com/index/introducing-codex/))

These two independently converge on the same core idea: keep the outer loop deterministic and let the model focus on proposing code changes.

## Frameworks reviewed

### 1) OpenHands

- Designed for long-horizon software tasks with local/remote runtime support. ([docs](https://docs.all-hands.dev/openhands/usage/runtimes/runtime-options), [repo](https://github.com/All-Hands-AI/OpenHands))
- Has benchmark tooling and explicit evaluator harnesses in-repo. ([benchmark harness](https://github.com/All-Hands-AI/benchmarks), [evaluation docs](https://docs.all-hands.dev/openhands/usage/how-to/evaluation-harness))
- Fit here: strong if you want an open-source autonomous coding runtime and containerized isolation.

### 2) Claude Code

- Has hooks for pre/post actions and policy enforcement from settings. ([hooks docs](https://docs.claude.com/en/docs/claude-code/hooks), [settings docs](https://docs.claude.com/en/docs/claude-code/settings))
- Fit here: good for tightly controlling tool usage and injecting guardrails around each iteration.

### 3) OpenAI Agents SDK / Codex stack

- OpenAI provides an Agents SDK (Python/TypeScript), handoffs, and tracing support. ([Python docs](https://openai.github.io/openai-agents-python/), [JS docs](https://openai.github.io/openai-agents-js/))
- Codex/harness guidance emphasizes explicit test gates and loop reliability over one-shot prompting. ([introducing-codex](https://openai.com/index/introducing-codex/))
- Fit here: strong if you want first-party integration plus detailed usage metering.

### 4) LangGraph / AutoGen (orchestrators)

- LangGraph focuses on stateful graph orchestration, checkpointing, and resumability. ([LangGraph docs](https://langchain-ai.github.io/langgraph/reference/graphs/))
- AutoGen provides multi-agent patterns and an agent framework from Microsoft. ([AutoGen docs](https://microsoft.github.io/autogen/dev/))
- Fit here: useful when you want custom multi-agent topologies (planner/critic/implementer) beyond a single worker loop.

### 5) Aider

- Very effective code-editing assistant with benchmark visibility and terminal workflow focus. ([Aider docs](https://aider.chat/docs/), [leaderboard](https://aider.chat/docs/leaderboards/))
- Fit here: good inner-loop editor, but you still need an external harness for stop conditions and budget policy.

## Budget notes

- OpenAI API pricing page explicitly documents input/output/cached-input rates and Batch API discounts. ([pricing](https://platform.openai.com/docs/pricing))
- Claude/OpenAI/OpenHands can all be wired into the same outer budget guard if the harness can parse either:
  - explicit `COST_USD=...`, or
  - token usage fields with configured per-million rates.

## Recommended setup for this puzzle

- Keep this repository-local harness as the source of truth for:
  - score gating,
  - out-of-sample holdout checks,
  - budget stop,
  - anti-local-minima policy.
- Plug in whichever coding agent runtime is strongest for your environment (Codex, Claude Code, or OpenHands).
- Do not let the model decide when to stop; let the harness stop only on objective pass/fail criteria.

## Local-minima escape policy baked into the harness

- Multi-fold train gate on disjoint seed ranges.
- Separate out-of-sample `1000`-sim holdout gate.
- Stagnation detection and forced diversify mode.
- Periodic controlled restarts from baseline/elite candidates.
- Elite pool with hash dedupe so exploration can branch without forgetting best known candidates.

