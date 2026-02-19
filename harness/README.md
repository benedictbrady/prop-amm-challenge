# Generic Agent Harness

This directory now contains a reusable, cloud-ready optimization loop with two separable layers:

- Agent backend: how code changes are proposed (`harness/agents/*`)
- Task adapter: how candidates are evaluated (`harness/tasks/*`)

The core loop (`harness/core/loop.py`) is problem-agnostic.

## Architecture

- `harness/core/loop.py`: generic iteration engine, budget management, anti-local-minima policy, state persistence
- `harness/tasks/prop_amm/adapter.py`: Prop AMM-specific evaluator
- `harness/tasks/prop_amm/task.toml`: Prop AMM scoring config (folds, thresholds, parse regex)
- `harness/agents/codex_cli_agent.py`: default cloud backend (wraps `codex exec`)
- `harness/agents/openai_ops_agent.py`: OpenAI Responses ops-capable backend
- `harness/agents/openai_file_editor.py`: minimal one-file editor backend
- `harness/configs/prop_amm.local.example.toml`: local config template
- `harness/configs/prop_amm.cloud.example.toml`: cloud config template
- `harness/ADAPTER_INTERFACE.md`: contract for creating new task adapters

## Quick Start (local)

```bash
cp harness/configs/prop_amm.local.example.toml harness/configs/prop_amm.local.toml
python3 harness/core/loop.py --config harness/configs/prop_amm.local.toml --dry-run
python3 harness/core/loop.py --config harness/configs/prop_amm.local.toml
```

## Quick Start (cloud)

Use either:

- GitHub Actions workflow: `.github/workflows/harness-cloud.yml`
- Docker image: `harness/cloud/Dockerfile`

See `harness/CLOUD.md`.

## Budget Accounting Notes

- If agent output contains `COST_USD=...` or token usage fields, the harness uses parsed cost.
- Set `budget.max_usd = 0` to disable internal budget stop checks.
- If cost cannot be parsed and the agent exits successfully, `fallback_per_iteration_usd` is charged.
- If the agent process fails early (non-zero exit), `fallback_on_failure_usd` is charged (default `0.0`).
- Legacy state files are migrated to this policy automatically on next run.

## Continuous Mode + Sysadmin Guard

- Set `loop.max_iterations = 0` for unbounded iterations.
- Set `loop.stop_on_target = false` to continue optimizing after target hits.
- Enable `[sysadmin]` to run a periodic high-reasoning health check and auto-remediation actions.
- Default cloud config runs sysadmin checks every 10 minutes using `SYSADMIN_MODEL` (default `gpt-5-codex`) via `harness/agents/openai_sysadmin.py`.

## Why this is reusable

To support a new problem, keep the same loop and either:

1. Replace only `task.command_template` with a new adapter.
2. Swap `agent.backend` (or override `agent.command_template`) to use a different core agent runtime.

No loop code changes are required if the adapter follows `harness/ADAPTER_INTERFACE.md`.

## Agent backend abstraction

Preferred config path:

```toml
[agent]
backend = "codex_cli" # or openai_ops, openai_file_editor
use_shell = true

[agent.backend_options]
model_expr = "$AGENT_MODEL"
```

Backward compatibility:

- `agent.command_template` still works and takes precedence.
- This keeps existing setups stable while letting you switch core agents with minimal config edits.

## New Problem Bootstrap

1. Create `harness/tasks/<problem>/adapter.py` that implements `context` + `evaluate`.
2. Create `harness/tasks/<problem>/task.toml` with your gates/commands.
3. Copy `harness/configs/prop_amm.local.example.toml` and replace only `[task]` + file paths.
4. Reuse the same `harness/core/loop.py` and cloud workflow.

## Legacy script

`harness/agent_harness.py` remains as the original Prop AMM-specific harness. Prefer `harness/core/loop.py` for new work.
