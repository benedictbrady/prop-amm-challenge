# Task Adapter Interface

The generic loop (`harness/core/loop.py`) is problem-agnostic. Problem logic lives in an adapter command configured in TOML.

## Required actions

The loop calls the adapter command twice per iteration by replacing `{action}` in `task.command_template`:

- `context`
- `evaluate`

Your adapter must print exactly one JSON object to stdout for each action.

## `context` output schema

```json
{
  "target_label": "Holdout avg edge > 525 on 1000 sims",
  "prompt_context": "Task-specific instructions and evaluation context"
}
```

Fields:

- `target_label` (string): concise success criterion shown in prompt and logs.
- `prompt_context` (string): problem-specific context sent to the coding agent.

## `evaluate` output schema

```json
{
  "ok": true,
  "train_avg": 512.3,
  "train_worst": 487.1,
  "promoted_to_holdout": true,
  "holdout_avg": 529.4,
  "passed_target": true
}
```

Required fields:

- `ok` (bool): whether evaluation completed as expected.
- `promoted_to_holdout` (bool)
- `passed_target` (bool)

Optional numeric fields used by the loop for ranking/local-minima logic:

- `train_avg` (number)
- `train_worst` (number)
- `holdout_avg` (number)

Additional fields are preserved in state logs.

## Template variables available in `task.command_template`

- `{action}`: `context` or `evaluate`
- `{workspace}`: absolute workspace path (shell-quoted)
- `{strategy_file}`: absolute strategy file path (shell-quoted)
- `{mode}`: `exploit`, `diversify`, or `restart`
- `{iteration}`: integer iteration index
- `{state_dir}`: absolute harness state directory (shell-quoted)
- `{iter_dir}`: absolute per-iteration directory (shell-quoted)

