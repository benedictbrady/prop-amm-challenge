#!/usr/bin/env python3
"""Generic autonomous coding loop with pluggable task adapter and agent backend."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from harness.core.agent_backends import AgentBackendError, resolve_agent_backend
except ModuleNotFoundError:  # pragma: no cover - script execution path fallback
    from agent_backends import AgentBackendError, resolve_agent_backend  # type: ignore

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing TOML parser (`tomllib` or `tomli`)") from exc

COST_PATTERNS = [
    re.compile(r"(?:TOTAL_)?COST_USD\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r'"cost_usd"\s*:\s*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE),
]

TOKEN_PATTERNS = {
    "input": [
        re.compile(r'"input_tokens"\s*:\s*(\d+)', re.IGNORECASE),
        re.compile(r"\binput_tokens\b\s*[:=]\s*(\d+)", re.IGNORECASE),
        re.compile(r"\bprompt_tokens\b\s*[:=]\s*(\d+)", re.IGNORECASE),
    ],
    "cached_input": [
        re.compile(r'"cached_input_tokens"\s*:\s*(\d+)', re.IGNORECASE),
        re.compile(r"\bcached_input_tokens\b\s*[:=]\s*(\d+)", re.IGNORECASE),
    ],
    "output": [
        re.compile(r'"output_tokens"\s*:\s*(\d+)', re.IGNORECASE),
        re.compile(r"\boutput_tokens\b\s*[:=]\s*(\d+)", re.IGNORECASE),
        re.compile(r"\bcompletion_tokens\b\s*[:=]\s*(\d+)", re.IGNORECASE),
    ],
}

DEFAULT_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    Iteration: {iteration}
    Mode: {mode}
    Mode directive: {mode_instructions}

    Target:
    - {target_label}

    Primary file to edit:
    - {strategy_file}

    Task-specific context:
    {task_context}

    Budget:
    - Remaining USD: {budget_remaining:.2f}
    - Spent USD: {budget_spent:.2f}

    Recent history:
    {recent_history}

    Elite candidates:
    {elite_summary}

    Deliverables:
    1) Make code changes.
    2) Run fast sanity checks only (the harness runs objective gates).
    3) End response with:
       - changes:
       - rationale:
       - risk:
       - optional COST_USD=<number>
    """
)

DEFAULT_SYSADMIN_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are the on-call sysadmin for an autonomous coding harness.
    Your job is to keep it running continuously and recover from failures.
    Never recommend stopping permanently.

    Context:
    - Iteration: {iteration}
    - Target: {target_label}
    - Failure streak (agent): {failure_streak}
    - Latest iteration dir: {iter_dir}
    - State file: {state_path}
    - Strategy file: {strategy_file}

    Recent history:
    {recent_history}

    Latest failures:
    {recent_failures}

    Allowed remediation actions:
    - noop
    - sleep_60
    - sleep_300
    - restart_from_baseline

    Return ONLY JSON with this schema:
    {{
      "decision": "continue",
      "health": "healthy|degraded|broken",
      "root_cause": "short diagnosis",
      "action": "noop|sleep_60|sleep_300|restart_from_baseline",
      "notes": "short operator notes"
    }}
    """
)


@dataclass
class PricingSpec:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


@dataclass
class Config:
    workspace: Path
    strategy_file: Path
    baseline_strategy_file: Path | None
    state_dir: Path
    prompt_template_path: Path | None
    max_iterations: int
    stop_on_target: bool
    agent_timeout_sec: int
    task_timeout_sec: int
    stagnation_window: int
    stagnation_min_delta: float
    diversification_interval: int
    restart_interval: int
    elite_pool_size: int
    budget_max_usd: float
    budget_fallback_per_iteration: float
    budget_fallback_on_failure: float
    pricing: PricingSpec | None
    agent_backend: str
    agent_command_template: str
    agent_use_shell: bool
    task_command_template: str
    task_use_shell: bool
    sysadmin_enabled: bool
    sysadmin_interval_seconds: int
    sysadmin_failure_streak_trigger: int
    sysadmin_timeout_sec: int
    sysadmin_prompt_template_path: Path | None
    sysadmin_command_template: str
    sysadmin_use_shell: bool


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool


class HarnessError(RuntimeError):
    pass


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> Config:
    raw = tomllib.loads(read_text(path))

    paths = raw.get("paths", {})
    loop = raw.get("loop", {})
    budget = raw.get("budget", {})
    agent = raw.get("agent", {})
    task = raw.get("task", {})
    sysadmin = raw.get("sysadmin", {})

    workspace = Path(paths.get("workspace", ".")).expanduser().resolve()

    strategy_file_raw = paths.get("strategy_file")
    if not strategy_file_raw:
        raise HarnessError("paths.strategy_file is required")
    strategy_file = (workspace / strategy_file_raw).resolve()

    baseline_raw = paths.get("baseline_strategy_file")
    baseline_strategy_file = (
        (workspace / baseline_raw).resolve() if baseline_raw is not None else None
    )

    state_dir_raw = paths.get("state_dir", ".harness")
    state_dir = (workspace / state_dir_raw).resolve()

    prompt_template_raw = paths.get("prompt_template")
    prompt_template_path = (
        (workspace / prompt_template_raw).resolve()
        if prompt_template_raw is not None
        else None
    )

    sysadmin_prompt_template_raw = sysadmin.get("prompt_template")
    sysadmin_prompt_template_path = (
        (workspace / str(sysadmin_prompt_template_raw)).resolve()
        if sysadmin_prompt_template_raw is not None
        else None
    )

    pricing = None
    pricing_map: dict[str, float] | None = None
    if "pricing" in raw:
        p = raw["pricing"]
        pricing = PricingSpec(
            input_per_million=float(p["input_per_million"]),
            cached_input_per_million=float(p["cached_input_per_million"]),
            output_per_million=float(p["output_per_million"]),
        )
        pricing_map = {
            "input_per_million": pricing.input_per_million,
            "cached_input_per_million": pricing.cached_input_per_million,
            "output_per_million": pricing.output_per_million,
        }

    try:
        resolved_agent_backend = resolve_agent_backend(agent, pricing=pricing_map)
    except AgentBackendError as exc:
        raise HarnessError(str(exc)) from exc

    task_command_template = str(task.get("command_template", "")).strip()
    if not task_command_template:
        raise HarnessError("task.command_template is required")

    sysadmin_enabled = bool(sysadmin.get("enabled", False))
    default_sysadmin_command = (
        "python3 harness/agents/openai_sysadmin.py "
        "--prompt-file {prompt_file} "
        "--model $SYSADMIN_MODEL "
        "--fallback-model $SYSADMIN_MODEL "
        "--reasoning-effort high "
    )
    sysadmin_command_template = str(
        sysadmin.get("command_template", default_sysadmin_command)
    ).strip()

    cfg = Config(
        workspace=workspace,
        strategy_file=strategy_file,
        baseline_strategy_file=baseline_strategy_file,
        state_dir=state_dir,
        prompt_template_path=prompt_template_path,
        max_iterations=int(loop.get("max_iterations", 200)),
        stop_on_target=bool(loop.get("stop_on_target", True)),
        agent_timeout_sec=int(loop.get("agent_timeout_sec", 1800)),
        task_timeout_sec=int(loop.get("task_timeout_sec", 1800)),
        stagnation_window=int(loop.get("stagnation_window", 8)),
        stagnation_min_delta=float(loop.get("stagnation_min_delta", 2.0)),
        diversification_interval=int(loop.get("diversification_interval", 5)),
        restart_interval=int(loop.get("restart_interval", 11)),
        elite_pool_size=int(loop.get("elite_pool_size", 6)),
        budget_max_usd=float(budget.get("max_usd", 1000.0)),
        budget_fallback_per_iteration=float(budget.get("fallback_per_iteration_usd", 5.0)),
        budget_fallback_on_failure=float(budget.get("fallback_on_failure_usd", 0.0)),
        pricing=pricing,
        agent_backend=resolved_agent_backend.name,
        agent_command_template=resolved_agent_backend.command_template,
        agent_use_shell=resolved_agent_backend.use_shell,
        task_command_template=task_command_template,
        task_use_shell=bool(task.get("use_shell", True)),
        sysadmin_enabled=sysadmin_enabled,
        sysadmin_interval_seconds=int(sysadmin.get("interval_seconds", 600)),
        sysadmin_failure_streak_trigger=int(sysadmin.get("failure_streak_trigger", 0)),
        sysadmin_timeout_sec=int(sysadmin.get("timeout_sec", 300)),
        sysadmin_prompt_template_path=sysadmin_prompt_template_path,
        sysadmin_command_template=sysadmin_command_template,
        sysadmin_use_shell=bool(sysadmin.get("use_shell", True)),
    )

    if cfg.max_iterations < 0:
        raise HarnessError("loop.max_iterations must be >= 0 (0 means unbounded)")
    if cfg.agent_timeout_sec <= 0:
        raise HarnessError("loop.agent_timeout_sec must be > 0")
    if cfg.task_timeout_sec <= 0:
        raise HarnessError("loop.task_timeout_sec must be > 0")
    if cfg.stagnation_window < 2:
        raise HarnessError("loop.stagnation_window must be >= 2")
    if cfg.diversification_interval < 2:
        raise HarnessError("loop.diversification_interval must be >= 2")
    if cfg.restart_interval < 2:
        raise HarnessError("loop.restart_interval must be >= 2")
    if cfg.elite_pool_size < 1:
        raise HarnessError("loop.elite_pool_size must be >= 1")
    if cfg.budget_max_usd < 0:
        raise HarnessError("budget.max_usd must be >= 0 (0 disables internal cap)")
    if cfg.budget_fallback_per_iteration < 0:
        raise HarnessError("budget.fallback_per_iteration_usd must be >= 0")
    if cfg.budget_fallback_on_failure < 0:
        raise HarnessError("budget.fallback_on_failure_usd must be >= 0")
    if cfg.sysadmin_interval_seconds < 1:
        raise HarnessError("sysadmin.interval_seconds must be >= 1")
    if cfg.sysadmin_failure_streak_trigger < 0:
        raise HarnessError("sysadmin.failure_streak_trigger must be >= 0")
    if cfg.sysadmin_timeout_sec <= 0:
        raise HarnessError("sysadmin.timeout_sec must be > 0")

    if not cfg.workspace.exists():
        raise HarnessError(f"workspace does not exist: {cfg.workspace}")
    if not cfg.strategy_file.exists():
        raise HarnessError(f"strategy file does not exist: {cfg.strategy_file}")
    if cfg.baseline_strategy_file is not None and not cfg.baseline_strategy_file.exists():
        raise HarnessError(
            f"baseline strategy file does not exist: {cfg.baseline_strategy_file}"
        )
    if cfg.prompt_template_path is not None and not cfg.prompt_template_path.exists():
        raise HarnessError(f"prompt template does not exist: {cfg.prompt_template_path}")
    if (
        cfg.sysadmin_prompt_template_path is not None
        and not cfg.sysadmin_prompt_template_path.exists()
    ):
        raise HarnessError(
            f"sysadmin prompt template does not exist: {cfg.sysadmin_prompt_template_path}"
        )
    if cfg.sysadmin_enabled and not cfg.sysadmin_command_template:
        raise HarnessError("sysadmin.command_template is required when sysadmin.enabled=true")

    return cfg


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "version": 1,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "budget_spent_usd": 0.0,
            "cost_policy_version": 2,
            "iterations": [],
            "sysadmin_checks": [],
            "elites": [],
            "best_train_avg": float("-inf"),
            "best_holdout_avg": float("-inf"),
            "best_holdout_candidate": None,
            "target_label": None,
            "stopped_reason": None,
        }
    return json.loads(read_text(state_path))


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_text(state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")


def run_command(
    command: str,
    *,
    cwd: Path,
    timeout_sec: int,
    use_shell: bool,
) -> CommandResult:
    started = datetime.now(timezone.utc)
    try:
        if use_shell:
            proc = subprocess.run(
                command,
                cwd=cwd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        else:
            proc = subprocess.run(
                shlex.split(command),
                cwd=cwd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )

        duration = (datetime.now(timezone.utc) - started).total_seconds()
        return CommandResult(
            command=command,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_sec=duration,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = (datetime.now(timezone.utc) - started).total_seconds()
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return CommandResult(
            command=command,
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            duration_sec=duration,
            timed_out=True,
        )


def extract_first(patterns: list[re.Pattern[str]], text: str) -> int | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def parse_agent_cost(text: str, pricing: PricingSpec | None) -> float | None:
    latest_match: tuple[int, float] | None = None
    for pattern in COST_PATTERNS:
        for match in pattern.finditer(text):
            value = float(match.group(1))
            latest_match = (match.start(), value)

    if latest_match is not None:
        return latest_match[1]

    if pricing is None:
        return None

    input_tokens = extract_first(TOKEN_PATTERNS["input"], text)
    cached_input_tokens = extract_first(TOKEN_PATTERNS["cached_input"], text) or 0
    output_tokens = extract_first(TOKEN_PATTERNS["output"], text)

    if input_tokens is None and output_tokens is None:
        return None

    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0

    return (
        (input_tokens / 1_000_000.0) * pricing.input_per_million
        + (cached_input_tokens / 1_000_000.0) * pricing.cached_input_per_million
        + (output_tokens / 1_000_000.0) * pricing.output_per_million
    )


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def safe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def compute_iteration_cost(
    *,
    parsed_cost: float | None,
    fallback_per_iteration: float,
    fallback_on_failure: float,
    agent_exit_code: int | None,
    agent_timed_out: bool,
) -> tuple[float, str]:
    if parsed_cost is not None:
        return parsed_cost, "parsed"
    if agent_timed_out or agent_exit_code == 0:
        return fallback_per_iteration, "fallback_per_iteration"
    return fallback_on_failure, "fallback_on_failure"


def budget_limit_enabled(cfg: Config) -> bool:
    return cfg.budget_max_usd > 0


def migrate_cost_accounting(state: dict[str, Any], cfg: Config) -> bool:
    if int(state.get("cost_policy_version", 1)) >= 2:
        return False

    iterations = state.get("iterations", [])
    if not isinstance(iterations, list):
        state["cost_policy_version"] = 2
        state["budget_spent_usd"] = 0.0
        return True

    running_spend = 0.0
    for item in iterations:
        if not isinstance(item, dict):
            continue
        agent = item.get("agent")
        if not isinstance(agent, dict):
            agent = {}
            item["agent"] = agent

        parsed_cost = safe_float(agent.get("cost_usd")) if bool(agent.get("cost_parsed")) else None
        iter_cost, cost_source = compute_iteration_cost(
            parsed_cost=parsed_cost,
            fallback_per_iteration=cfg.budget_fallback_per_iteration,
            fallback_on_failure=cfg.budget_fallback_on_failure,
            agent_exit_code=safe_int(agent.get("exit_code")),
            agent_timed_out=bool(agent.get("timed_out", False)),
        )
        running_spend += iter_cost

        agent["cost_usd"] = iter_cost
        agent["cost_source"] = cost_source
        item["budget_spent_usd"] = running_spend

    state["budget_spent_usd"] = running_spend
    if state.get("stopped_reason") == "budget_exhausted" and (
        not budget_limit_enabled(cfg) or running_spend < cfg.budget_max_usd
    ):
        state["stopped_reason"] = None
    state["cost_policy_version"] = 2
    return True


def recent_history_summary(state: dict[str, Any], count: int = 8) -> str:
    iterations = state.get("iterations", [])
    if not iterations:
        return "- none yet"

    lines = []
    for item in iterations[-count:]:
        lines.append(
            "- i={i} mode={mode} train_avg={train_avg} promoted={promoted} holdout_avg={holdout} passed={passed}".format(
                i=item.get("iteration"),
                mode=item.get("mode"),
                train_avg=(
                    "n/a"
                    if item.get("train_avg") is None
                    else f"{float(item['train_avg']):.2f}"
                ),
                promoted=bool(item.get("promoted_to_holdout", False)),
                holdout=(
                    "n/a"
                    if item.get("holdout_avg") is None
                    else f"{float(item['holdout_avg']):.2f}"
                ),
                passed=bool(item.get("passed_target", False)),
            )
        )

    return "\n".join(lines)


def elite_summary(state: dict[str, Any], count: int = 5) -> str:
    elites = state.get("elites", [])
    if not elites:
        return "- none yet"

    lines = []
    for item in elites[:count]:
        lines.append(
            "- i={iteration} train_avg={train_avg:.2f} train_worst={train_worst:.2f} hash={digest} file={path}".format(
                iteration=item["iteration"],
                train_avg=item["train_avg"],
                train_worst=item["train_worst"],
                digest=item["strategy_hash"][:10],
                path=item["candidate_file"],
            )
        )
    return "\n".join(lines)


def recent_agent_failure_streak(state: dict[str, Any]) -> int:
    streak = 0
    for item in reversed(state.get("iterations", [])):
        agent = item.get("agent", {})
        task = item.get("task", {})
        eval_result = task.get("result", {}) if isinstance(task, dict) else {}
        validation = eval_result.get("validation", {}) if isinstance(eval_result, dict) else {}
        if not isinstance(agent, dict):
            break
        exit_code = safe_int(agent.get("exit_code"))
        timed_out = bool(agent.get("timed_out", False))
        evaluate_exit = safe_int(task.get("evaluate_command_exit")) if isinstance(task, dict) else None
        validation_exit = safe_int(validation.get("exit_code")) if isinstance(validation, dict) else None
        eval_error = bool(eval_result.get("error")) if isinstance(eval_result, dict) else False

        agent_failed = exit_code != 0 or timed_out
        task_failed = evaluate_exit not in (None, 0) or validation_exit not in (None, 0) or eval_error
        if not agent_failed and not task_failed:
            break
        streak += 1
    return streak


def recent_failure_summary(state: dict[str, Any], count: int = 5) -> str:
    rows: list[str] = []
    for item in reversed(state.get("iterations", [])):
        if len(rows) >= count:
            break
        agent = item.get("agent", {})
        task = item.get("task", {})
        eval_result = task.get("result", {}) if isinstance(task, dict) else {}
        agent_exit = agent.get("exit_code") if isinstance(agent, dict) else None
        eval_error = (
            eval_result.get("error")
            if isinstance(eval_result, dict)
            else None
        )
        validation = eval_result.get("validation") if isinstance(eval_result, dict) else {}
        validation_exit = (
            validation.get("exit_code")
            if isinstance(validation, dict)
            else None
        )
        if agent_exit in (0, None) and not eval_error and validation_exit in (0, None):
            continue
        rows.append(
            "- i={i} agent_exit={agent_exit} eval_error={eval_error} validation_exit={validation_exit}".format(
                i=item.get("iteration"),
                agent_exit=agent_exit,
                eval_error=(str(eval_error)[:120] if eval_error is not None else "n/a"),
                validation_exit=validation_exit,
            )
        )
    if not rows:
        return "- none"
    return "\n".join(rows)


def render_sysadmin_prompt(
    cfg: Config,
    state: dict[str, Any],
    *,
    iteration: int,
    iter_dir: Path,
    state_path: Path,
    target_label: str,
) -> str:
    template = (
        read_text(cfg.sysadmin_prompt_template_path)
        if cfg.sysadmin_prompt_template_path is not None
        else DEFAULT_SYSADMIN_PROMPT_TEMPLATE
    )
    return template.format(
        iteration=iteration,
        target_label=target_label or "pass objective gate",
        failure_streak=recent_agent_failure_streak(state),
        iter_dir=str(iter_dir),
        state_path=str(state_path),
        strategy_file=str(cfg.strategy_file),
        recent_history=recent_history_summary(state, count=10),
        recent_failures=recent_failure_summary(state, count=8),
    )


def parse_first_json_object(text: str) -> dict[str, Any] | None:
    payload = text.strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", payload)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def parse_iso_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def should_run_sysadmin(cfg: Config, state: dict[str, Any], iteration: int) -> tuple[bool, str]:
    if not cfg.sysadmin_enabled:
        return False, "disabled"

    last = safe_int(state.get("sysadmin_last_iteration"))
    if last is not None and last == iteration:
        return False, "already_checked"

    if iteration == 0:
        return True, "startup"

    last_ts = parse_iso_timestamp(state.get("sysadmin_last_timestamp"))
    if last_ts is None:
        return True, "first_check"
    elapsed_sec = int((datetime.now(timezone.utc) - last_ts).total_seconds())
    if elapsed_sec >= cfg.sysadmin_interval_seconds:
        return True, "interval"

    if (
        cfg.sysadmin_failure_streak_trigger > 0
        and recent_agent_failure_streak(state) >= cfg.sysadmin_failure_streak_trigger
    ):
        return True, "failure_streak"

    return False, "not_due"


def apply_sysadmin_action(cfg: Config, action: str) -> str:
    if action == "noop":
        return "noop"
    if action == "sleep_60":
        time.sleep(60)
        return "slept_60s"
    if action == "sleep_300":
        time.sleep(300)
        return "slept_300s"
    if action == "restart_from_baseline":
        if (
            cfg.baseline_strategy_file is not None
            and cfg.baseline_strategy_file.exists()
            and cfg.baseline_strategy_file.resolve() != cfg.strategy_file.resolve()
        ):
            shutil.copy2(cfg.baseline_strategy_file, cfg.strategy_file)
            return "reseeded_from_baseline"
        restore = run_command(
            f"git checkout -- {shlex.quote(str(cfg.strategy_file))}",
            cwd=cfg.workspace,
            timeout_sec=60,
            use_shell=True,
        )
        if restore.exit_code == 0:
            return "reseeded_from_git_head"
        return "restart_from_baseline_skipped_missing_baseline"
    return f"unknown_action:{action}"


def run_sysadmin_check(
    cfg: Config,
    state: dict[str, Any],
    *,
    iteration: int,
    iter_dir: Path,
    state_path: Path,
    target_label: str,
) -> dict[str, Any]:
    sys_dir = iter_dir / "sysadmin"
    sys_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = render_sysadmin_prompt(
        cfg,
        state,
        iteration=iteration,
        iter_dir=iter_dir,
        state_path=state_path,
        target_label=target_label,
    )
    prompt_path = sys_dir / "sysadmin_prompt.md"
    write_text(prompt_path, prompt_text)

    command = cfg.sysadmin_command_template.format(
        prompt_file=shlex.quote(str(prompt_path)),
        state_file=shlex.quote(str(state_path)),
        strategy_file=shlex.quote(str(cfg.strategy_file)),
        workspace=shlex.quote(str(cfg.workspace)),
        iteration=iteration,
        iter_dir=shlex.quote(str(iter_dir)),
    )
    result = run_command(
        command,
        cwd=cfg.workspace,
        timeout_sec=cfg.sysadmin_timeout_sec,
        use_shell=cfg.sysadmin_use_shell,
    )
    write_text(sys_dir / "stdout.log", result.stdout)
    write_text(sys_dir / "stderr.log", result.stderr)

    parsed = parse_first_json_object(result.stdout)
    action = "noop"
    decision = "continue"
    health = "degraded"
    root_cause = "sysadmin output unavailable"
    notes = ""
    combined_output = (result.stdout + "\n" + result.stderr).lower()

    if "insufficient_quota" in combined_output or "rate limit" in combined_output:
        action = "sleep_300"
        health = "broken"
        root_cause = "OpenAI quota/rate-limit while running sysadmin check"
    elif result.timed_out or result.exit_code != 0:
        action = "sleep_60"
        root_cause = "sysadmin check command failed"

    if parsed is not None:
        decision = str(parsed.get("decision", "continue")).strip().lower() or "continue"
        health = str(parsed.get("health", "degraded")).strip().lower() or "degraded"
        root_cause = str(parsed.get("root_cause", "n/a")).strip()
        notes = str(parsed.get("notes", "")).strip()
        action = str(parsed.get("action", "noop")).strip().lower() or "noop"

    applied = apply_sysadmin_action(cfg, action)

    return {
        "timestamp": now_iso(),
        "iteration": iteration,
        "trigger": state.get("sysadmin_trigger_reason"),
        "command": command,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "duration_sec": result.duration_sec,
        "decision": decision,
        "health": health,
        "root_cause": root_cause,
        "notes": notes,
        "requested_action": action,
        "applied_action": applied,
        "raw_stdout_path": str(sys_dir / "stdout.log"),
        "raw_stderr_path": str(sys_dir / "stderr.log"),
    }


def choose_mode(state: dict[str, Any], cfg: Config, iteration: int) -> str:
    if iteration > 0 and iteration % cfg.restart_interval == 0:
        return "restart"

    if iteration > 0 and iteration % cfg.diversification_interval == 0:
        return "diversify"

    scores = [
        it["train_avg"]
        for it in state.get("iterations", [])
        if isinstance(it.get("train_avg"), (int, float))
    ]

    if len(scores) >= cfg.stagnation_window:
        prior = scores[:-cfg.stagnation_window]
        recent = scores[-cfg.stagnation_window :]
        prior_best = max(prior) if prior else recent[0]
        recent_best = max(recent)
        if recent_best - prior_best < cfg.stagnation_min_delta:
            return "diversify"

    return "exploit"


def maybe_reset_strategy(cfg: Config, state: dict[str, Any], mode: str) -> str | None:
    if mode != "restart":
        return None

    source: Path | None = None
    elites = state.get("elites", [])

    if len(elites) >= 2:
        source = Path(elites[1]["candidate_file"])
    elif cfg.baseline_strategy_file is not None:
        source = cfg.baseline_strategy_file

    if source is None or not source.exists():
        return None

    # Restart mode may point to the same file (e.g. baseline == strategy file).
    # Skip reset in that case to avoid SameFileError.
    if source.resolve() == cfg.strategy_file.resolve():
        return None

    shutil.copy2(source, cfg.strategy_file)
    return str(source)


def mode_instructions(mode: str) -> str:
    if mode == "exploit":
        return "Refine the current best idea and harden edge cases."
    if mode == "diversify":
        return "Make a materially different strategy move; avoid small parameter nudges."
    if mode == "restart":
        return "Restart from a different anchor and avoid immediately reconverging."
    return "Improve robustness."


def render_prompt(
    cfg: Config,
    state: dict[str, Any],
    *,
    iteration: int,
    mode: str,
    task_context: str,
    target_label: str,
) -> str:
    template = (
        read_text(cfg.prompt_template_path)
        if cfg.prompt_template_path is not None
        else DEFAULT_PROMPT_TEMPLATE
    )

    spent = float(state.get("budget_spent_usd", 0.0))
    remaining = (
        max(0.0, cfg.budget_max_usd - spent)
        if budget_limit_enabled(cfg)
        else float("inf")
    )

    return template.format(
        iteration=iteration,
        mode=mode,
        mode_instructions=mode_instructions(mode),
        strategy_file=str(cfg.strategy_file),
        task_context=task_context,
        target_label=target_label,
        budget_remaining=remaining,
        budget_spent=spent,
        recent_history=recent_history_summary(state),
        recent_failures=recent_failure_summary(state),
        elite_summary=elite_summary(state),
    )


def build_template_vars(
    cfg: Config,
    *,
    action: str,
    mode: str,
    iteration: int,
    iter_dir: Path,
) -> dict[str, Any]:
    return {
        "action": action,
        "workspace": shlex.quote(str(cfg.workspace)),
        "strategy_file": shlex.quote(str(cfg.strategy_file)),
        "mode": mode,
        "iteration": iteration,
        "state_dir": shlex.quote(str(cfg.state_dir)),
        "iter_dir": shlex.quote(str(iter_dir)),
    }


def run_task_action(
    cfg: Config,
    *,
    action: str,
    mode: str,
    iteration: int,
    iter_dir: Path,
) -> CommandResult:
    command = cfg.task_command_template.format(**build_template_vars(
        cfg,
        action=action,
        mode=mode,
        iteration=iteration,
        iter_dir=iter_dir,
    ))
    return run_command(
        command,
        cwd=cfg.workspace,
        timeout_sec=cfg.task_timeout_sec,
        use_shell=cfg.task_use_shell,
    )


def parse_json_stdout(result: CommandResult, *, action: str) -> dict[str, Any]:
    if result.exit_code != 0:
        raise HarnessError(
            f"task action '{action}' failed (exit={result.exit_code} timeout={result.timed_out})"
        )

    payload = result.stdout.strip()
    if not payload:
        raise HarnessError(f"task action '{action}' returned empty stdout")

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HarnessError(f"task action '{action}' did not return JSON") from exc

    if not isinstance(parsed, dict):
        raise HarnessError(f"task action '{action}' JSON root must be object")

    return parsed


def update_elites(
    state: dict[str, Any],
    *,
    iteration: int,
    candidate_file: Path,
    strategy_hash: str,
    train_avg: float,
    train_worst: float,
    cfg: Config,
) -> None:
    elites = state.setdefault("elites", [])
    if any(item.get("strategy_hash") == strategy_hash for item in elites):
        return

    elites.append(
        {
            "iteration": iteration,
            "candidate_file": str(candidate_file),
            "strategy_hash": strategy_hash,
            "train_avg": train_avg,
            "train_worst": train_worst,
            "added_at": now_iso(),
        }
    )
    elites.sort(key=lambda item: (item["train_avg"], item["train_worst"]), reverse=True)
    del elites[cfg.elite_pool_size :]


def print_iteration_header(
    iteration: int,
    mode: str,
    budget_spent: float,
    budget_max: float,
    *,
    budget_enabled: bool,
) -> None:
    budget_label = f"${budget_max:.2f}" if budget_enabled else "unlimited"
    print(
        f"\n=== Iteration {iteration} | mode={mode} | budget=${budget_spent:.2f}/{budget_label} ==="
    )


def run_harness(cfg: Config) -> int:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    iterations_dir = cfg.state_dir / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)

    state_path = cfg.state_dir / "state.json"
    state = load_state(state_path)
    if state.get("stopped_reason") == "budget_exhausted" and (
        not budget_limit_enabled(cfg)
        or float(state.get("budget_spent_usd", 0.0)) < cfg.budget_max_usd
    ):
        state["stopped_reason"] = None
        save_state(state_path, state)
    if cfg.max_iterations == 0 and state.get("stopped_reason") == "max_iterations_reached":
        state["stopped_reason"] = None
        save_state(state_path, state)
    if migrate_cost_accounting(state, cfg):
        save_state(state_path, state)

    start_iter = len(state.get("iterations", []))
    iteration_source = (
        itertools.count(start_iter)
        if cfg.max_iterations == 0
        else range(start_iter, cfg.max_iterations)
    )

    for iteration in iteration_source:
        spent = float(state.get("budget_spent_usd", 0.0))
        if budget_limit_enabled(cfg) and spent >= cfg.budget_max_usd:
            state["stopped_reason"] = "budget_exhausted"
            save_state(state_path, state)
            print("Budget exhausted before next iteration.")
            return 2

        mode = choose_mode(state, cfg, iteration)
        print_iteration_header(
            iteration,
            mode,
            spent,
            cfg.budget_max_usd,
            budget_enabled=budget_limit_enabled(cfg),
        )

        iter_dir = iterations_dir / f"iter_{iteration:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        reset_source = maybe_reset_strategy(cfg, state, mode)
        if reset_source is not None:
            print(f"Restart mode: seeded strategy from {reset_source}")

        context_result = run_task_action(
            cfg,
            action="context",
            mode=mode,
            iteration=iteration,
            iter_dir=iter_dir,
        )
        write_text(iter_dir / "task_context_stdout.log", context_result.stdout)
        write_text(iter_dir / "task_context_stderr.log", context_result.stderr)
        context_payload = parse_json_stdout(context_result, action="context")

        task_context = str(context_payload.get("prompt_context", "")).strip()
        if not task_context:
            task_context = "- no task context provided"
        target_label = str(context_payload.get("target_label", "")).strip()
        if not target_label:
            target_label = "pass objective gate"
        state["target_label"] = target_label

        prompt = render_prompt(
            cfg,
            state,
            iteration=iteration,
            mode=mode,
            task_context=task_context,
            target_label=target_label,
        )
        prompt_path = iter_dir / "agent_prompt.md"
        write_text(prompt_path, prompt)

        agent_command = cfg.agent_command_template.format(
            prompt_file=shlex.quote(str(prompt_path)),
            strategy_file=shlex.quote(str(cfg.strategy_file)),
            workspace=shlex.quote(str(cfg.workspace)),
            iteration=iteration,
            mode=mode,
            state_dir=shlex.quote(str(cfg.state_dir)),
            iter_dir=shlex.quote(str(iter_dir)),
        )

        agent_result = run_command(
            agent_command,
            cwd=cfg.workspace,
            timeout_sec=cfg.agent_timeout_sec,
            use_shell=cfg.agent_use_shell,
        )
        write_text(iter_dir / "agent_stdout.log", agent_result.stdout)
        write_text(iter_dir / "agent_stderr.log", agent_result.stderr)

        combined_output = agent_result.stdout + "\n" + agent_result.stderr
        parsed_cost = parse_agent_cost(combined_output, cfg.pricing)
        iter_cost, cost_source = compute_iteration_cost(
            parsed_cost=parsed_cost,
            fallback_per_iteration=cfg.budget_fallback_per_iteration,
            fallback_on_failure=cfg.budget_fallback_on_failure,
            agent_exit_code=agent_result.exit_code,
            agent_timed_out=agent_result.timed_out,
        )
        state["budget_spent_usd"] = float(state.get("budget_spent_usd", 0.0)) + iter_cost

        print(
            f"Agent exit={agent_result.exit_code} timeout={agent_result.timed_out} "
            f"dur={agent_result.duration_sec:.1f}s cost=${iter_cost:.2f} ({cost_source})"
        )

        if not cfg.strategy_file.exists():
            raise HarnessError(f"Strategy file missing after agent run: {cfg.strategy_file}")

        candidate_file = iter_dir / "candidate_snapshot"
        ext = cfg.strategy_file.suffix or ".txt"
        candidate_file = candidate_file.with_suffix(ext)
        shutil.copy2(cfg.strategy_file, candidate_file)
        strategy_hash = sha256_file(candidate_file)

        eval_result = run_task_action(
            cfg,
            action="evaluate",
            mode=mode,
            iteration=iteration,
            iter_dir=iter_dir,
        )
        write_text(iter_dir / "task_eval_stdout.log", eval_result.stdout)
        write_text(iter_dir / "task_eval_stderr.log", eval_result.stderr)

        eval_payload: dict[str, Any]
        if eval_result.exit_code == 0:
            eval_payload = parse_json_stdout(eval_result, action="evaluate")
        else:
            eval_payload = {
                "ok": False,
                "passed_target": False,
                "error": (
                    f"task evaluate failed exit={eval_result.exit_code} "
                    f"timeout={eval_result.timed_out}"
                ),
            }

        train_avg = safe_float(eval_payload.get("train_avg"))
        train_worst = safe_float(eval_payload.get("train_worst"))
        holdout_avg = safe_float(eval_payload.get("holdout_avg"))
        promoted = bool(eval_payload.get("promoted_to_holdout", False))
        passed = bool(eval_payload.get("passed_target", False))

        if train_avg is not None and train_worst is not None:
            update_elites(
                state,
                iteration=iteration,
                candidate_file=candidate_file,
                strategy_hash=strategy_hash,
                train_avg=train_avg,
                train_worst=train_worst,
                cfg=cfg,
            )
            best_train = float(state.get("best_train_avg", float("-inf")))
            if train_avg > best_train:
                state["best_train_avg"] = train_avg

        if holdout_avg is not None:
            best_holdout = float(state.get("best_holdout_avg", float("-inf")))
            if holdout_avg > best_holdout:
                state["best_holdout_avg"] = holdout_avg
                state["best_holdout_candidate"] = str(candidate_file)

        iteration_record = {
            "iteration": iteration,
            "timestamp": now_iso(),
            "mode": mode,
            "reset_source": reset_source,
            "strategy_hash": strategy_hash,
            "candidate_file": str(candidate_file),
            "agent": {
                "backend": cfg.agent_backend,
                "command": agent_command,
                "exit_code": agent_result.exit_code,
                "timed_out": agent_result.timed_out,
                "duration_sec": agent_result.duration_sec,
                "cost_usd": iter_cost,
                "cost_parsed": parsed_cost is not None,
                "cost_source": cost_source,
            },
            "task": {
                "context_command_exit": context_result.exit_code,
                "evaluate_command_exit": eval_result.exit_code,
                "evaluate_timed_out": eval_result.timed_out,
                "result": eval_payload,
            },
            "train_avg": train_avg,
            "train_worst": train_worst,
            "promoted_to_holdout": promoted,
            "holdout_avg": holdout_avg,
            "passed_target": passed,
            "budget_spent_usd": state.get("budget_spent_usd"),
        }
        state.setdefault("iterations", []).append(iteration_record)
        save_state(state_path, state)

        run_guard, trigger = should_run_sysadmin(cfg, state, iteration)
        if run_guard:
            state["sysadmin_trigger_reason"] = trigger
            sysadmin_record = run_sysadmin_check(
                cfg,
                state,
                iteration=iteration,
                iter_dir=iter_dir,
                state_path=state_path,
                target_label=target_label,
            )
            state.pop("sysadmin_trigger_reason", None)
            state["sysadmin_last_iteration"] = iteration
            state["sysadmin_last_timestamp"] = sysadmin_record.get("timestamp")
            state.setdefault("sysadmin_checks", []).append(sysadmin_record)
            state["iterations"][-1]["sysadmin"] = sysadmin_record
            save_state(state_path, state)
            print(
                "Sysadmin check: trigger={trigger} health={health} action={action} cause={cause}".format(
                    trigger=trigger,
                    health=sysadmin_record.get("health"),
                    action=sysadmin_record.get("applied_action"),
                    cause=sysadmin_record.get("root_cause"),
                )
            )

        print(
            "Eval summary: train_avg={train_avg} train_worst={train_worst} "
            "holdout_avg={holdout_avg} promoted={promoted} passed={passed}".format(
                train_avg="n/a" if train_avg is None else f"{train_avg:.2f}",
                train_worst="n/a" if train_worst is None else f"{train_worst:.2f}",
                holdout_avg="n/a" if holdout_avg is None else f"{holdout_avg:.2f}",
                promoted=promoted,
                passed=passed,
            )
        )

        if passed:
            if cfg.stop_on_target:
                state["stopped_reason"] = "target_reached"
                save_state(state_path, state)
                print(f"SUCCESS: reached target ({target_label}).")
                return 0
            print(f"Target reached ({target_label}); continuing because stop_on_target=false.")

        if budget_limit_enabled(cfg) and float(state.get("budget_spent_usd", 0.0)) >= cfg.budget_max_usd:
            state["stopped_reason"] = "budget_exhausted"
            save_state(state_path, state)
            print("Budget exhausted; stopping.")
            return 2

    state["stopped_reason"] = "max_iterations_reached"
    save_state(state_path, state)
    print("Reached max iterations without hitting target.")
    return 3


def dry_run(cfg: Config) -> int:
    iter_dir = cfg.state_dir / "dry_run"
    iter_dir.mkdir(parents=True, exist_ok=True)

    context_result = run_task_action(
        cfg,
        action="context",
        mode="exploit",
        iteration=0,
        iter_dir=iter_dir,
    )

    print("Dry run summary")
    print(f"workspace: {cfg.workspace}")
    print(f"strategy_file: {cfg.strategy_file}")
    print(f"state_dir: {cfg.state_dir}")
    print(f"max_iterations: {cfg.max_iterations}")
    print(f"stop_on_target: {cfg.stop_on_target}")
    print(f"agent_backend: {cfg.agent_backend}")
    print(f"agent_command_template: {cfg.agent_command_template}")
    print(f"task_command_template: {cfg.task_command_template}")
    print(f"sysadmin_enabled: {cfg.sysadmin_enabled}")
    if cfg.sysadmin_enabled:
        print(f"sysadmin_command_template: {cfg.sysadmin_command_template}")
        print(f"sysadmin_interval_seconds: {cfg.sysadmin_interval_seconds}")
        print(f"sysadmin_failure_streak_trigger: {cfg.sysadmin_failure_streak_trigger}")

    if context_result.exit_code != 0:
        print("\nTask context call failed.")
        print(context_result.stderr.strip())
        return 1

    context_payload = parse_json_stdout(context_result, action="context")

    state = {
        "iterations": [],
        "elites": [],
        "budget_spent_usd": 0.0,
    }
    prompt = render_prompt(
        cfg,
        state,
        iteration=0,
        mode="exploit",
        task_context=str(context_payload.get("prompt_context", "- none")),
        target_label=str(context_payload.get("target_label", "pass objective gate")),
    )

    print("\n--- Prompt Preview ---")
    print(prompt)
    print("--- End Prompt Preview ---")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic cloud/local autonomous coding loop")
    parser.add_argument("--config", required=True, type=Path, help="Path to TOML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, call task context, and print prompt preview",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config.resolve())
    if args.dry_run:
        return dry_run(cfg)
    return run_harness(cfg)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
