#!/usr/bin/env python3
"""Agent backend registry for the generic harness loop."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentBackendSpec:
    name: str
    command_template: str
    use_shell: bool


class AgentBackendError(ValueError):
    """Raised when agent backend config is invalid."""


def resolve_agent_backend(
    agent_cfg: Mapping[str, Any],
    *,
    pricing: Mapping[str, float] | None,
) -> AgentBackendSpec:
    explicit_command = str(agent_cfg.get("command_template", "")).strip()
    use_shell = _parse_bool(agent_cfg.get("use_shell", True), field="agent.use_shell")
    backend_name = str(agent_cfg.get("backend", "")).strip().lower()

    # Backward-compatible path: explicit command templates continue to work.
    if explicit_command:
        return AgentBackendSpec(
            name=backend_name or "command_template",
            command_template=explicit_command,
            use_shell=use_shell,
        )

    resolved_backend = backend_name or "openai_ops"
    options = agent_cfg.get("backend_options", {})
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise AgentBackendError("agent.backend_options must be a TOML table")

    builder = _BACKEND_BUILDERS.get(resolved_backend)
    if builder is None:
        supported = ", ".join(sorted(_BACKEND_BUILDERS))
        raise AgentBackendError(
            f"Unsupported agent.backend '{resolved_backend}'. Supported: {supported}"
        )

    command = builder(options, pricing=pricing)
    if not command.strip():
        raise AgentBackendError("Resolved empty command template for agent backend")

    return AgentBackendSpec(
        name=resolved_backend,
        command_template=command,
        use_shell=use_shell,
    )


def _build_openai_ops_backend(
    options: Mapping[str, Any], *, pricing: Mapping[str, float] | None
) -> str:
    rates = _pricing_defaults(pricing)
    model_expr = _shell_expr(options, "model_expr", default="$AGENT_MODEL")
    fallback_model_expr = _shell_expr(
        options, "fallback_model_expr", default=model_expr
    )
    reasoning_effort = _str_option(options, "reasoning_effort", default="high")
    max_output_tokens = _int_option(options, "max_output_tokens", default=2400, min_value=200)
    max_steps = _int_option(options, "max_steps", default=10, min_value=1)
    command_timeout_sec = _int_option(
        options, "command_timeout_sec", default=240, min_value=5
    )

    return (
        "python3 harness/agents/openai_ops_agent.py "
        "--prompt-file {prompt_file} "
        "--strategy-file {strategy_file} "
        "--workspace {workspace} "
        f"--model {model_expr} "
        f"--fallback-model {fallback_model_expr} "
        f"--reasoning-effort {shlex.quote(reasoning_effort)} "
        f"--max-output-tokens {max_output_tokens} "
        f"--max-steps {max_steps} "
        f"--command-timeout-sec {command_timeout_sec} "
        f"--input-per-million {rates['input_per_million']} "
        f"--cached-input-per-million {rates['cached_input_per_million']} "
        f"--output-per-million {rates['output_per_million']}"
    )


def _build_openai_file_editor_backend(
    options: Mapping[str, Any], *, pricing: Mapping[str, float] | None
) -> str:
    rates = _pricing_defaults(pricing)
    model_expr = _shell_expr(options, "model_expr", default="$AGENT_MODEL")
    fallback_model_expr = _shell_expr(
        options, "fallback_model_expr", default=model_expr
    )
    reasoning_effort = _str_option(options, "reasoning_effort", default="high")
    max_output_tokens = _int_option(options, "max_output_tokens", default=3200, min_value=200)

    return (
        "python3 harness/agents/openai_file_editor.py "
        "--prompt-file {prompt_file} "
        "--strategy-file {strategy_file} "
        f"--model {model_expr} "
        f"--fallback-model {fallback_model_expr} "
        f"--reasoning-effort {shlex.quote(reasoning_effort)} "
        f"--max-output-tokens {max_output_tokens} "
        f"--input-per-million {rates['input_per_million']} "
        f"--cached-input-per-million {rates['cached_input_per_million']} "
        f"--output-per-million {rates['output_per_million']}"
    )


def _build_codex_cli_backend(
    options: Mapping[str, Any], *, pricing: Mapping[str, float] | None
) -> str:
    rates = _pricing_defaults(pricing)
    model_expr = _shell_expr(options, "model_expr", default="$AGENT_MODEL")
    fallback_model_expr = _shell_expr(
        options, "fallback_model_expr", default=model_expr
    )
    reasoning_effort = _str_option(options, "reasoning_effort", default="high")
    sandbox = _str_option(options, "sandbox", default="danger-full-access")
    codex_bin = _str_option(options, "codex_bin", default="codex")
    profile = _str_option(options, "profile", default="")
    skip_git_repo_check = _parse_bool(
        options.get("skip_git_repo_check", False),
        field="agent.backend_options.skip_git_repo_check",
    )
    dangerous_bypass = _parse_bool(
        options.get("dangerously_bypass_approvals_and_sandbox", True),
        field="agent.backend_options.dangerously_bypass_approvals_and_sandbox",
    )
    max_event_text_chars = _int_option(
        options, "max_event_text_chars", default=12000, min_value=400
    )

    parts = [
        "python3 harness/agents/codex_cli_agent.py",
        "--prompt-file {prompt_file}",
        "--workspace {workspace}",
        "--strategy-file {strategy_file}",
        f"--codex-bin {shlex.quote(codex_bin)}",
        f"--model {model_expr}",
        f"--fallback-model {fallback_model_expr}",
        f"--reasoning-effort {shlex.quote(reasoning_effort)}",
        f"--sandbox {shlex.quote(sandbox)}",
        f"--max-event-text-chars {max_event_text_chars}",
        f"--input-per-million {rates['input_per_million']}",
        f"--cached-input-per-million {rates['cached_input_per_million']}",
        f"--output-per-million {rates['output_per_million']}",
    ]

    if dangerous_bypass:
        parts.append("--dangerously-bypass-approvals-and-sandbox")
    if skip_git_repo_check:
        parts.append("--skip-git-repo-check")
    if profile:
        parts.append(f"--profile {shlex.quote(profile)}")

    # Keep raw Codex event traces per iteration for reliability debugging.
    parts.append("--raw-stdout-file {iter_dir}/codex_raw_stdout.log")
    parts.append("--raw-stderr-file {iter_dir}/codex_raw_stderr.log")

    return " ".join(parts)


def _str_option(options: Mapping[str, Any], key: str, *, default: str) -> str:
    value = options.get(key, default)
    if not isinstance(value, str):
        raise AgentBackendError(f"agent.backend_options.{key} must be a string")
    return value.strip()


def _shell_expr(options: Mapping[str, Any], key: str, *, default: str) -> str:
    value = _str_option(options, key, default=default)
    if not value:
        raise AgentBackendError(f"agent.backend_options.{key} must not be empty")
    return value


def _parse_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise AgentBackendError(f"{field} must be a boolean")


def _int_option(
    options: Mapping[str, Any],
    key: str,
    *,
    default: int,
    min_value: int,
) -> int:
    value = options.get(key, default)
    if isinstance(value, bool):
        raise AgentBackendError(f"agent.backend_options.{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AgentBackendError(
            f"agent.backend_options.{key} must be an integer"
        ) from exc
    if parsed < min_value:
        raise AgentBackendError(
            f"agent.backend_options.{key} must be >= {min_value}"
        )
    return parsed


def _float_option(
    value: Any,
    *,
    field: str,
    min_value: float,
) -> float:
    if isinstance(value, bool):
        raise AgentBackendError(f"{field} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AgentBackendError(f"{field} must be a number") from exc
    if parsed < min_value:
        raise AgentBackendError(f"{field} must be >= {min_value}")
    return parsed


def _pricing_defaults(pricing: Mapping[str, float] | None) -> dict[str, float]:
    defaults = {
        "input_per_million": 1.25,
        "cached_input_per_million": 0.125,
        "output_per_million": 10.0,
    }
    if pricing is None:
        return defaults
    out = defaults.copy()
    for key in out:
        if key in pricing:
            out[key] = _float_option(
                pricing[key], field=f"pricing.{key}", min_value=0.0
            )
    return out


_BACKEND_BUILDERS = {
    "openai_ops": _build_openai_ops_backend,
    "openai_file_editor": _build_openai_file_editor_backend,
    "codex_cli": _build_codex_cli_backend,
}
