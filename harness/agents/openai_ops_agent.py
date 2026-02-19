#!/usr/bin/env python3
"""OpenAI-backed general-purpose ops agent for harness iterations.

Unlike the one-file editor backend, this agent can run arbitrary shell commands
inside the workspace and make broad codebase changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are an autonomous coding and systems agent.
You are running inside a repository and can execute shell commands.
Your job is to improve task score robustly, fix failures, and keep progress moving.

Important constraints:
- Prefer small, testable changes.
- Always investigate current failures before changing strategy.
- Avoid repeating commands that already failed without changing approach.
- End with `finish` when you have completed a meaningful attempt for this iteration.

Return ONLY JSON with one of:
{"action":"run","command":"<shell command>","why":"short reason"}
{"action":"finish","summary":"short summary"}
"""


@dataclass
class CommandResult:
    command: str
    exit_code: int
    timed_out: bool
    duration_sec: float
    stdout: str
    stderr: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-backed operations agent")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--strategy-file", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fallback-model", default="gpt-5")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=2400)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--command-timeout-sec", type=int, default=240)
    parser.add_argument("--max-command-output-chars", type=int, default=12000)
    parser.add_argument("--input-per-million", type=float)
    parser.add_argument("--cached-input-per-million", type=float)
    parser.add_argument("--output-per-million", type=float)
    return parser.parse_args(argv)


def extract_output_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    parts: list[str] = []
    output = getattr(resp, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    ctext = getattr(c, "text", None)
                    if isinstance(ctext, str):
                        parts.append(ctext)
                    elif isinstance(c, dict) and isinstance(c.get("text"), str):
                        parts.append(c["text"])
    if parts:
        return "\n".join(parts)

    dump_fn = getattr(resp, "model_dump", None)
    if callable(dump_fn):
        dumped = dump_fn()
        if isinstance(dumped, dict):
            queue: list[Any] = [dumped]
            while queue:
                item = queue.pop(0)
                if isinstance(item, dict):
                    text_val = item.get("text")
                    if isinstance(text_val, str) and text_val.strip():
                        parts.append(text_val)
                    value_val = item.get("value")
                    if isinstance(value_val, str) and value_val.strip():
                        parts.append(value_val)
                    queue.extend(item.values())
                elif isinstance(item, list):
                    queue.extend(item)
    return "\n".join(parts)


def parse_json_obj(text: str) -> dict[str, Any] | None:
    payload = text.strip()
    if not payload:
        return None
    try:
        value = json.loads(payload)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", payload)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def token_usage(resp: Any) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}

    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    input_tokens = _get(usage, "input_tokens")
    output_tokens = _get(usage, "output_tokens")

    cached_input_tokens = 0
    details = _get(usage, "input_tokens_details")
    if details is not None:
        cached_input_tokens = _get(details, "cached_tokens") or 0

    out: dict[str, int] = {}
    if isinstance(input_tokens, int):
        out["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        out["output_tokens"] = output_tokens
    if isinstance(cached_input_tokens, int):
        out["cached_input_tokens"] = cached_input_tokens
    return out


def add_usage(total: dict[str, int], delta: dict[str, int]) -> None:
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        total[key] = total.get(key, 0) + delta.get(key, 0)


def maybe_cost(args: argparse.Namespace, usage: dict[str, int]) -> float | None:
    if (
        args.input_per_million is None
        or args.cached_input_per_million is None
        or args.output_per_million is None
    ):
        return None
    input_tokens = usage.get("input_tokens", 0)
    cached_input_tokens = usage.get("cached_input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return (
        (input_tokens / 1_000_000.0) * args.input_per_million
        + (cached_input_tokens / 1_000_000.0) * args.cached_input_per_million
        + (output_tokens / 1_000_000.0) * args.output_per_million
    )


def run_shell(command: str, *, cwd: Path, timeout_sec: int) -> CommandResult:
    import time

    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return CommandResult(
            command=command,
            exit_code=proc.returncode,
            timed_out=False,
            duration_sec=time.monotonic() - started,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return CommandResult(
            command=command,
            exit_code=124,
            timed_out=True,
            duration_sec=time.monotonic() - started,
            stdout=stdout,
            stderr=stderr,
        )


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - 30)
    return text[:keep] + "\n...[truncated]..."


def unique_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_user_prompt(
    *,
    task_prompt: str,
    strategy_file: Path,
    workspace: Path,
    history: list[dict[str, Any]],
    step: int,
    max_steps: int,
) -> str:
    history_lines: list[str] = []
    for idx, item in enumerate(history[-6:], start=max(1, len(history) - 5)):
        history_lines.append(
            f"Step {idx} command: {item.get('command')}\n"
            f"exit={item.get('exit_code')} timeout={item.get('timed_out')} dur={item.get('duration_sec')}\n"
            f"stdout:\n{item.get('stdout')}\n"
            f"stderr:\n{item.get('stderr')}\n"
        )
    history_block = "\n".join(history_lines) if history_lines else "(no commands yet)"

    return (
        f"Iteration task prompt:\n{task_prompt}\n\n"
        f"Workspace: {workspace}\n"
        f"Primary strategy file: {strategy_file}\n"
        f"Current step: {step}/{max_steps}\n\n"
        f"Recent command history:\n{history_block}\n\n"
        "Decide the single best next action."
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc

    task_prompt = read_text(args.prompt_file)
    workspace = args.workspace.resolve()
    strategy_file = args.strategy_file.resolve()
    if not workspace.exists():
        raise SystemExit(f"workspace does not exist: {workspace}")
    if not strategy_file.exists():
        raise SystemExit(f"strategy file does not exist: {strategy_file}")

    before_hash = sha256_file(strategy_file)

    client = OpenAI()
    models = unique_items([args.model, args.fallback_model])
    efforts = unique_items([args.reasoning_effort, "medium", "low"])

    history: list[dict[str, Any]] = []
    usage_totals: dict[str, int] = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    final_summary = "iteration attempt completed"

    for step in range(1, args.max_steps + 1):
        user_prompt = build_user_prompt(
            task_prompt=task_prompt,
            strategy_file=strategy_file,
            workspace=workspace,
            history=history,
            step=step,
            max_steps=args.max_steps,
        )

        parsed: dict[str, Any] | None = None
        last_error = ""
        for model in models:
            for effort in efforts:
                try:
                    resp = client.responses.create(
                        model=model,
                        reasoning={"effort": effort},
                        max_output_tokens=args.max_output_tokens,
                        input=[
                            {
                                "role": "system",
                                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                            },
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": user_prompt}],
                            },
                        ],
                    )
                except Exception as exc:  # pragma: no cover - runtime/network
                    last_error = f"{type(exc).__name__}: {exc}"
                    continue

                add_usage(usage_totals, token_usage(resp))
                out_text = extract_output_text(resp)
                parsed = parse_json_obj(out_text)
                if parsed is not None:
                    break
                last_error = f"model {model} effort={effort} returned non-JSON output"
            if parsed is not None:
                break

        if parsed is None:
            raise SystemExit(f"agent planning failed: {last_error}")

        action = str(parsed.get("action", "")).strip().lower()
        if action == "finish":
            final_summary = str(parsed.get("summary", "finished")).strip() or "finished"
            break

        if action != "run":
            history.append(
                {
                    "command": "<invalid-action>",
                    "exit_code": 1,
                    "timed_out": False,
                    "duration_sec": 0.0,
                    "stdout": "",
                    "stderr": f"invalid action: {action}",
                }
            )
            continue

        command = str(parsed.get("command", "")).strip()
        if not command:
            history.append(
                {
                    "command": "<empty>",
                    "exit_code": 1,
                    "timed_out": False,
                    "duration_sec": 0.0,
                    "stdout": "",
                    "stderr": "empty command",
                }
            )
            continue

        result = run_shell(command, cwd=workspace, timeout_sec=args.command_timeout_sec)
        history.append(
            {
                "command": command,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_sec": round(result.duration_sec, 3),
                "stdout": truncate(result.stdout, args.max_command_output_chars),
                "stderr": truncate(result.stderr, args.max_command_output_chars),
            }
        )

    after_hash = sha256_file(strategy_file)
    changed = before_hash != after_hash
    if changed:
        print("changes: updated strategy and/or workspace artifacts")
    else:
        print("changes: no-op (no strategy-file hash delta)")

    print(f"summary: {final_summary}")
    if history:
        last = history[-1]
        print(
            "last_command: exit={exit_code} timeout={timed_out} cmd={cmd}".format(
                exit_code=last.get("exit_code"),
                timed_out=last.get("timed_out"),
                cmd=last.get("command", ""),
            )
        )

    print(json.dumps(usage_totals, separators=(",", ":")))
    cost = maybe_cost(args, usage_totals)
    if cost is not None:
        print(f"COST_USD={cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
