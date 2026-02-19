#!/usr/bin/env python3
"""Codex CLI wrapper agent for harness iterations.

This backend runs `codex exec` non-interactively, parses JSONL events for token
usage, and emits normalized accounting lines (`input_tokens`, `COST_USD`) that
the outer harness already understands.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex CLI wrapper backend")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--strategy-file", required=True, type=Path)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default="gpt-5-codex")
    parser.add_argument("--fallback-model", default="")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--sandbox",
        default="danger-full-access",
        choices=["read-only", "workspace-write", "danger-full-access"],
    )
    parser.add_argument("--dangerously-bypass-approvals-and-sandbox", action="store_true")
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument("--profile", default="")
    parser.add_argument("--raw-stdout-file", type=Path)
    parser.add_argument("--raw-stderr-file", type=Path)
    parser.add_argument("--max-event-text-chars", type=int, default=12000)
    parser.add_argument("--input-per-million", type=float)
    parser.add_argument("--cached-input-per-million", type=float)
    parser.add_argument("--output-per-million", type=float)
    return parser.parse_args(argv)


def unique_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def parse_jsonl_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        payload = line.strip()
        if not payload.startswith("{"):
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def extract_usage_totals(events: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for event in events:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                totals[key] += value
    return totals


def extract_agent_message(events: list[dict[str, Any]], max_chars: int) -> str:
    latest = ""
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            latest = text.strip()
    if len(latest) > max_chars:
        return latest[: max_chars - 20] + "\n...[truncated]"
    return latest


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


def build_command(args: argparse.Namespace, model: str) -> list[str]:
    cmd: list[str] = [
        args.codex_bin,
        "exec",
        "--json",
        "-C",
        str(args.workspace),
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{args.reasoning_effort}"',
    ]
    if args.profile.strip():
        cmd.extend(["--profile", args.profile.strip()])
    if args.skip_git_repo_check:
        cmd.append("--skip-git-repo-check")
    if args.dangerously_bypass_approvals_and_sandbox:
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        cmd.extend(["--sandbox", args.sandbox])
    cmd.append("-")
    return cmd


def run_model(args: argparse.Namespace, *, model: str, prompt_text: str) -> subprocess.CompletedProcess[str]:
    cmd = build_command(args, model)
    return subprocess.run(
        cmd,
        input=prompt_text,
        capture_output=True,
        text=True,
        check=False,
    )


def print_usage(usage: dict[str, int]) -> None:
    if any(usage.values()):
        print(json.dumps(usage, separators=(",", ":")))


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.workspace = args.workspace.resolve()
    args.strategy_file = args.strategy_file.resolve()
    args.prompt_file = args.prompt_file.resolve()

    if not args.workspace.exists():
        raise SystemExit(f"workspace does not exist: {args.workspace}")
    if not args.strategy_file.exists():
        raise SystemExit(f"strategy file does not exist: {args.strategy_file}")
    if not args.prompt_file.exists():
        raise SystemExit(f"prompt file does not exist: {args.prompt_file}")
    if args.max_event_text_chars < 200:
        raise SystemExit("--max-event-text-chars must be >= 200")

    prompt_text = read_text(args.prompt_file)
    models = unique_items([args.model, args.fallback_model])
    if not models:
        raise SystemExit("At least one model must be provided")

    last_proc: subprocess.CompletedProcess[str] | None = None
    last_model = ""
    for model in models:
        try:
            proc = run_model(args, model=model, prompt_text=prompt_text)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"codex binary not found: {args.codex_bin}. Install with `npm install -g @openai/codex`."
            ) from exc
        last_proc = proc
        last_model = model
        if proc.returncode == 0:
            events = parse_jsonl_events(proc.stdout)
            message = extract_agent_message(events, args.max_event_text_chars)
            if message:
                print(message)
            else:
                print("changes: codex run completed")
            usage = extract_usage_totals(events)
            print_usage(usage)
            cost = maybe_cost(args, usage)
            if cost is not None:
                print(f"COST_USD={cost:.6f}")
            if args.raw_stdout_file is not None:
                write_text(args.raw_stdout_file, proc.stdout)
            if args.raw_stderr_file is not None:
                write_text(args.raw_stderr_file, proc.stderr)
            return 0

    if last_proc is None:
        raise SystemExit("codex_cli_agent failed to launch any model")

    if args.raw_stdout_file is not None:
        write_text(args.raw_stdout_file, last_proc.stdout)
    if args.raw_stderr_file is not None:
        write_text(args.raw_stderr_file, last_proc.stderr)

    sys.stderr.write(
        f"codex exec failed for models={models} (last_model={last_model}, exit={last_proc.returncode})\n"
    )
    combined = (last_proc.stdout + "\n" + last_proc.stderr).strip()
    if combined:
        tail = combined[-args.max_event_text_chars :]
        sys.stderr.write(tail + "\n")
    return last_proc.returncode if last_proc.returncode != 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
