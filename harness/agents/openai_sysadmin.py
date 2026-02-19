#!/usr/bin/env python3
"""OpenAI-backed sysadmin checker for the harness."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are an expert reliability sysadmin for an autonomous coding harness.
Diagnose failures, choose one safe remediation action, and keep the system running.
Return JSON only.
"""

ALLOWED_ACTIONS = {"noop", "sleep_60", "sleep_300", "restart_from_baseline"}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
        if isinstance(value, dict):
            return value
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI sysadmin checker")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fallback-model", default="gpt-5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc

    prompt = read_text(args.prompt_file)
    client = OpenAI()
    models = [args.model]
    if args.fallback_model and args.fallback_model not in models:
        models.append(args.fallback_model)

    data: dict[str, Any] | None = None
    last_error = ""
    for model in models:
        try:
            resp = client.responses.create(
                model=model,
                reasoning={"effort": args.reasoning_effort},
                max_output_tokens=args.max_output_tokens,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
                ],
            )
            text = extract_output_text(resp)
            data = parse_json_obj(text)
            if data is not None:
                break
            last_error = f"model {model} returned non-JSON output"
        except Exception as exc:  # pragma: no cover - network/runtime errors
            last_error = f"{type(exc).__name__}: {exc}"
            continue

    if data is None:
        print(
            json.dumps(
                {
                    "decision": "continue",
                    "health": "broken",
                    "root_cause": f"sysadmin_model_failure: {last_error}",
                    "action": "sleep_60",
                    "notes": "fallback decision due to sysadmin model failure",
                },
                separators=(",", ":"),
            )
        )
        return 0

    action = str(data.get("action", "noop")).strip().lower()
    if action not in ALLOWED_ACTIONS:
        action = "noop"

    out = {
        "decision": "continue",
        "health": str(data.get("health", "degraded")).strip().lower() or "degraded",
        "root_cause": str(data.get("root_cause", "n/a")).strip(),
        "action": action,
        "notes": str(data.get("notes", "")).strip(),
    }
    print(json.dumps(out, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
