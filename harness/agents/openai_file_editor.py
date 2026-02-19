#!/usr/bin/env python3
"""Simple OpenAI-backed file editor backend for the generic harness."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

MARKER_OPEN = "<UPDATED_SOURCE>"
MARKER_CLOSE = "</UPDATED_SOURCE>"

SYSTEM_PROMPT = """You are a precise coding assistant.
You will receive a task prompt and the current source for one file.
Return the FULL updated file content between exact markers:
<UPDATED_SOURCE>
...full file...
</UPDATED_SOURCE>
Then include concise notes outside the markers.
Do not omit required symbols or metadata.
"""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def extract_updated_source(text: str) -> str:
    if MARKER_OPEN in text and MARKER_CLOSE in text:
        start = text.index(MARKER_OPEN) + len(MARKER_OPEN)
        end = text.index(MARKER_CLOSE, start)
        return text[start:end].strip("\n")

    codeblock = re.search(r"```(?:rust)?\n([\s\S]*?)\n```", text)
    if codeblock:
        return codeblock.group(1).strip("\n")

    raise ValueError("No <UPDATED_SOURCE> block or fenced code block found")


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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-backed one-file editor")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--strategy-file", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fallback-model", default="gpt-5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-output-tokens", type=int, default=3200)
    parser.add_argument("--input-per-million", type=float)
    parser.add_argument("--cached-input-per-million", type=float)
    parser.add_argument("--output-per-million", type=float)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc

    prompt_text = read_text(args.prompt_file)
    current_source = read_text(args.strategy_file)

    user_prompt = (
        "Task prompt:\n"
        f"{prompt_text}\n\n"
        "Current file path:\n"
        f"{args.strategy_file}\n\n"
        "Current file source:\n"
        "```rust\n"
        f"{current_source}\n"
        "```\n"
    )

    client = OpenAI()
    models = [args.model]
    if args.fallback_model and args.fallback_model not in models:
        models.append(args.fallback_model)

    resp: Any | None = None
    last_error = ""
    for model in models:
        try:
            resp = client.responses.create(
                model=model,
                reasoning={"effort": args.reasoning_effort},
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
            break
        except Exception as exc:  # pragma: no cover - network/runtime errors
            last_error = f"{type(exc).__name__}: {exc}"
            continue

    if resp is None:
        raise SystemExit(f"Model request failed for all models: {last_error}")

    output_text = extract_output_text(resp)
    if not output_text.strip():
        print("changes: no-op (model returned empty response)")
        usage = token_usage(resp)
        if usage:
            print(json.dumps(usage, separators=(",", ":")))
        cost = maybe_cost(args, usage)
        if cost is not None:
            print(f"COST_USD={cost:.6f}")
        return 0

    try:
        updated_source = extract_updated_source(output_text)
    except ValueError as exc:
        raise SystemExit(f"Could not extract updated source from response: {exc}") from exc
    if not updated_source.strip():
        raise SystemExit("Extracted updated source was empty")

    write_text(args.strategy_file, updated_source + "\n")

    print("changes: updated strategy file from model output")

    usage = token_usage(resp)
    if usage:
        print(json.dumps(usage, separators=(",", ":")))

    cost = maybe_cost(args, usage)
    if cost is not None:
        print(f"COST_USD={cost:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
