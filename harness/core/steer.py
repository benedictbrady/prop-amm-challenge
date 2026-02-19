#!/usr/bin/env python3
"""Operator steering utility for the generic harness loop."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing TOML parser (`tomllib` or `tomli`)") from exc


@dataclass(frozen=True)
class SteerConfig:
    workspace: Path
    state_dir: Path
    steer_file: Path | None
    default_note_iterations: int


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        return value
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject operator steers for harness/core/loop.py")
    parser.add_argument("--config", required=True, type=Path, help="Harness TOML config path")

    sub = parser.add_subparsers(dest="cmd", required=True)

    fs = sub.add_parser("fresh-start", help="Throw out current working trajectory and restart")
    fs.add_argument("--note", default="", help="Optional high-priority note for the next iterations")
    fs.add_argument(
        "--note-iterations",
        type=int,
        default=None,
        help="How many iterations to keep note active",
    )
    fs.add_argument(
        "--reset-budget-spent",
        action="store_true",
        help="Reset state.budget_spent_usd to 0 for the new trajectory",
    )
    fs.add_argument(
        "--keep-iterations",
        action="store_true",
        help="Do not clear historical iterations",
    )
    fs.add_argument(
        "--keep-elites",
        action="store_true",
        help="Do not clear elite pool",
    )
    fs.add_argument(
        "--no-reset-strategy",
        action="store_true",
        help="Do not reset strategy file to baseline/git",
    )
    fs.add_argument(
        "--persistent",
        action="store_true",
        help="Keep steer file after one application",
    )

    note = sub.add_parser("note", help="Inject a high-priority operator note without full restart")
    note.add_argument("--text", required=True, help="Note text to inject into prompt context")
    note.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="How many iterations to keep this note active",
    )
    note.add_argument(
        "--persistent",
        action="store_true",
        help="Keep steer file after one application",
    )

    clear = sub.add_parser("clear-note", help="Clear current operator note")
    clear.add_argument(
        "--persistent",
        action="store_true",
        help="Keep steer file after one application",
    )

    sub.add_parser("status", help="Show pending steer payload and active note status")
    return parser.parse_args(argv)


def load_steer_config(config_path: Path) -> SteerConfig:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths = raw.get("paths", {})
    control = raw.get("control", {})

    workspace = Path(paths.get("workspace", ".")).expanduser().resolve()
    state_dir = (workspace / str(paths.get("state_dir", ".harness"))).resolve()
    steer_raw = control.get("steer_file", ".harness/steer.json")
    steer_file: Path | None
    if steer_raw in (None, "", False):
        steer_file = None
    else:
        steer_file = (workspace / str(steer_raw)).resolve()

    default_note_iterations = int(control.get("default_note_iterations", 5))
    if default_note_iterations < 1:
        default_note_iterations = 5

    return SteerConfig(
        workspace=workspace,
        state_dir=state_dir,
        steer_file=steer_file,
        default_note_iterations=default_note_iterations,
    )


def require_steer_file(config_path: Path) -> tuple[Path, SteerConfig]:
    cfg = load_steer_config(config_path.resolve())
    if cfg.steer_file is None:
        raise SystemExit(
            "Steering is disabled: set [control].steer_file in config to enable operator steers."
        )
    return cfg.steer_file, cfg


def cmd_status(config_path: Path) -> int:
    steer_file, cfg = require_steer_file(config_path)
    state_path = cfg.state_dir / "state.json"
    state = read_json(state_path) or {}
    pending = read_json(steer_file)

    print(f"steer_file: {steer_file}")
    print(f"pending_steer: {'yes' if pending else 'no'}")
    if pending:
        print(json.dumps(pending, indent=2, sort_keys=True))

    note = state.get("operator_note")
    remaining = state.get("operator_note_remaining")
    if isinstance(note, str) and note.strip() and isinstance(remaining, int) and remaining > 0:
        print(f"active_note_remaining: {remaining}")
        print(f"active_note: {note.strip()}")
    else:
        print("active_note: none")
    return 0


def cmd_fresh_start(args: argparse.Namespace, config_path: Path) -> int:
    steer_file, cfg = require_steer_file(config_path)
    payload: dict[str, Any] = {
        "action": "fresh_start",
        "apply_once": not args.persistent,
        "keep_budget_spent": not args.reset_budget_spent,
        "clear_iterations": not args.keep_iterations,
        "clear_elites": not args.keep_elites,
        "reset_strategy": not args.no_reset_strategy,
        "clear_sysadmin_checks": True,
        "reset_best_metrics": True,
        "clear_stopped_reason": True,
    }
    if args.note.strip():
        payload["note"] = args.note.strip()
    if args.note.strip():
        payload["note_iterations"] = (
            max(1, args.note_iterations)
            if args.note_iterations is not None
            else cfg.default_note_iterations
        )

    write_text(steer_file, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote steer: {steer_file}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_note(args: argparse.Namespace, config_path: Path) -> int:
    steer_file, cfg = require_steer_file(config_path)
    payload: dict[str, Any] = {
        "action": "note",
        "apply_once": not args.persistent,
        "note": args.text.strip(),
    }
    payload["note_iterations"] = (
        max(1, args.iterations)
        if args.iterations is not None
        else cfg.default_note_iterations
    )

    write_text(steer_file, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote steer: {steer_file}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_clear_note(args: argparse.Namespace, config_path: Path) -> int:
    steer_file, _cfg = require_steer_file(config_path)
    payload: dict[str, Any] = {
        "action": "clear_note",
        "apply_once": not args.persistent,
    }
    write_text(steer_file, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote steer: {steer_file}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    if args.cmd == "status":
        return cmd_status(config_path)
    if args.cmd == "fresh-start":
        return cmd_fresh_start(args, config_path)
    if args.cmd == "note":
        return cmd_note(args, config_path)
    if args.cmd == "clear-note":
        return cmd_clear_note(args, config_path)
    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
