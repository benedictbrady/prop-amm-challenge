#!/usr/bin/env python3
"""Prop AMM task adapter for the generic harness loop."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing TOML parser (`tomllib` or `tomli`)") from exc


@dataclass
class FoldSpec:
    name: str
    simulations: int
    steps: int
    seed_start: int
    seed_stride: int


@dataclass
class TaskConfig:
    holdout_target: float
    min_train_avg_for_holdout: float
    min_train_worst_for_holdout: float
    validate_before_eval: bool
    validate_command_template: str
    run_command_template: str
    validate_timeout_sec: int
    run_timeout_sec: int
    avg_edge_regex: re.Pattern[str]
    total_edge_regex: re.Pattern[str]
    train_folds: list[FoldSpec]
    holdout_fold: FoldSpec


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool


class AdapterError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_fold(raw: dict[str, Any], *, idx: int, section: str) -> FoldSpec:
    required = ["simulations", "steps", "seed_start", "seed_stride"]
    for key in required:
        if key not in raw:
            raise AdapterError(f"Missing {section}[{idx}].{key}")

    fold = FoldSpec(
        name=str(raw.get("name", f"{section}_{idx}")),
        simulations=int(raw["simulations"]),
        steps=int(raw["steps"]),
        seed_start=int(raw["seed_start"]),
        seed_stride=int(raw["seed_stride"]),
    )

    if fold.simulations <= 0:
        raise AdapterError(f"{section}[{idx}].simulations must be > 0")
    if fold.steps <= 0:
        raise AdapterError(f"{section}[{idx}].steps must be > 0")
    if fold.seed_stride <= 0:
        raise AdapterError(f"{section}[{idx}].seed_stride must be > 0")

    return fold


def load_config(path: Path) -> TaskConfig:
    raw = tomllib.loads(read_text(path))

    target = raw.get("target", {})
    eval_cfg = raw.get("evaluation", {})
    parsing = raw.get("parsing", {})

    train_raw = raw.get("train_folds")
    if not isinstance(train_raw, list) or not train_raw:
        raise AdapterError("train_folds must be a non-empty array")
    train_folds = [
        parse_fold(item, idx=i, section="train_folds") for i, item in enumerate(train_raw)
    ]

    holdout_raw = raw.get("holdout_fold")
    if not isinstance(holdout_raw, dict):
        raise AdapterError("holdout_fold table is required")
    holdout_fold = parse_fold(holdout_raw, idx=0, section="holdout_fold")

    avg_edge_pattern = str(
        parsing.get("avg_edge_regex", r"Avg edge:\\s*([-+]?\\d+(?:\\.\\d+)?)")
    )
    total_edge_pattern = str(
        parsing.get("total_edge_regex", r"Total edge:\\s*([-+]?\\d+(?:\\.\\d+)?)")
    )

    return TaskConfig(
        holdout_target=float(target.get("holdout_avg_edge", 525.0)),
        min_train_avg_for_holdout=float(target.get("min_train_avg_for_holdout", 505.0)),
        min_train_worst_for_holdout=float(target.get("min_train_worst_for_holdout", 470.0)),
        validate_before_eval=bool(eval_cfg.get("validate_before_eval", True)),
        validate_command_template=str(
            eval_cfg.get(
                "validate_command_template",
                "cargo run -q -p prop-amm -- validate {strategy_file}",
            )
        ),
        run_command_template=str(
            eval_cfg.get(
                "run_command_template",
                "cargo run -q -p prop-amm -- run {strategy_file} --simulations {simulations} --steps {steps} --seed-start {seed_start} --seed-stride {seed_stride}",
            )
        ),
        validate_timeout_sec=int(eval_cfg.get("validate_timeout_sec", 1800)),
        run_timeout_sec=int(eval_cfg.get("run_timeout_sec", 1800)),
        avg_edge_regex=re.compile(avg_edge_pattern),
        total_edge_regex=re.compile(total_edge_pattern),
        train_folds=train_folds,
        holdout_fold=holdout_fold,
    )


def run_command(command: str, *, cwd: Path, timeout_sec: int) -> CommandResult:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
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


def parse_metric(text: str, regex: re.Pattern[str], name: str) -> float:
    match = regex.search(text)
    if not match:
        raise AdapterError(f"Could not parse {name} from output")
    return float(match.group(1))


def command_vars(
    *,
    strategy_file: Path,
    fold: FoldSpec,
    mode: str,
    iteration: int,
) -> dict[str, Any]:
    return {
        "strategy_file": shlex.quote(str(strategy_file)),
        "simulations": fold.simulations,
        "steps": fold.steps,
        "seed_start": fold.seed_start,
        "seed_stride": fold.seed_stride,
        "fold_name": fold.name,
        "mode": mode,
        "iteration": iteration,
    }


def run_fold(
    cfg: TaskConfig,
    *,
    workspace: Path,
    strategy_file: Path,
    fold: FoldSpec,
    mode: str,
    iteration: int,
    log_dir: Path | None,
) -> dict[str, Any]:
    command = cfg.run_command_template.format(**command_vars(
        strategy_file=strategy_file,
        fold=fold,
        mode=mode,
        iteration=iteration,
    ))

    result = run_command(command, cwd=workspace, timeout_sec=cfg.run_timeout_sec)

    if log_dir is not None:
        write_text(log_dir / f"run_{fold.name}_stdout.log", result.stdout)
        write_text(log_dir / f"run_{fold.name}_stderr.log", result.stderr)

    output: dict[str, Any] = {
        "fold": fold.name,
        "command": command,
        "exit_code": result.exit_code,
        "duration_sec": result.duration_sec,
        "timed_out": result.timed_out,
    }

    if result.exit_code == 0:
        try:
            output["avg_edge"] = parse_metric(result.stdout, cfg.avg_edge_regex, "avg edge")
            output["total_edge"] = parse_metric(result.stdout, cfg.total_edge_regex, "total edge")
        except AdapterError as exc:
            output["parse_error"] = str(exc)
    else:
        output["error"] = "run command failed"

    return output


def context_payload(cfg: TaskConfig) -> dict[str, Any]:
    train_lines = [
        (
            f"- {fold.name}: sims={fold.simulations}, steps={fold.steps}, "
            f"seeds={fold.seed_start} + i*{fold.seed_stride}"
        )
        for fold in cfg.train_folds
    ]

    holdout = cfg.holdout_fold
    holdout_line = (
        f"- {holdout.name}: sims={holdout.simulations}, steps={holdout.steps}, "
        f"seeds={holdout.seed_start} + i*{holdout.seed_stride}"
    )

    target_label = (
        f"Holdout avg edge > {cfg.holdout_target:.2f} on "
        f"{holdout.simulations} out-of-sample simulations"
    )

    prompt_context = "\n".join(
        [
            "Objective and gates:",
            f"- target: {target_label}",
            (
                f"- promotion gate: train_avg >= {cfg.min_train_avg_for_holdout:.2f} "
                f"and train_worst >= {cfg.min_train_worst_for_holdout:.2f}"
            ),
            "",
            "Train folds:",
            *train_lines,
            "",
            "Holdout fold:",
            holdout_line,
            "",
            "Rules:",
            "- Do not run the expensive holdout manually; harness runs it only after promotion.",
            "- Optimize for robustness across folds, not a single seed range.",
        ]
    )

    return {
        "target_label": target_label,
        "prompt_context": prompt_context,
    }


def evaluate_payload(
    cfg: TaskConfig,
    *,
    workspace: Path,
    strategy_file: Path,
    mode: str,
    iteration: int,
    log_dir: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "timestamp": now_iso(),
        "validation_ok": True,
        "train_results": [],
        "train_avg": None,
        "train_worst": None,
        "promoted_to_holdout": False,
        "holdout_result": None,
        "holdout_avg": None,
        "passed_target": False,
        "target_label": context_payload(cfg)["target_label"],
        "thresholds": {
            "holdout_target": cfg.holdout_target,
            "min_train_avg_for_holdout": cfg.min_train_avg_for_holdout,
            "min_train_worst_for_holdout": cfg.min_train_worst_for_holdout,
        },
    }

    if cfg.validate_before_eval:
        validate_command = cfg.validate_command_template.format(
            strategy_file=shlex.quote(str(strategy_file)),
            mode=mode,
            iteration=iteration,
        )
        validate_result = run_command(
            validate_command,
            cwd=workspace,
            timeout_sec=cfg.validate_timeout_sec,
        )
        if log_dir is not None:
            write_text(log_dir / "validate_stdout.log", validate_result.stdout)
            write_text(log_dir / "validate_stderr.log", validate_result.stderr)

        payload["validation"] = {
            "command": validate_command,
            "exit_code": validate_result.exit_code,
            "duration_sec": validate_result.duration_sec,
            "timed_out": validate_result.timed_out,
        }

        if validate_result.exit_code != 0:
            payload["validation_ok"] = False
            payload["ok"] = False
            payload["error"] = "validation failed"
            return payload

    train_results: list[dict[str, Any]] = []
    for fold in cfg.train_folds:
        result = run_fold(
            cfg,
            workspace=workspace,
            strategy_file=strategy_file,
            fold=fold,
            mode=mode,
            iteration=iteration,
            log_dir=log_dir,
        )
        train_results.append(result)

    payload["train_results"] = train_results

    fold_avgs = [
        float(item["avg_edge"])
        for item in train_results
        if isinstance(item.get("avg_edge"), (int, float))
    ]

    if len(fold_avgs) != len(cfg.train_folds):
        payload["ok"] = False
        payload["error"] = "one or more train folds failed"
        return payload

    train_avg = statistics.mean(fold_avgs)
    train_worst = min(fold_avgs)
    payload["train_avg"] = train_avg
    payload["train_worst"] = train_worst

    promoted = (
        train_avg >= cfg.min_train_avg_for_holdout
        and train_worst >= cfg.min_train_worst_for_holdout
    )
    payload["promoted_to_holdout"] = promoted

    if not promoted:
        return payload

    holdout_result = run_fold(
        cfg,
        workspace=workspace,
        strategy_file=strategy_file,
        fold=cfg.holdout_fold,
        mode=mode,
        iteration=iteration,
        log_dir=log_dir,
    )
    payload["holdout_result"] = holdout_result

    holdout_avg = holdout_result.get("avg_edge")
    if isinstance(holdout_avg, (int, float)):
        payload["holdout_avg"] = float(holdout_avg)
        payload["passed_target"] = float(holdout_avg) > cfg.holdout_target
    else:
        payload["ok"] = False
        payload["error"] = "holdout failed or unparsable"

    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prop AMM task adapter")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--action", required=True, choices=["context", "evaluate"])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--strategy-file", required=True, type=Path)
    parser.add_argument("--mode", default="exploit")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--log-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    cfg = load_config(args.config.resolve())
    workspace = args.workspace.resolve()
    strategy_file = args.strategy_file.resolve()
    log_dir = args.log_dir.resolve() if args.log_dir is not None else None

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    if not workspace.exists():
        raise AdapterError(f"workspace does not exist: {workspace}")
    if not strategy_file.exists():
        raise AdapterError(f"strategy file does not exist: {strategy_file}")

    if args.action == "context":
        print(json.dumps(context_payload(cfg), separators=(",", ":")))
        return 0

    payload = evaluate_payload(
        cfg,
        workspace=workspace,
        strategy_file=strategy_file,
        mode=args.mode,
        iteration=args.iteration,
        log_dir=log_dir,
    )

    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except AdapterError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stdout)
        raise SystemExit(2)
