#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${HARNESS_WORKDIR:-/workspace}"
CONFIG="${HARNESS_CONFIG:-harness/configs/prop_amm.cloud.toml}"

cd "$WORKDIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

python3 harness/core/loop.py --config "$CONFIG"
