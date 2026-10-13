#!/usr/bin/env bash
# Train one arm over the frozen split.
#   bash scripts/run_arm.sh onpolicy_distill [configs/default.yaml]
set -euo pipefail

ARM="${1:?usage: run_arm.sh <offpolicy_kd|onpolicy_distill|rlvr|continued_pretrain> [config]}"
CONFIG="${2:-configs/default.yaml}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

if [[ ! -f runs/audit/representability.json ]]; then
  echo "No frozen audit found. Run scripts/run_audit.sh first:" >&2
  echo "no arm may run before the split is frozen." >&2
  exit 1
fi

python -m src.cli train --config "$CONFIG" --arm "$ARM"
