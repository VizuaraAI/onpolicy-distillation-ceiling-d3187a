#!/usr/bin/env bash
# Behavioral and representational readouts for one trained run.
#   bash scripts/run_eval.sh runs/onpolicy_distill [configs/default.yaml]
set -euo pipefail

RUN="${1:?usage: run_eval.sh <run_dir_or_model_path> [config]}"
CONFIG="${2:-configs/default.yaml}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

echo "Behavioral readout: pass@1 and pass@k..."
python -m src.cli eval --config "$CONFIG" --run "$RUN"

echo
echo "Representational readout: crosscoder diff with latent scaling..."
python -m src.cli diff --config "$CONFIG" --run "$RUN"

echo
echo "Copy the numbers into experiments/log_template.md before the paper."
