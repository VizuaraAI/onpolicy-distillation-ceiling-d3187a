#!/usr/bin/env bash
# The whole experiment, in the only order that keeps it falsifiable.
#
# Arm order is deliberate: on-policy distillation first so you know the
# implementation works on a REPRESENTED capability, then continued
# pretraining because it is the positive control and the experiment is
# uninterpretable without it, then the remaining two arms.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

bash scripts/run_audit.sh "$CONFIG"

for ARM in onpolicy_distill continued_pretrain offpolicy_kd rlvr; do
  echo
  echo "=== arm: $ARM ==="
  bash scripts/run_arm.sh "$ARM" "$CONFIG"
  bash scripts/run_eval.sh "runs/$ARM" "$CONFIG"
done

echo
echo "All arms done. Fill in experiments/log_template.md, then the paper."
