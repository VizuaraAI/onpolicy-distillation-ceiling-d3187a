#!/usr/bin/env bash
# Score representability on the base student and freeze the split.
# Run this ONCE, before any training. Commit the resulting JSON.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

echo "Smoke testing the battery first..."
python -m src.cli battery --config "$CONFIG"

echo
echo "Running the representability audit (this needs a GPU)..."
python -m src.cli audit --config "$CONFIG"

echo
echo "Split frozen. Commit runs/audit/representability.json now."
