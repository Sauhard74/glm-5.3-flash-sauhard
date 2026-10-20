#!/usr/bin/env bash
# Role-aware KL eval for GLM variants; one full log per model in logs/kl-<model>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
LIMIT="${LIMIT:-12}"
for m in "$@"; do
  [[ -f results/$m-l$LIMIT/kl.json ]] && { echo "== $m done already"; continue; }
  echo "== kl $m $(date)"
  uv run python -m evals.kl --model models/$m --limit "$LIMIT" --name "$m-l$LIMIT" > logs/kl-$m.log 2>&1
  echo "== exit $? $m $(date)"
done
echo "AB_DONE $(date)"
