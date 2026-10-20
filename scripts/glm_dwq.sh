#!/usr/bin/env bash
# Blockwise DWQ on a built GLM student, then the role-aware KL eval.
set -uo pipefail
cd "$(dirname "$0")/.."
B=${BASE:-glm-reap153-76-lsq} N=${N:-$B-dwq}
echo "== dwq $B -> $N $(date)"
uv run python -m recover.blockdwq --model glm --src models/src-glm-5.3-flash --student models/$B --data artifacts/glm/data \
  --out models/$N ${DWQ_ARGS:-} > logs/dwq-$N.log 2>&1 || { echo "DWQ_FAILED $?"; tail -5 logs/dwq-$N.log; exit 1; }
echo "== dwq done $(date)"
scripts/glm_ab_eval.sh $N
