#!/usr/bin/env bash
# A's exact config with LSQ rounding, then the same role-aware KL eval as A.
set -uo pipefail
cd "$(dirname "$0")/.."
N=${N:-glm-reap153-76-lsq}
echo "== build $N $(date)"
uv run python -m build.make --model glm --src models/src-glm-5.3-flash --calib artifacts/glm/calib ${PRUNE_ARGS:---prune 153 --score reap} \
  --budget-gb ${BUDGET:-76} --drop-vision --rounding lsq --out models/$N.partial --summary-json logs/$N-summary.json > logs/build-$N.log 2>&1 \
  && mv models/$N.partial models/$N || { echo "BUILD_FAILED"; tail -5 logs/build-$N.log; exit 1; }
tail -6 logs/build-$N.log
echo "== built $(date)"
scripts/glm_ab_eval.sh $N
