#!/usr/bin/env bash
# Waits for the GLM calibration to finish, then builds models/glm-5.3-flash-custom.
# Refuses to build if any layer lacks real calibration stats (build would silently fall back to weight saliency).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/calib-glm.log CAL=artifacts/glm/calib OUT=models/glm-candidate-auto76
BUDGET="${BUDGET_GB:-76}"

until grep -q CALIB_DONE "$LOG"; do
  if ! pgrep -f "calib.run --model glm" >/dev/null && ! grep -q CALIB_DONE "$LOG"; then
    echo "calibration process exited without CALIB_DONE; not building"; exit 1
  fi
  sleep 60
done

N=$(uv run python -c "import json; print(json.load(open('$CAL/meta.json'))['num_layers'])")
HAVE=$(ls "$CAL"/layer_*.json | wc -l | tr -d ' ')
if (( HAVE < N )); then echo "only $HAVE/$N layer stats present; not building"; exit 1; fi

echo "== build $(date)"
rm -rf "$OUT.partial"
nice -n 5 uv run python -m build.make --model glm --src models/src-glm-5.3-flash --calib "$CAL" \
  --prune auto --budget-gb "$BUDGET" --drop-vision --out "$OUT.partial" --summary-json logs/glm-build-summary.json \
  2>&1 | tee logs/glm-build.log
if grep -q "using weight saliency" logs/glm-build.log; then
  echo "build used weight-saliency fallback for some layer; leaving $OUT.partial unpromoted"; exit 1
fi
rm -rf "$OUT" && mv "$OUT.partial" "$OUT"
echo "BUILD_DONE $(date) -> $OUT"
