#!/usr/bin/env bash
# Builds GLM pruning variants and A/Bs them with the role-aware KL eval (assistant vs prompt/tool positions).
set -uo pipefail
cd "$(dirname "$0")/.."
SRC=models/src-glm-5.3-flash CAL=artifacts/glm/calib LIMIT="${LIMIT:-12}"
build() {  # name, extra args...
  local name=$1; shift
  [[ -d models/$name ]] && { echo "== $name exists"; return; }
  echo "== build $name $(date)"
  uv run python -m build.make --model glm --src $SRC --calib $CAL --drop-vision --out models/$name.partial \
    --summary-json logs/$name-summary.json "$@" 2>&1 | grep -E "kept/layer|prune loss|total |Error|Traceback" \
    && mv models/$name.partial models/$name
}
build glm-mass153-76 --prune 153 --score mass --budget-gb 76
build glm-massauto-80 --prune auto --score mass --budget-gb 80
for m in glm-candidate-auto76 glm-mass153-76 glm-massauto-80; do
  echo "== kl $m $(date)"
  uv run python -m evals.kl --model models/$m --limit "$LIMIT" --name "$m-l$LIMIT" 2>&1 | grep -E "^\[kl\] .*(00|50) |Error|Traceback" | tail -3
  uv run python - "$m-l$LIMIT" <<'PY'
import json,sys,glob
f=sorted(glob.glob(f"results/**/*{sys.argv[1]}*kl*.json",recursive=True)+glob.glob(f"results/*{sys.argv[1]}*.json"))[-1]
s=json.load(open(f))["summary"]
print(sys.argv[1], {k:round(v,4) for k,v in s.items() if isinstance(v,float) and (k.endswith("_kl") or k.endswith("_top1") or k in("mean_kl","top1_agree"))})
PY
done
echo "AB_DONE $(date)"
