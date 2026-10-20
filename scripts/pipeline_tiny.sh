#!/usr/bin/env bash
# End-to-end pipeline on tiny real-architecture checkpoints: make_tiny → calib_data → calib → build → evals → serve dry-run.
# Usage: scripts/pipeline_tiny.sh [glm|qwen|both]   (outputs under artifacts/tiny/)
set -euo pipefail
cd "$(dirname "$0")/.."
ARCHES=("${1:-both}"); [[ "${ARCHES[0]}" == both ]] && ARCHES=(glm qwen)

for A in "${ARCHES[@]}"; do
  W=artifacts/tiny/$A
  rm -rf "$W" "models/tiny-$A-pruned"
  mkdir -p "$W"
  TOK=tests/fixtures/tokenizers/$A
  echo "== [$A] tiny source checkpoint"
  VOCAB=$(uv run python -c "from tokenizers import Tokenizer; n=Tokenizer.from_file('$TOK/tokenizer.json').get_vocab_size(); print(-(-n//128)*128)")
  uv run python tests/fixtures/make_tiny.py --arch "$A" --out "$W/src" --tokenizer-dir "$TOK" --vocab-size "$VOCAB"
  echo "== [$A] calibration data (smoke)"
  uv run python -m calib_data.build --model "$A" --out "$W/data" --smoke --tokenizer "$TOK"
  echo "== [$A] layer-wise calibration"
  uv run python -m calib.run --model "$A" --src "$W/src" --data "$W/data" --out "$W/calib" --mem-gb 8
  echo "== [$A] prune + mixed-precision build"
  uv run python -m build.make --model "$A" --src "$W/src" --calib "$W/calib" --prune 0.5 --drop-vision \
    --budget-gb "${BUDGET_GB:-0.09}" --out "models/tiny-$A-pruned"
  echo "== [$A] evals"
  uv run python -m evals.kl --model "models/tiny-$A-pruned" --ref "$W/calib/ref" --smoke --results-dir "$W/results"
  uv run python -m evals.speed --model "models/tiny-$A-pruned" --smoke --results-dir "$W/results"
  echo "== [$A] serve fit check"
  uv run python -m serve --model "models/tiny-$A-pruned" --dry-run
done
echo "PIPELINE OK: ${ARCHES[*]}"
