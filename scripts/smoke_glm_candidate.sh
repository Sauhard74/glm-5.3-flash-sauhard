#!/usr/bin/env bash
# Frees memory (stops the dashboard model), runs KL + speed on the GLM candidate, then restores Qwen.
set -uo pipefail
cd "$(dirname "$0")/.."
M="${1:-models/glm-candidate-auto76}"
TOK=$(cat control/control_token); API=http://127.0.0.1:8090/api
H=(-H "x-control-token: $TOK" -H "content-type: application/json")
curl -s "${H[@]}" -X POST $API/stop; echo; sleep 15
echo "== kl $(date)";    uv run python -m evals.kl --model "$M"
echo "== speed $(date)"; uv run python -m evals.speed --model "$M"
echo "== restore qwen $(date)"
curl -s "${H[@]}" -X POST $API/start -d '{"profile":"qwen"}'; echo
echo "SMOKE_DONE $(date)"
