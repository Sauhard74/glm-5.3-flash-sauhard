#!/usr/bin/env bash
# Task evals (code, tools, math) on GLM candidates, one process at a time. Usage: scripts/glm_task_evals.sh <models...>
set -uo pipefail
cd "$(dirname "$0")/.."
for m in "$@"; do
  for s in code tools math; do
    case $s in
      code) a="--limit ${CODE_LIMIT:-40}" ;;
      tools) a="--limit ${TOOLS_LIMIT:-15}" ;;
      math) a="--datasets math500 --math500-n ${MATH_N:-40}" ;;
    esac
    echo "== $s $m $(date)"
    uv run python -m evals.$s --model models/$m --resume $a > logs/task-$s-$m.log 2>&1 || echo "== FAILED $s $m $?"
  done
done
uv run python -m evals.compare --results-dir results
echo "TASKS_DONE $(date)"
