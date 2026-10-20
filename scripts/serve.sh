#!/usr/bin/env bash
# Launch the memory-capped local server for one model.
#
#   scripts/serve.sh <model> [python -m serve options]
#
# <model> is a directory (models/<name> is tried when a bare name is given) or an
# HF repo id. Options can also come from SERVE_* environment variables, e.g.
# SERVE_MEMORY_LIMIT_GB=80 SERVE_PORT=8642. `--help` lists everything.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

if [[ $# -lt 1 || "$1" == -* ]]; then
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec uv run --frozen python -m serve --help
  fi
  echo "usage: scripts/serve.sh <model> [options]   (scripts/serve.sh --help for options)" >&2
  exit 64
fi

model="$1"
shift
if [[ ! -e "$model" && -d "models/$model" ]]; then
  model="models/$model"
fi
if [[ -d "$model" ]]; then
  model="$(cd "$model" && pwd)"
fi

exec uv run --frozen python -m serve --model "$model" "$@"
