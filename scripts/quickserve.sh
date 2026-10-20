#!/usr/bin/env bash
# Usage: scripts/quickserve.sh [model_dir] — OpenAI (/v1/chat/completions) + Anthropic (/v1/messages) on :8080
# Env: MTP=1|0, DRAFTER, DRAFT_BLOCK, CAP_GB, WIRE_GB, CACHE_GB, PREFILL, PORT, APC_MEMORY_MAX_GB, APC_DISK_MAX_GB
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-models/qwen-flash-next-reap288}"

export CAP_GB="${CAP_GB:-80}" WIRE_GB="${WIRE_GB:-0}" CACHE_GB="${CACHE_GB:-2}"
export APC_ENABLED=1 APC_MEMORY_MAX_GB="${APC_MEMORY_MAX_GB:-6}" APC_DISK_MAX_GB="${APC_DISK_MAX_GB:-40}"
export APC_CHECKPOINT_ENTRIES=2 HF_HUB_OFFLINE=1
[[ -z "${MLX_VLM_SERVER_API_KEY:-}" && -f control/api_key ]] && export MLX_VLM_SERVER_API_KEY="$(cat control/api_key)"

DRAFT_ARGS=()
if [[ "${MTP:-1}" == "1" ]]; then
  DRAFT_ARGS=(--draft-model "${DRAFTER:-models/qwen-flash-next-mtp-drafter}" --draft-kind mtp --draft-block-size "${DRAFT_BLOCK:-2}")
fi

exec uv run python scripts/run_server.py \
  --model "$MODEL" \
  ${DRAFT_ARGS[@]+"${DRAFT_ARGS[@]}"} \
  --max-num-seqs 1 \
  --prefill-step-size "${PREFILL:-2048}" \
  --host 127.0.0.1 --port "${PORT:-8080}" \
  "${@:2}"
