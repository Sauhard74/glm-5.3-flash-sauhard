#!/usr/bin/env bash
# Run Claude Code against the local model served by scripts/quickserve.sh (start that first).
set -euo pipefail
MODEL_ID="${MODEL_ID:-models/qwen-flash-next-reap288}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8080}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$(cat "$(dirname "$0")/../control/api_key")}"
export ANTHROPIC_MODEL="$MODEL_ID" ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL_ID" CLAUDE_CODE_SUBAGENT_MODEL="$MODEL_ID"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-131072}"
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
exec claude "$@"
