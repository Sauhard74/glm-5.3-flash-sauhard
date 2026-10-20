#!/usr/bin/env bash
# Render the LaunchAgent and (optionally) load it.
#
#   scripts/launchd/install.sh <model> [--port 8642] [--memory-limit-gb 80] [--idle-unload-s 900]
#                              [--out FILE] [--load]
#   scripts/launchd/install.sh --unload
#
# Without --load it only writes the plist (default ~/Library/LaunchAgents/com.local-ai.serve.plist).
set -euo pipefail

label="com.local-ai.serve"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
target="$HOME/Library/LaunchAgents/$label.plist"

if [[ "${1:-}" == "--unload" ]]; then
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  echo "unloaded $label"
  exit 0
fi

model="${1:?usage: install.sh <model> [--port N] [--memory-limit-gb N] [--idle-unload-s N] [--out FILE] [--load]}"
shift
port=8642 memory=80 idle=900 load=0 out="$target"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    --memory-limit-gb) memory="$2"; shift 2 ;;
    --idle-unload-s) idle="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --load) load=1; shift ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done

uv_bin="$(command -v uv)" || { echo "uv not on PATH" >&2; exit 1; }
mkdir -p "$repo/logs" "$(dirname "$out")"
sed -e "s|@REPO@|$repo|g" \
    -e "s|@UV_DIR@|$(dirname "$uv_bin")|g" \
    -e "s|@MODEL@|$model|g" \
    -e "s|@PORT@|$port|g" \
    -e "s|@MEMORY_LIMIT_GB@|$memory|g" \
    -e "s|@IDLE_UNLOAD_S@|$idle|g" \
    "$here/$label.plist.template" > "$out"
plutil -lint "$out" >/dev/null
echo "wrote $out"

if [[ $load == 1 ]]; then
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$out"
  echo "loaded $label (logs: $repo/logs/serve.log)"
fi
