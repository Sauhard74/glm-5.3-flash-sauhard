#!/usr/bin/env bash
# Install the control daemon as a LaunchAgent: starts at login, restarts if it dies, survives closing the terminal.
# Dashboard: http://127.0.0.1:8090    Uninstall: scripts/install-control.sh --uninstall
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=com.local-ai.control
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_N="$(id -u)"

launchctl bootout "gui/$UID_N/$LABEL" 2>/dev/null || true
if [[ "${1:-}" == "--uninstall" ]]; then rm -f "$PLIST"; echo "uninstalled"; exit 0; fi

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/logs"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$HOME/.local/bin/uv</string><string>run</string><string>python</string><string>-m</string><string>control.daemon</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>AbandonProcessGroup</key><true/>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$ROOT/logs/control.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/control.log</string>
</dict>
</plist>
EOF
launchctl bootstrap "gui/$UID_N" "$PLIST"
echo "installed $PLIST — dashboard at http://127.0.0.1:8090"
