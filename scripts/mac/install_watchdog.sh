#!/bin/bash
# Installs (or re-installs) the launchd agent that runs scripts/mac/watchdog.sh
# at login and every 10 minutes.  Uninstall:  bash scripts/mac/install_watchdog.sh --uninstall
set -euo pipefail
LABEL="com.thinhpd.pnd-watchdog"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
SCRIPT="$HOME/pnd-thesis/scripts/mac/watchdog.sh"
DOMAIN="gui/$(id -u)"

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
if [[ "${1:-}" == "--uninstall" ]]; then
  rm -f "$PLIST"; echo "watchdog removed"; exit 0
fi

chmod +x "$SCRIPT"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$SCRIPT</string></array>
  <key>RunAtLoad</key><true/>
  <key>StartInterval</key><integer>600</integer>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/pnd-watchdog.out</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/pnd-watchdog.out</string>
</dict>
</plist>
EOF
launchctl bootstrap "$DOMAIN" "$PLIST"
echo "watchdog installed: $PLIST (runs at login + every 10 min)"
echo "log: ~/Library/Logs/pnd-watchdog.log    pause: touch ~/.pnd-watchdog/pause"
