#!/bin/bash
# Keeps the thesis work alive 24/7 on the Mac mini.
# Run by launchd (scripts/mac/com.thinhpd.pnd-watchdog.plist) at login and every 10 min.
# Idempotent: when everything is healthy it only writes a heartbeat line.
#   1. tmux session 'thesis' exists, with a Claude Code session in it
#      (recreated after a reboot/crash and told to resume under /loop).
#   2. The A4 search (scripts/search.py) is running until it finishes cleanly
#      (it resumes from its Optuna DBs, finished trials are never lost).
#   3. Disk space warning.
# Pause everything: touch ~/.pnd-watchdog/pause   (resume: rm that file)
# Log: ~/Library/Logs/pnd-watchdog.log
set -u
PND_DIR="$HOME/pnd-thesis"
STATE="$HOME/.pnd-watchdog"
LOG="$HOME/Library/Logs/pnd-watchdog.log"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
mkdir -p "$STATE" "$(dirname "$LOG")"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

[[ -f "$STATE/pause" ]] && { log "paused (rm $STATE/pause to resume)"; exit 0; }
cd "$PND_DIR" || { log "ERROR no $PND_DIR"; exit 0; }

# allow at most N restarts of <what> per <window> seconds (stops crash loops)
may_restart() {
  local what=$1 max=$2 window=$3 now f
  now=$(date +%s); f="$STATE/restarts_$what"
  touch "$f"
  awk -v n="$now" -v w="$window" '$1 > n - w' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  if (( $(wc -l < "$f") >= max )); then return 1; fi
  echo "$now" >> "$f"; return 0
}

ENV_SETUP="source .venv/bin/activate; export PYTORCH_ENABLE_MPS_FALLBACK=1"

# ---------- 1. tmux session + Claude ----------
claude_running() {
  local t
  for t in $(tmux list-panes -s -t thesis -F '#{pane_tty}' 2>/dev/null); do
    ps -t "${t#/dev/}" -o comm= 2>/dev/null | grep -qx claude && return 0
  done
  return 1
}
start_claude() {
  local cmd="$ENV_SETUP; claude --permission-mode auto \"\$(cat scripts/mac/resume_prompt.txt)\"; exec \$SHELL -l"
  if tmux has-session -t thesis 2>/dev/null; then
    tmux new-window -t thesis -n claude -c "$PND_DIR" "$cmd"
  else
    tmux new-session -d -s thesis -n claude -c "$PND_DIR" "$cmd"
  fi
}
if ! claude_running; then
  if may_restart claude 3 21600; then
    start_claude && log "Claude was not running -> started with resume prompt (/loop)"
  else
    log "WARN Claude keeps dying (3 restarts in 6 h); not restarting. Check: tmux attach -t thesis"
  fi
fi

# ---------- 2. A4 search ----------
SDIR="artifacts/search"
if [[ -d "$SDIR" ]] && ls "$SDIR"/optuna_*.db >/dev/null 2>&1; then
  if pgrep -f "scripts/search.py" >/dev/null; then
    :
  elif [[ -f "$SDIR/search.done" ]] && grep -qx "exit=0" "$SDIR/search.done"; then
    :   # finished cleanly (budget, no-improve or stop_at reached)
  elif may_restart search 3 86400; then
    rm -f "$SDIR/search.done"
    tmux new-window -d -t thesis -n a4 -c "$PND_DIR" \
      "bash -c '$ENV_SETUP; caffeinate -is python scripts/search.py --phase all 2>&1 | tee -a $SDIR/search.log; echo \"exit=\${PIPESTATUS[0]}\" > $SDIR/search.done; exec \$SHELL -l'"
    log "search.py was not running and not finished -> restarted (resumes from Optuna DB)"
  else
    log "WARN search.py failed 3 times in 24 h; not restarting. See $SDIR/search.log"
  fi
fi

# ---------- 3. disk ----------
free_gb=$(df -g "$PND_DIR" | awk 'NR==2 {print $4}')
(( free_gb < 20 )) && log "WARN low disk: ${free_gb} GB free"

log "heartbeat ok (disk ${free_gb} GB free)"
# keep the log small
if (( $(wc -l < "$LOG") > 5000 )); then tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
exit 0
