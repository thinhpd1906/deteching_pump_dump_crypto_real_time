#!/usr/bin/env bash
# =====================================================================
# One-command Mac mini setup + hand-over to Claude Code.
#
# On a fresh Mac, run in Terminal:
#   curl -fsSL https://raw.githubusercontent.com/thinhpd1906/deteching_pump_dump_crypto_real_time/main/scripts/mac/bootstrap.sh -o /tmp/pnd-bootstrap.sh && bash /tmp/pnd-bootstrap.sh
#
# Safe to re-run: every step checks whether it is already done.
# It will ask you for: your Mac password (sudo, once or twice), the
# contents of kaggle.json (first run only), and Claude login (first run).
# Everything else is automatic. Full log: ~/pnd-bootstrap.log
#
# Env options:
#   PND_DIR=~/pnd-thesis     where the repo lives
#   PND_SKIP_POWER=1         skip pmset / auto-update / SSH changes
#   PND_UNATTENDED=1         start Claude with --dangerously-skip-permissions
#                            (no permission prompts; only on this dedicated Mac)
# =====================================================================
set -euo pipefail

REPO_URL="https://github.com/thinhpd1906/deteching_pump_dump_crypto_real_time.git"
PND_DIR="${PND_DIR:-$HOME/pnd-thesis}"
DL_DIR="$HOME/pnd-dl"
LOG="$HOME/pnd-bootstrap.log"
BRANCH="mac/phase-a"
exec 3>&1 4>&2                      # keep the real terminal for the final tmux attach
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok] %s\033[0m\n' "$*"; }
TODO=()

step "0. Preflight"
[[ "$(uname -s)" == "Darwin" ]] || { echo "This script is for macOS."; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || warn "Not Apple Silicon; MPS will not be available."
echo "started $(date -u +%FT%TZ)  repo dir: $PND_DIR"

step "1. Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
eval "$(/opt/homebrew/bin/brew shellenv)"
grep -q 'brew shellenv' "$HOME/.zprofile" 2>/dev/null || \
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
ok "brew $(brew --version | head -1)"

step "2. Tools"
brew install git gh uv tmux jq libomp openjdk@17 colima docker docker-compose
brew list --cask tailscale >/dev/null 2>&1 || brew install --cask tailscale
ok "tools installed"

step "3. Repository"
if [[ -d "$PND_DIR/.git" ]]; then
  git -C "$PND_DIR" fetch --all --prune
  git -C "$PND_DIR" checkout main && git -C "$PND_DIR" pull --ff-only || \
    warn "could not fast-forward main (local changes?) - continuing with current tree"
else
  git clone "$REPO_URL" "$PND_DIR"
fi
cd "$PND_DIR"
git config user.name  >/dev/null || git config user.name  "thinhpd"
git config user.email >/dev/null || git config user.email "phanthinhpdt@gmail.com"
ok "repo at $(git rev-parse --short HEAD)"

step "4. Python 3.11 venv + dependencies"
uv python install 3.11
[[ -d .venv ]] || uv venv --python 3.11 .venv
set +u
# shellcheck disable=SC1091
source .venv/bin/activate
set -u
uv pip install -r requirements.txt optuna pytest kaggle
python - <<'EOF'
import torch
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
EOF

step "5. Kaggle credentials"
mkdir -p "$HOME/.kaggle"
if [[ ! -s "$HOME/.kaggle/kaggle.json" ]]; then
  echo "Paste the full contents of kaggle.json (Windows: C:\\Users\\phant\\.kaggle\\kaggle.json,"
  echo "or kaggle.com -> Settings -> API -> Create New Token), then press Enter:"
  read -r KJSON
  printf '%s\n' "$KJSON" > "$HOME/.kaggle/kaggle.json"
fi
chmod 600 "$HOME/.kaggle/kaggle.json"
ok "kaggle.json present"

step "6. Data (gitignored) -> Kaggle datasets + raw tarball"
mkdir -p "$DL_DIR" artifacts/adaptive data
if python scripts/make_data_manifest.py --verify --only processed >/dev/null 2>&1; then
  ok "processed data already verified"
else
  rm -rf "$DL_DIR/processed_dl"
  kaggle datasets download -d phancthnh/pnd-thesis-processed -p "$DL_DIR/processed_dl" --unzip
  if [[ -f "$DL_DIR/processed_dl/pnd-processed.zip" ]]; then
    unzip -oq "$DL_DIR/processed_dl/pnd-processed.zip" -d data/
  elif [[ -d "$DL_DIR/processed_dl/processed" ]]; then
    cp -R "$DL_DIR/processed_dl/processed" data/
  else
    warn "unexpected layout in $DL_DIR/processed_dl:"; ls -R "$DL_DIR/processed_dl" | head -20
  fi
fi
if python scripts/make_data_manifest.py --verify --only checkpoints >/dev/null 2>&1; then
  ok "checkpoint already verified"
else
  rm -rf "$DL_DIR/ckpt_dl"
  kaggle datasets download -d phancthnh/pnd-thesis-checkpoint -p "$DL_DIR/ckpt_dl" --unzip
  CKPT="$(find "$DL_DIR/ckpt_dl" -name at_pretrained.pt | head -1)"
  if [[ -n "$CKPT" ]]; then cp "$CKPT" artifacts/adaptive/
  else warn "at_pretrained.pt not found in $DL_DIR/ckpt_dl"; fi
fi
if [[ -f "$DL_DIR/pnd-raw.tgz" && ! -d data/raw ]]; then
  tar -xzf "$DL_DIR/pnd-raw.tgz"
fi
[[ -d data/raw ]] || TODO+=("Copy raw data from Windows (PowerShell):  scp C:\\Users\\phant\\Desktop\\pnd-raw.tgz $(whoami)@<this-mac>:~/pnd-dl/   then re-run this script. (Training works without it; only dataset rebuilds/Gate 1 need raw.)")

VERIFY_GROUPS="labels processed checkpoints"   # (not GROUPS: that name is reserved in bash)
if [[ -d data/raw ]]; then VERIFY_GROUPS="$VERIFY_GROUPS raw"; fi
# shellcheck disable=SC2086
if python scripts/make_data_manifest.py --verify --only $VERIFY_GROUPS; then
  ok "data parity with Windows confirmed ($VERIFY_GROUPS)"
else
  warn "DATA MISMATCH - Claude will be told to stop at task A0 until fixed."
fi

step "7. Always-on power settings + SSH (needs your password)"
if [[ -z "${PND_SKIP_POWER:-}" ]]; then
  sudo pmset -a sleep 0 disksleep 0 displaysleep 10 powernap 0 autorestart 1 womp 1
  sudo defaults write /Library/Preferences/com.apple.SoftwareUpdate AutomaticallyInstallMacOSUpdates -bool false
  sudo systemsetup -setremotelogin on 2>/dev/null || \
    TODO+=("Turn on SSH: System Settings -> General -> Sharing -> Remote Login: ON")
  pmset -g | grep -E ' sleep|autorestart' || true
  fdesetup status | grep -q "On" && TODO+=("FileVault is ON: after a power cut the Mac waits at the unlock screen. Test whether you can unlock over SSH after a reboot; otherwise decide whether to turn FileVault off (see CLAUDE.md section 19).")
else
  warn "PND_SKIP_POWER set - skipped"
fi
TODO+=("Open the Tailscale app on this Mac and log in (same account as Windows), then from Windows: ssh $(whoami)@<mac-tailscale-name>")
TODO+=("Recommended: plug the Mac into a small UPS.")

step "8. Docker runtime for Kafka (colima)"
colima status >/dev/null 2>&1 || colima start --cpu 4 --memory 8 || warn "colima failed to start; task A5 will retry"

step "9. Claude Code"
command -v claude >/dev/null 2>&1 || curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
command -v claude >/dev/null 2>&1 || { warn "claude not on PATH; open a new terminal and re-run"; exit 1; }
ok "claude $(claude --version 2>/dev/null || echo installed)"

step "10. Work branch"
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"
ok "on branch $(git branch --show-current)"

step "11. Remaining manual items"
if ((${#TODO[@]})); then for t in "${TODO[@]}"; do echo " - $t"; done; else echo " none"; fi
printf '%s\n' "${TODO[@]:-}" > "$HOME/pnd-bootstrap-todo.txt"

step "12. Hand over to Claude (tmux session 'thesis')"
KICKOFF="We are on the Mac mini; bootstrap just finished (log: ~/pnd-bootstrap.log, open manual items: ~/pnd-bootstrap-todo.txt). Read CLAUDE.md fully, especially section 19 (Mac plan). Execute Phase A in order starting at task A0. Stop at every 'STOP: ask Bryan' point and wait for my answer. Before each task tell me in two sentences what you will change; after each task report the acceptance check and commit to branch mac/phase-a."
CLAUDE_FLAGS=""
[[ -n "${PND_UNATTENDED:-}" ]] && CLAUDE_FLAGS="--dangerously-skip-permissions"
printf '%s' "$KICKOFF" > "$HOME/.pnd-kickoff.txt"
if tmux has-session -t thesis 2>/dev/null; then
  warn "tmux session 'thesis' already exists - attaching instead of starting a second Claude"
else
  tmux new-session -d -s thesis -c "$PND_DIR" \
    "source .venv/bin/activate; export PYTORCH_ENABLE_MPS_FALLBACK=1; claude $CLAUDE_FLAGS \"\$(cat ~/.pnd-kickoff.txt)\"; exec \$SHELL -l"
fi
echo "Detach with Ctrl-b then d. Re-attach any time (also over SSH): tmux attach -t thesis"
echo "If Claude asked you to log in first and then sat idle, paste the kickoff prompt from ~/.pnd-kickoff.txt"
sleep 1
exec 1>&3 2>&4                      # back to the real terminal
if [[ -t 1 ]]; then exec tmux attach -t thesis; fi
