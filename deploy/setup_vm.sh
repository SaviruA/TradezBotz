#!/usr/bin/env bash
# Provision a fresh Debian VM to run the TradezBotz research pipeline.
#
#   ./setup_vm.sh /path/to/repo
#
# Idempotent: safe to re-run after a code change or a reboot.
set -euo pipefail

REPO_DIR="${1:-$HOME/tradezbotz}"
SERVICE_USER="${SUDO_USER:-$USER}"

echo "==> installing system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git sqlite3 tzdata

echo "==> creating virtualenv"
python3 -m venv "$REPO_DIR/.venv"
"$REPO_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$REPO_DIR/.venv/bin/pip" install --quiet -r "$REPO_DIR/requirements.txt"

echo "==> checking configuration"
if [[ ! -f "$REPO_DIR/.env" ]]; then
  echo "!! $REPO_DIR/.env is missing."
  echo "   Copy .env.example to .env and fill in SEC_USER_AGENT and MASSIVE_API_KEY."
  echo "   Do NOT commit it -- .gitignore already excludes it."
  exit 1
fi
chmod 600 "$REPO_DIR/.env"

if grep -q "your.email@example.com" "$REPO_DIR/.env"; then
  echo "!! SEC_USER_AGENT still holds the placeholder email."
  echo "   The SEC blocks clients without a real contact address."
  exit 1
fi

echo "==> installing systemd units"
for unit in tradezbotz-ingest.service tradezbotz-ingest.timer tradezbotz-backfill.service; do
  sed -e "s|@REPO_DIR@|$REPO_DIR|g" -e "s|@USER@|$SERVICE_USER|g" \
      "$REPO_DIR/deploy/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done

sudo systemctl daemon-reload
sudo systemctl enable --now tradezbotz-ingest.timer

echo
echo "==> done"
"$REPO_DIR/.venv/bin/python" -m tradezbotz status
cat <<'EOF'

Next:
  # one-time historical pull (two years, matching the price window)
  ./.venv/bin/python -m tradezbotz ingest-edgar --days 730
  ./.venv/bin/python -m tradezbotz enqueue-symbols

  # start the long backfill under systemd so it survives logout
  sudo systemctl start tradezbotz-backfill
  journalctl -u tradezbotz-backfill -f

  # daily ingest runs automatically; check it with
  systemctl list-timers tradezbotz-ingest.timer
EOF
