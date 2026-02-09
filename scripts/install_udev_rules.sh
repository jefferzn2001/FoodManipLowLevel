#!/usr/bin/env bash
set -euo pipefail

RULE_SRC="udev/80-i2rt-can.rules"
RULE_DST="/etc/udev/rules.d/80-i2rt-can.rules"

if [[ ! -f "$RULE_SRC" ]]; then
  echo "ERROR: Missing $RULE_SRC"
  exit 1
fi

echo "[i2rt] Installing udev rule: $RULE_SRC -> $RULE_DST"
sudo cp "$RULE_SRC" "$RULE_DST"

echo "[i2rt] Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "[i2rt] Done. Unplug/replug the CAN adapters."
echo "[i2rt] Verify with: ls -l /sys/class/net/can*"

