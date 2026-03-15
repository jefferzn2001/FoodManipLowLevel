#!/bin/sh

USER_ID="$(id -u)"
USER=$(logname)

if [ "$USER_ID" -ne 0 ]; then
    echo "Please run this as root."
    exit 1
fi

INSTALL_DIR=$(dirname "$0")
# Resolve to absolute path so we find rules regardless of cwd when run as sudo
if [ -d "$INSTALL_DIR" ]; then
    INSTALL_DIR=$(cd "$INSTALL_DIR" && pwd)
fi
RULES_DIR="$INSTALL_DIR/rules"

if [ ! -d "$RULES_DIR" ]; then
    echo "No rules directory at $RULES_DIR; skipping udev rule install."
    echo "Optional: run from repo root: sudo bash scripts/install_udev_rules.sh"
else
    for udev_rule in "$RULES_DIR"/*.rules; do
        [ -f "$udev_rule" ] || continue
        rule=$(basename "$udev_rule")
        echo "Installing $rule"
        cp "$udev_rule" "/etc/udev/rules.d/$rule"
    done
fi

echo "Adding $USER to group plugdev, video"
adduser $USER plugdev
adduser $USER video

echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo "Finished, please re-insert devices."
