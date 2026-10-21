#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/platform/macos/launchers/launch_tensorboard.sh" "$@"
