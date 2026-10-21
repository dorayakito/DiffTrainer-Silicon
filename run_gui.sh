#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/platform/macos/launchers/run_gui.sh" "$@"
