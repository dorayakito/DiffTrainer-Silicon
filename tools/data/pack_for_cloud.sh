#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 DIFFSINGER_BINARY_DIR CONFIG_YAML OUTPUT.tar.gz" >&2
  exit 2
fi

binary_dir="$(cd "$1" && pwd)"
config_file="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
output_path="$3"

if [[ ! -d "$binary_dir" ]]; then
  echo "Binary dataset directory not found: $binary_dir" >&2
  exit 1
fi
if [[ ! -f "$config_file" ]]; then
  echo "Training config not found: $config_file" >&2
  exit 1
fi

mkdir -p "$(dirname "$output_path")"
output_path="$(cd "$(dirname "$output_path")" && pwd)/$(basename "$output_path")"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/binary"
cp -R "$binary_dir"/. "$stage/binary/"
cp "$config_file" "$stage/config.yaml"

python - "$stage/config.yaml" <<'PY'
import sys
import yaml
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
config["binary_data_dir"] = "binary"
with path.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
PY

 tar -czf "$output_path" -C "$stage" .
echo "Created $output_path"
