"""Apply the MPS device-selection patch to an extracted SOME tree."""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "platform" / "macos" / "patches" / "some-macos-device.patch"
MARKER = "from difftrainer_app.modules.device import pick_device"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("some_dir", type=Path)
    args = parser.parse_args()
    target = args.some_dir.resolve()
    base_infer = target / "inference" / "base_infer.py"
    if not base_infer.is_file():
        raise SystemExit(f"Not a SOME tree: {target}")
    if MARKER in base_infer.read_text(encoding="utf-8"):
        print("SOME MPS device patch already present")
        return
    subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(PATCH)],
        cwd=target,
        check=True,
    )
    print(f"Applied SOME MPS device patch to {target}")


if __name__ == "__main__":
    main()
