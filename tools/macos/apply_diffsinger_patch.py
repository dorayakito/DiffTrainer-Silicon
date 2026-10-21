"""Apply maintained macOS compatibility patches to an extracted DiffSinger tree."""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCHES = (
    ROOT / "platform" / "macos" / "patches" / "diffsinger-macos-dataloader.patch",
    ROOT / "platform" / "macos" / "patches" / "diffsinger-macos-devices.patch",
)


def apply_if_needed(target, patch_path, marker):
    marker_path, marker_text = marker
    source = target / marker_path
    if marker_text in source.read_text(encoding="utf-8"):
        print(f"Patch already present: {patch_path.name}")
        return
    subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(patch_path)],
        cwd=target,
        check=True,
    )
    print(f"Applied {patch_path.name} to {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diffsinger_dir", type=Path)
    args = parser.parse_args()
    target = args.diffsinger_dir.resolve()
    if not (target / "basics" / "base_task.py").is_file():
        raise SystemExit(f"Not a DiffSinger tree: {target}")
    apply_if_needed(
        target,
        PATCHES[0],
        (Path("basics/base_task.py"), "loader_options = {'num_workers': num_workers"),
    )
    apply_if_needed(
        target,
        PATCHES[1],
        (Path("basics/base_svs_infer.py"), "device = pick_device()"),
    )


if __name__ == "__main__":
    main()
