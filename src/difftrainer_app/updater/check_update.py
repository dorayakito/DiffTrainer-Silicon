"""Check for and install application updates without touching user data."""
from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests
from tkinter import messagebox
from tqdm import tqdm

from difftrainer_app.paths import REPOSITORY_ROOT, ensure_user_config

REPOSITORY = "https://github.com/dorayakito/DiffTrainer-Silicon"
VERSION_URL = f"{REPOSITORY}/main/src/difftrainer_app/gui/difftrainer.py"
ARCHIVE_URL = f"{REPOSITORY}/archive/refs/heads/main.zip"
VERSION_PATTERN = re.compile(r'^version\s*=\s*[\'\"]([^\'\"]+)[\'\"]', re.MULTILINE)
ROOT_FILES = ("difftrainer.py", "quickinference.py", "auto_torch.py", "check_update.py", "requirements.txt")


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _repo_root(extracted: Path) -> Path:
    candidates = list(extracted.glob("*/src/difftrainer_app/gui/difftrainer.py"))
    if len(candidates) != 1:
        raise ValueError("Update archive has no unique DiffTrainer src package")
    return candidates[0].parents[3]


def install_archive(archive_path: Path, repo_root: Path = REPOSITORY_ROOT) -> None:
    """Install app code from a GitHub archive, preserving all local data."""
    repo_root = Path(repo_root).resolve()
    with tempfile.TemporaryDirectory(prefix="difftrainer-update-") as temporary:
        temporary_path = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe path in update archive: {member.filename}")
            archive.extractall(temporary_path)
        source_root = _repo_root(temporary_path)
        source_package = source_root / "src" / "difftrainer_app"
        if not source_package.is_dir():
            raise ValueError("Update archive is missing src/difftrainer_app")

        staged_package = temporary_path / "staged-difftrainer_app"
        shutil.copytree(source_package, staged_package)
        package_parent = repo_root / "src"
        package_parent.mkdir(parents=True, exist_ok=True)
        target_package = package_parent / "difftrainer_app"
        backup_package = temporary_path / "previous-difftrainer_app"
        if target_package.exists():
            target_package.rename(backup_package)
        try:
            staged_package.rename(target_package)
            for filename in ROOT_FILES:
                source = source_root / filename
                if source.is_file():
                    shutil.copy2(source, repo_root / filename)
        except Exception:
            if target_package.exists():
                shutil.rmtree(target_package)
            if backup_package.exists():
                backup_package.rename(target_package)
            raise


def check_for_updates() -> None:
    local_source = (REPOSITORY_ROOT / "src/difftrainer_app/gui/difftrainer.py").read_text(encoding="utf-8")
    local_match = VERSION_PATTERN.search(local_source)
    if not local_match:
        raise RuntimeError("Could not read the installed app version")
    local_version = local_match.group(1)

    response = requests.get(VERSION_URL, timeout=30)
    response.raise_for_status()
    remote_match = VERSION_PATTERN.search(response.text)
    if not remote_match:
        raise RuntimeError("Could not read the published app version")
    remote_version = remote_match.group(1)
    if version_tuple(local_version) >= version_tuple(remote_version):
        return

    if not messagebox.askyesno(
        "DiffTrainer update",
        f"DiffTrainer {remote_version} is available. You have {local_version}. Update the app code now?",
    ):
        return

    old_requirements = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")
    ensure_user_config()
    with tempfile.TemporaryDirectory(prefix="difftrainer-download-") as temporary:
        archive_path = Path(temporary) / "main.zip"
        with requests.get(ARCHIVE_URL, stream=True, timeout=60) as download:
            download.raise_for_status()
            total = int(download.headers.get("content-length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading DiffTrainer") as progress:
                with archive_path.open("wb") as output:
                    for chunk in download.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            output.write(chunk)
                            progress.update(len(chunk))
        install_archive(archive_path)

    requirements_url = f"{REPOSITORY}/main/requirements.txt"
    remote_requirements = requests.get(requirements_url, timeout=30)
    remote_requirements.raise_for_status()
    if old_requirements != remote_requirements.text:
        messagebox.showinfo(
            "Environment may need updating",
            "Python requirements changed. Update the environment if the app reports missing or incompatible packages.",
        )


if __name__ == "__main__":
    check_for_updates()
