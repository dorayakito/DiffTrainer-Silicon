"""Shared paths and migration for application resources and user settings."""
from __future__ import annotations

import shutil
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
RESOURCE_ROOT = PACKAGE_ROOT / "resources"
ASSETS_ROOT = RESOURCE_ROOT / "assets"
STRINGS_ROOT = RESOURCE_ROOT / "strings"
CONFIG_ROOT = REPOSITORY_ROOT / "config"
SETTINGS_PATH = CONFIG_ROOT / "guisettings.yaml"
LEGACY_SETTINGS_PATH = ASSETS_ROOT / "guisettings.yaml"


def ensure_user_config() -> Path:
    """Move settings into config once, retaining the old copy for rollback."""
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists() and LEGACY_SETTINGS_PATH.exists():
        shutil.copy2(LEGACY_SETTINGS_PATH, SETTINGS_PATH)
    return SETTINGS_PATH
