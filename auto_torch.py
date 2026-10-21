"""Compatibility entry point for the PyTorch setup helper."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
runpy.run_module("difftrainer_app.updater.auto_torch", run_name="__main__")
