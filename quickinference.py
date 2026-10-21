"""Compatibility entry point for Quick Inference."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# The implementation constructs and runs its Tk window when imported.
from difftrainer_app.inference import quickinference as _quickinference  # noqa: F401,E402
