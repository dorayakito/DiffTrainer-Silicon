"""Compatibility entry point for the DiffTrainer desktop app."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from difftrainer_app.gui.difftrainer import App  # noqa: E402

if __name__ == "__main__":
    app = App()
    app.mainloop()
