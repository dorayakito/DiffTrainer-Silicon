"""Central device selection for the DiffTrainer application layer."""
import os
import platform

# This must be set before importing torch in the parent or its child processes.
if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device():
    """Return the best available device name: CUDA, MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def trainer_accelerator():
    """Return the Lightning accelerator name for the selected device."""
    return {"cuda": "gpu", "mps": "mps", "cpu": "cpu"}[pick_device()]


def device_label():
    """Return a short, user-facing description of the selected device."""
    device = pick_device()
    return {
        "cuda": "NVIDIA CUDA (GPU)",
        "mps": "Apple Metal (MPS)",
        "cpu": "CPU",
    }[device]
