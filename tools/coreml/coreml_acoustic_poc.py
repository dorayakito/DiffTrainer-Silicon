#!/usr/bin/env python3
"""Convert and compare one real DiffSinger acoustic-backbone invocation.

Run this on an Apple Silicon Mac inside an environment that has DiffSinger's
dependencies plus coremltools installed. This is a conversion experiment, not
a Core ML inference backend for DiffTrainer.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path


class BackboneCall:
    """Expose the captured inference call with a Core ML-friendly signature."""

    def __init__(self, module, input_count):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, wrapped, count):
                super().__init__()
                self.wrapped = wrapped
                self.count = count

            def forward(self, *inputs, **kwargs):
                if self.count == 3:
                    condition = kwargs.get("cond")
                    if condition is not None and len(inputs) == 2:
                        return self.wrapped(inputs[0], inputs[1], condition)
                    return self.wrapped(*inputs)
                if self.count == 5:
                    return self.wrapped(*inputs, **kwargs)
                raise RuntimeError(f"Unsupported backbone input count: {self.count}")

        self.module = Wrapper(module, input_count).eval()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffsinger", type=Path, required=True,
                        help="Path to the downloaded DiffSinger checkout")
    parser.add_argument("--experiment", required=True,
                        help="Checkpoint directory name under DiffSinger/checkpoints")
    parser.add_argument("--ds", type=Path, required=True,
                        help="A .ds file containing at least one acoustic segment")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination .mlpackage path")
    parser.add_argument("--language", help="Language name for the .ds segment")
    parser.add_argument("--speaker", help="Speaker name or mixture for the .ds segment")
    parser.add_argument("--checkpoint-step", type=int,
                        help="Optional checkpoint step to load")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    diffsinger = args.diffsinger.expanduser().resolve()
    checkpoint_root = (diffsinger / "checkpoints").resolve()
    experiment = (checkpoint_root / args.experiment).resolve()
    ds_path = args.ds.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not (diffsinger / "scripts" / "infer.py").is_file():
        raise SystemExit(f"DiffSinger checkout not found: {diffsinger}")
    if checkpoint_root not in experiment.parents or not experiment.is_dir():
        raise SystemExit("--experiment must name an existing directory under DiffSinger/checkpoints")
    if not ds_path.is_file():
        raise SystemExit(f".ds file not found: {ds_path}")
    if output.suffix != ".mlpackage":
        raise SystemExit("--output must end in .mlpackage")
    if output.exists():
        raise SystemExit(f"Output already exists; choose a new path: {output}")
    if args.runs < 1 or args.warmup < 0:
        raise SystemExit("--runs must be positive and --warmup cannot be negative")

    try:
        import coremltools as ct
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Install PyTorch and Core ML Tools in this environment before running the PoC."
        ) from exc

    if sys.platform != "darwin" or not torch.backends.mps.is_available():
        raise SystemExit("Run this experiment on an Apple Silicon Mac with MPS available.")

    os.chdir(diffsinger)
    sys.path.insert(0, str(diffsinger.parent))
    sys.path.insert(0, str(diffsinger))
    # This checkout keeps DiffTrainer helpers under src/difftrainer_app/modules;
    # the local DiffSinger copy imports its device helper through dt_modules.
    sys.path.insert(0, str(diffsinger.parent / "src"))
    from difftrainer_app.modules import device as device_module
    dt_modules = types.ModuleType("dt_modules")
    dt_modules.__path__ = []
    sys.modules["dt_modules"] = dt_modules
    sys.modules["dt_modules.device"] = device_module
    from utils.hparams import set_hparams
    from utils.infer_utils import parse_commandline_spk_mix

    # DiffSinger's current set_hparams() parses sys.argv when config is empty;
    # passing exp_name as a Python argument alone is ignored in that path.
    sys.argv = [sys.argv[0], "--exp_name", args.experiment, "--infer"]
    set_hparams(print_hparams=False)
    from inference.ds_acoustic import DiffSingerAcousticInfer

    infer = DiffSingerAcousticInfer(
        device="cpu", load_vocoder=True, ckpt_steps=args.checkpoint_step
    )
    with ds_path.open(encoding="utf-8") as stream:
        params = json.load(stream)
    if isinstance(params, dict):
        params = [params]
    if not params:
        raise SystemExit("The .ds file has no segments.")

    param = dict(params[0])
    if args.language and not param.get("lang"):
        param["lang"] = args.language
    if args.speaker and not param.get("spk_mix"):
        param["spk_mix"] = parse_commandline_spk_mix(args.speaker)
    batch = infer.preprocess_input(param, idx=0)

    captured = []
    diffusion = infer.model.diffusion
    backbone_attr = next(
        (name for name in ("velocity_fn", "denoise_fn")
         if getattr(diffusion, name, None) is not None),
        None,
    )
    backbone = getattr(diffusion, backbone_attr, None) if backbone_attr else None
    if backbone is None:
        raise SystemExit("Could not locate the acoustic diffusion/reflow backbone.")

    def capture_call(_module, positional, keyword):
        if captured:
            return
        if "cond" in keyword:
            values = (positional[0], positional[1], keyword["cond"])
        else:
            values = positional
        if not all(torch.is_tensor(value) for value in values):
            return
        captured.append(tuple(value.detach().cpu().contiguous() for value in values))

    hook = backbone.register_forward_pre_hook(capture_call, with_kwargs=True)
    try:
        torch.manual_seed(1234)
        infer.forward_model(batch)
    finally:
        hook.remove()
    if not captured:
        raise SystemExit("Inference did not produce a capturable backbone call.")

    example_inputs = captured[0]
    if len(example_inputs) not in (3, 5):
        raise SystemExit(f"Unsupported inference call with {len(example_inputs)} tensor inputs.")
    wrapped = BackboneCall(backbone, len(example_inputs)).module
    with torch.inference_mode():
        traced = torch.jit.trace(wrapped, example_inputs, strict=False)
        torch_output = wrapped(*example_inputs).detach().cpu().numpy()

    input_names = ["spec", "diffusion_step", "condition"]
    if len(example_inputs) == 5:
        input_names += ["diffusion_step_2", "mask"]
    convert_started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, shape=tuple(value.shape))
                for name, value in zip(input_names, example_inputs)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    convert_seconds = time.perf_counter() - convert_started
    converted.save(str(output))

    load_started = time.perf_counter()
    converted = ct.models.MLModel(str(output), compute_units=ct.ComputeUnit.CPU_AND_NE)
    model_load_seconds = time.perf_counter() - load_started

    features = {name: value.numpy() for name, value in zip(input_names, example_inputs)}
    prediction_started = time.perf_counter()
    coreml_output = converted.predict(features)
    first_prediction_seconds = time.perf_counter() - prediction_started
    output_name = next(iter(coreml_output))
    coreml_values = np.asarray(coreml_output[output_name])
    difference = np.abs(torch_output - coreml_values)

    plan_summary = {"status": "unavailable; inspect the package in Xcode"}
    try:
        compute_plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            converted.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        program = compute_plan.model_structure.program
        if program is not None:
            usage_counts = {}
            operation_count = 0
            for function in program.functions.values():
                for operation in function.block.operations:
                    operation_count += 1
                    usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(operation)
                    preferred = (
                        type(usage.preferred_compute_device).__name__
                        if usage is not None else "unknown"
                    )
                    usage_counts[preferred] = usage_counts.get(preferred, 0) + 1
            plan_summary = {
                "status": "estimated local compute plan",
                "operations": operation_count,
                "preferred_device_counts": usage_counts,
            }
        else:
            plan_summary = {"status": "model structure did not contain an ML Program"}
    except Exception as exc:
        plan_summary = {"status": "unavailable; inspect the package in Xcode", "reason": str(exc)}

    def run_acoustic(use_coreml):
        if use_coreml:
            class CoreMLBackbone(torch.nn.Module):
                def forward(self, *inputs, **kwargs):
                    ordered = inputs
                    if kwargs.get("cond") is not None and len(inputs) == 2:
                        ordered = (inputs[0], inputs[1], kwargs["cond"])
                    values = {
                        name: value.detach().cpu().contiguous().numpy()
                        for name, value in zip(input_names, ordered)
                    }
                    result = converted.predict(values)
                    return torch.as_tensor(
                        result[output_name], dtype=ordered[0].dtype, device=ordered[0].device
                    )

            setattr(diffusion, backbone_attr, CoreMLBackbone())
        else:
            setattr(diffusion, backbone_attr, backbone)
        torch.manual_seed(1234)
        with torch.inference_mode():
            acoustic_started = time.perf_counter()
            mel = infer.forward_model(batch)
            acoustic_seconds = time.perf_counter() - acoustic_started
            audio = infer.run_vocoder(mel, f0=batch["f0"])[0]
        return mel.detach().cpu().numpy(), audio.detach().cpu().numpy(), acoustic_seconds

    for _ in range(args.warmup):
        run_acoustic(False)
        run_acoustic(True)

    torch_times = []
    coreml_times = []
    torch_outputs = None
    coreml_outputs = None
    for _ in range(args.runs):
        torch_outputs = run_acoustic(False)
        torch_times.append(torch_outputs[2])
        coreml_outputs = run_acoustic(True)
        coreml_times.append(coreml_outputs[2])

    setattr(diffusion, backbone_attr, backbone)
    full_mel_error = np.abs(torch_outputs[0] - coreml_outputs[0])
    audio_error = np.abs(torch_outputs[1] - coreml_outputs[1])

    print(json.dumps({
        "model_scope": "DiffSinger acoustic diffusion/reflow backbone, one invocation",
        "experiment": args.experiment,
        "checkpoint_step": args.checkpoint_step,
        "input_shapes": {name: list(value.shape) for name, value in zip(input_names, example_inputs)},
        "output_shape": list(torch_output.shape),
        "conversion_seconds": round(convert_seconds, 4),
        "model_load_seconds": round(model_load_seconds, 4),
        "first_coreml_prediction_seconds": round(first_prediction_seconds, 4),
        "torch_median_seconds": round(statistics.median(torch_times), 6),
        "coreml_median_seconds": round(statistics.median(coreml_times), 6),
        "full_acoustic_output": {
            "shape": list(torch_outputs[0].shape),
            "finite": bool(np.isfinite(torch_outputs[0]).all() and np.isfinite(coreml_outputs[0]).all()),
            "max_absolute_error": float(full_mel_error.max()),
            "mean_absolute_error": float(full_mel_error.mean()),
        },
        "vocoder_audio": {
            "shape": list(torch_outputs[1].shape),
            "finite": bool(np.isfinite(torch_outputs[1]).all() and np.isfinite(coreml_outputs[1]).all()),
            "max_absolute_error": float(audio_error.max()),
            "mean_absolute_error": float(audio_error.mean()),
        },
        "single_call_outputs_finite": bool(np.isfinite(torch_output).all() and np.isfinite(coreml_values).all()),
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "coreml_package": str(output),
        "compute_units_requested": "CPU_AND_NE (CPU and Neural Engine; GPU excluded)",
        "compute_plan": plan_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
