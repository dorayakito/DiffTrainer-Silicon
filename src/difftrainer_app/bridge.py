"""JSON-lines command bridge used by the native macOS client."""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ACTIVE: subprocess.Popen[str] | None = None


def emit(event: str, **data: Any) -> None:
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def required(payload: dict[str, Any], *names: str) -> list[str]:
    values = []
    for name in names:
        value = str(payload.get(name, "")).strip()
        if not value:
            raise ValueError(f"Required field: {name}")
        values.append(value)
    return values


def run_child(args: list[str], cwd: Path) -> None:
    global ACTIVE
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in (ROOT, ROOT / "src", cwd, Path.cwd()) if p)
    ACTIVE = subprocess.Popen(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, start_new_session=True)
    assert ACTIVE.stdout is not None
    for line in ACTIVE.stdout:
        emit("log", message=line.rstrip())
    code = ACTIVE.wait()
    ACTIVE = None
    if code:
        raise RuntimeError(f"Process exited with status {code}")


def status(_: dict[str, Any]) -> dict[str, Any]:
    diff = ROOT / "DiffSinger"
    return {"checkout": str(ROOT), "diffsinger": diff.is_dir(), "python": sys.executable,
            "python_version": sys.version.split()[0], "device": device_label(),
            "app_version": app_version(), "configs": str(diff / "configs")}


def device_label() -> str:
    try:
        import torch
        return "MPS" if torch.backends.mps.is_available() else "CPU"
    except Exception:
        return "Unknown (PyTorch unavailable)"


def app_version() -> str:
    try:
        text = (ROOT / "src/difftrainer_app/gui/difftrainer.py").read_text(encoding="utf8")
        for line in text.splitlines():
            if line.startswith("version ="):
                return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return "Unknown"


def speakers(payload: dict[str, Any]) -> dict[str, Any]:
    (folder,) = required(payload, "data_folder")
    base = Path(folder)
    if not base.is_dir(): raise ValueError(f"Data folder not found: {base}")
    children = sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(children) == 1 and children[0].name == "wavs": children = [base]
    output = []
    ids: dict[str, int] = {}
    for path in children:
        if not (path / "transcriptions.csv").is_file(): continue
        base_name = path.name.split(".")[0]
        ids.setdefault(base_name, len(ids))
        parts = path.name.rsplit(".", 1)
        output.append({"folder": str(path), "name": base_name, "language": parts[1] if len(parts) == 2 else "other", "id": ids[base_name]})
    return {"speakers": output}


def prepare(payload: dict[str, Any]) -> dict[str, Any]:
    (source,) = required(payload, "input_folder")
    from difftrainer_app.modules import liteconvert, corpus_segmenter
    source_path = Path(source)
    if not source_path.is_dir(): raise ValueError(f"Input folder not found: {source}")
    target = source_path
    if payload.get("segment", False):
        target = ROOT / "raw_data" / f"{source_path.name}_segmented"
        target.mkdir(parents=True, exist_ok=True)
        report = target / "report.txt"
        emit("progress", message="Segmenting labeled audio")
        corpus_segmenter.process_folder(str(source_path), str(target), float(payload.get("max_length", 5)), str(report))
    emit("progress", message="Converting labels to transcriptions.csv")
    liteconvert.auto_config(str(target))
    liteconvert.lab2csv(str(target), str(target / "auto_lang_config.json"))
    if payload.get("estimate_notes"):
        some = ROOT / "SOME"
        model = ROOT / "DiffSinger/checkpoints/SOME/0119_continuous256_5spk/model_ckpt_steps_100000_simplified.ckpt"
        for transcript in target.rglob("transcriptions.csv"):
            dataset = transcript.parent
            run_child([sys.executable, str(some / "batch_infer.py"), "--model", str(model), "--dataset", str(dataset), "--overwrite"], ROOT)
            if payload.get("make_ds"):
                from difftrainer_app.modules.csv2ds import csv2ds
                csv2ds(str(transcript), str(dataset / "wavs"), .01, 512, 44100, str(payload.get("pitch_extractor", "rmvpe")), str(ROOT / "DiffSinger"))
    return {"output_folder": str(target)}


def configure(payload: dict[str, Any]) -> dict[str, Any]:
    (data_folder, output_folder, model_type) = required(payload, "data_folder", "output_folder", "model_type")
    import yaml
    base = ROOT / "DiffSinger/configs" / ("acoustic.yaml" if model_type == "acoustic" else "variance.yaml")
    if not base.is_file(): raise ValueError(f"Base configuration not found: {base}")
    speaker_data = speakers({"data_folder": data_folder})["speakers"]
    if not speaker_data: raise ValueError("No speaker folders with transcriptions.csv were found")
    datasets = []
    for spk in speaker_data:
        with open(Path(spk["folder"]) / "transcriptions.csv", newline="", encoding="utf8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 3: raise ValueError(f"{spk['name']} needs at least three transcript rows")
        datasets.append({"raw_data_dir": spk["folder"], "speaker": spk["name"], "spk_id": int(spk["id"]), "language": spk["language"], "test_prefixes": [r["name"] for r in rows[:3]]})
    with open(base, encoding="utf8") as f: config = yaml.safe_load(f)
    config["datasets"] = datasets
    config["num_spk"] = max(d["spk_id"] for d in datasets) + 1
    langloader_path = ROOT / "DiffSinger/dictionaries/langloader.yaml"
    if langloader_path.is_file():
        with open(langloader_path, encoding="utf8") as f: lang = yaml.safe_load(f) or {}
        config["dictionaries"] = lang.get("dictionaries", {})
        config["num_lang"] = len(config["dictionaries"])
        config["use_lang_id"] = len(config["dictionaries"]) > 1
        config["extra_phonemes"] = list(lang.get("extra_phonemes", []))
        merge_path = ROOT / "DiffSinger/dictionaries" / str(lang.get("merge_list", ""))
        if merge_path.is_file():
            with open(merge_path, encoding="utf8") as f: merge = yaml.safe_load(f) or {}
            config["merged_phoneme_groups"] = merge.get("merged_phoneme_groups", [])
    config["use_spk_id"] = config["num_spk"] > 1
    if model_type == "acoustic": config["use_mix_ln"] = config["num_spk"] > 1
    config["binary_data_dir"] = str(Path(output_folder) / "binary")
    name = str(payload.get("config_name", f"{model_type}_native")).strip()
    if not name or Path(name).name != name: raise ValueError("config_name must be a filename without a path")
    destination = ROOT / "DiffSinger/configs" / f"{name}.yaml"
    with open(destination, "w", encoding="utf8") as f: yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return {"config": str(destination), "speakers": datasets}


def train(payload: dict[str, Any], binarize: bool = False) -> dict[str, Any]:
    config, experiment = required(payload, "config", "experiment")
    ds = ROOT / "DiffSinger"
    if not ds.is_dir(): raise ValueError(f"DiffSinger checkout not found: {ds}")
    script = "scripts/binarize.py" if binarize else "scripts/train.py"
    args = [sys.executable, script, "--config", config]
    if not binarize: args += ["--exp_name", experiment, "--reset"]
    emit("progress", message=f"Starting {('binarization' if binarize else 'training')}")
    run_child(args, ds)
    return {"completed": True}


def export(payload: dict[str, Any], advanced: bool) -> dict[str, Any]:
    name, destination, acoustic = required(payload, "singer_name", "output_folder", "acoustic_checkpoint")
    if advanced:
        duration = str(payload.get("duration_checkpoint", "")) or None
        variance = str(payload.get("variance_checkpoint", "")) or None
        pitch = str(payload.get("pitch_checkpoint", "")) or None
        from difftrainer_app.modules.advexport import run_adv_config
        class Value:
            def __init__(self, value: Any): self.value = value
            def get(self) -> Any: return self.value
        run_adv_config(Value(name), destination, acoustic, duration, variance, pitch, payload.get("vocoder"), Value(bool(payload.get("make_dictionary"))))
    else:
        variance = str(payload.get("variance_checkpoint", "")).strip()
        if not variance: raise ValueError("Basic export requires a variance checkpoint folder")
        from difftrainer_app.modules.basicexport import run_OU_config
        class Value:
            def __init__(self, value: Any): self.value = value
            def get(self) -> Any: return self.value
        run_OU_config(Value(name), destination, acoustic, variance, payload.get("vocoder"), Value(bool(payload.get("make_dictionary"))))
    return {"output_folder": str(Path(destination) / "".join(name.split()))}


def inference(payload: dict[str, Any]) -> dict[str, Any]:
    input_ds, experiment, speaker, language, output = required(payload, "input_ds", "acoustic_checkpoint", "speaker", "language", "output_folder")
    run_child([sys.executable, "scripts/infer.py", "acoustic", input_ds, "--exp", experiment, "--spk", speaker,
               "--lang", language, "--out", output, "--title", str(payload.get("title", "native_inference"))], ROOT / "DiffSinger")
    return {"output_folder": output}


def dictionary(payload: dict[str, Any]) -> dict[str, Any]:
    path, = required(payload, "dictionary")
    if payload.get("action") == "read":
        return {"content": Path(path).read_text(encoding="utf8")}
    content = str(payload.get("content", ""))
    Path(path).write_text(content, encoding="utf8")
    return {"dictionary": path}


OPS = {"status": status, "speakers": speakers, "prepare": prepare, "configure": configure,
       "binarize": lambda p: train(p, True), "train": train,
       "export_basic": lambda p: export(p, False), "export_advanced": lambda p: export(p, True),
       "inference": inference, "dictionary": dictionary}


def terminate(_sig: int, _frame: Any) -> None:
    if ACTIVE and ACTIVE.poll() is None:
        try: os.killpg(ACTIVE.pid, signal.SIGTERM)
        except OSError: ACTIVE.terminate()
    raise SystemExit(130)


signal.signal(signal.SIGTERM, terminate)
for line in sys.stdin:
    try:
        request = json.loads(line)
        operation = str(request.get("operation", ""))
        if operation not in OPS: raise ValueError(f"Unknown operation: {operation}")
        emit("progress", message=f"Running {operation}")
        result = OPS[operation](request.get("payload", {}))
        emit("result", result=result)
    except Exception as exc:
        emit("error", message=str(exc), detail=traceback.format_exc())
