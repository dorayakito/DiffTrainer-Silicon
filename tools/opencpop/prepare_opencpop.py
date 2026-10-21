#!/usr/bin/env python3
"""Convert an authorized Opencpop `segments/` folder to DiffSinger CSV input.

The source folder is expected to contain `transcriptions.txt` and `wavs/`.
The script links the WAV folder into the local DiffSinger data directory and
writes the three acoustic-training columns expected by this checkout.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Authorized Opencpop segments directory with transcriptions.txt and wavs/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DiffSinger/data/opencpop/raw"),
        help="Local raw-data directory (default: DiffSinger/data/opencpop/raw)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing transcriptions.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    labels_path = source / "transcriptions.txt"
    wavs_path = source / "wavs"
    csv_path = output / "transcriptions.csv"

    if not labels_path.is_file():
        raise SystemExit(f"Opencpop labels not found: {labels_path}")
    if not wavs_path.is_dir():
        raise SystemExit(f"Opencpop segment WAV folder not found: {wavs_path}")
    if csv_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite to replace it: {csv_path}")

    rows: list[tuple[str, str, str]] = []
    missing_wavs: list[str] = []
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            fields = [field.strip() for field in raw_line.rstrip("\r\n").split("|")]
            if len(fields) != 7:
                raise SystemExit(
                    f"Expected 7 pipe-separated Opencpop fields at line {line_number}; "
                    f"found {len(fields)}."
                )
            name, _text, phonemes, _notes, _note_durations, phone_durations, _slurs = fields
            if name.lower() in {"name", "utterance wav name"}:
                continue
            if not name or Path(name).name != name or name in {".", ".."}:
                raise SystemExit(f"Invalid utterance name at line {line_number}: {name!r}")

            phones = phonemes.split()
            durations = phone_durations.split()
            if not phones or len(phones) != len(durations):
                raise SystemExit(
                    f"Phone/duration count mismatch for {name!r} at line {line_number}: "
                    f"{len(phones)} phones, {len(durations)} durations."
                )
            try:
                parsed_durations = [float(value) for value in durations]
            except ValueError as exc:
                raise SystemExit(f"Invalid phone duration for {name!r} at line {line_number}.") from exc
            if any(value < 0 for value in parsed_durations):
                raise SystemExit(f"Negative phone duration for {name!r} at line {line_number}.")

            if not any((wavs_path / f"{name}{extension}").is_file() for extension in (".wav", ".flac")):
                missing_wavs.append(name)
            rows.append((name, " ".join(phones), " ".join(durations)))

    if not rows:
        raise SystemExit(f"No utterance labels found in {labels_path}")
    if missing_wavs:
        sample = ", ".join(missing_wavs[:5])
        raise SystemExit(f"Missing WAV/FLAC for {len(missing_wavs)} utterances (for example: {sample})")

    output.mkdir(parents=True, exist_ok=True)
    linked_wavs = output / "wavs"
    if linked_wavs.exists() or linked_wavs.is_symlink():
        if not linked_wavs.is_symlink() or linked_wavs.resolve() != wavs_path:
            raise SystemExit(f"Refusing to replace existing WAV directory: {linked_wavs}")
    else:
        os.symlink(wavs_path, linked_wavs, target_is_directory=True)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output, delete=False
    ) as temp_stream:
        temp_path = Path(temp_stream.name)
        writer = csv.writer(temp_stream)
        writer.writerow(("name", "ph_seq", "ph_dur"))
        writer.writerows(rows)
    os.replace(temp_path, csv_path)

    print(f"Wrote {len(rows)} utterance rows to {csv_path}")
    print(f"Linked WAV files from {wavs_path} to {linked_wavs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
