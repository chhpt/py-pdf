#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
RAPIDOCR_VERSION = "3.8.1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check/extract RapidOCR bundled PP-OCRv4 mobile models.")
    parser.parse_args(argv)

    rapidocr_wheel = find_or_download_rapidocr_wheel()
    extract_rapidocr_models(rapidocr_wheel)
    return 0


def find_or_download_rapidocr_wheel() -> Path:
    wheel_dir = VENDOR / "_tmp"
    candidates = sorted(wheel_dir.glob(f"rapidocr-{RAPIDOCR_VERSION}-*.whl"))
    if candidates:
        return candidates[0]

    wheel_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--dest",
        str(wheel_dir),
        f"rapidocr=={RAPIDOCR_VERSION}",
    ]
    run(command)
    candidates = sorted(wheel_dir.glob(f"rapidocr-{RAPIDOCR_VERSION}-*.whl"))
    if not candidates:
        raise RuntimeError("rapidocr wheel was not downloaded")
    return candidates[0]


def extract_rapidocr_models(wheel: Path) -> None:
    destination = VENDOR / "models" / "rapidocr"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        model_names = [name for name in archive.namelist() if name.startswith("rapidocr/models/")]
        for name in model_names:
            if name.endswith("/"):
                continue
            relative = Path(name).relative_to("rapidocr/models")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    print(f"Extracted RapidOCR models to {destination}")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
