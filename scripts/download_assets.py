#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
RAPIDOCR_VERSION = "3.8.1"
MODEL_DIR = VENDOR / "models" / "rapidocr-v5-server"
V5_SERVER_ASSETS = {
    "ch_PP-OCRv5_det_server.onnx": (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_server.onnx",
        "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad",
    ),
    "ch_PP-OCRv5_rec_server.onnx": (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx",
        "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
    ),
    "ppocrv5_dict.txt": (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_server/ppocrv5_dict.txt",
        None,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download or inspect RapidOCR bundled models.")
    parser.add_argument(
        "--legacy-v4",
        action="store_true",
        help="Check/extract RapidOCR wheel-bundled PP-OCRv4 mobile models instead.",
    )
    args = parser.parse_args(argv)

    if not args.legacy_v4:
        download_v5_server_models()
        return 0

    rapidocr_wheel = find_or_download_rapidocr_wheel()
    extract_rapidocr_models(rapidocr_wheel)
    return 0


def download_v5_server_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (url, sha256) in V5_SERVER_ASSETS.items():
        destination = MODEL_DIR / filename
        if destination.exists() and (sha256 is None or file_sha256(destination) == sha256):
            print(f"Already exists: {destination}")
            continue
        print(f"Downloading {url}")
        download_file(url, destination)
        if sha256 is not None:
            actual = file_sha256(destination)
            if actual != sha256:
                destination.unlink(missing_ok=True)
                raise RuntimeError(f"SHA256 mismatch for {filename}: expected {sha256}, got {actual}")
    print(f"Downloaded PP-OCRv5 server models to {MODEL_DIR}")


def download_file(url: str, destination: Path) -> None:
    from urllib.request import urlopen

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
