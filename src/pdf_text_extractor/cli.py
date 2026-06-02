from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .extractor import ExtractOptions, OCRMode, extract_pdf, pages_to_text
from .ocr import OCRModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-extract",
        description="Extract Chinese text from native or scanned PDF files.",
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path.")
    parser.add_argument("-o", "--output", type=Path, help="Output UTF-8 text path.")
    parser.add_argument("--json", dest="json_output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--ocr",
        choices=[mode.value for mode in OCRMode],
        default=OCRMode.AUTO.value,
        help="OCR mode: auto, always, or never.",
    )
    parser.add_argument(
        "--ocr-model",
        choices=[model.value for model in OCRModel],
        default=OCRModel.AUTO.value,
        help="OCR model: auto, rapidocr-v4-mobile, or rapidocr-v5-server.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for OCR page rendering.")
    parser.add_argument(
        "--min-native-chars",
        type=int,
        default=20,
        help="Minimum non-space native text chars before skipping OCR in auto mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dpi <= 0:
        parser.error("--dpi must be greater than 0")
    if args.min_native_chars < 0:
        parser.error("--min-native-chars must be 0 or greater")

    options = ExtractOptions(
        ocr=OCRMode(args.ocr),
        ocr_model=OCRModel(args.ocr_model),
        dpi=args.dpi,
        min_native_chars=args.min_native_chars,
    )

    try:
        pages = extract_pdf(args.pdf, options)
        text = pages_to_text(pages)
        if args.output:
            _write_text(args.output, text)
        else:
            sys.stdout.write(text)
            if text and not text.endswith("\n"):
                sys.stdout.write("\n")

        if args.json_output:
            _write_json(args.json_output, pages)
    except Exception as exc:
        sys.stderr.write(f"pdf-extract: {exc}\n")
        return 1

    return 0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, pages: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(page) for page in pages]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
