from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from .extractor import ExtractOptions, OCRMode, extract_pdf, pages_to_text
from .ocr import OCRModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-extract",
        description="从原生或扫描版 PDF 中提取中文文字。",
    )
    parser.add_argument("pdf", type=Path, help="输入 PDF 文件路径。")
    parser.add_argument("-o", "--output", type=Path, help="输出 UTF-8 文本文件路径。")
    parser.add_argument("--json", dest="json_output", type=Path, help="可选 JSON 明细输出路径。")
    parser.add_argument(
        "--ocr",
        choices=[mode.value for mode in OCRMode],
        default=OCRMode.AUTO.value,
        help="OCR 模式：auto 自动判断，always 强制 OCR，never 禁用 OCR。",
    )
    parser.add_argument(
        "--ocr-model",
        choices=[model.value for model in OCRModel],
        default=OCRModel.AUTO.value,
        help="OCR 模型：auto 默认使用 rapidocr-v4-mobile，也可显式指定 rapidocr-v4-mobile。",
    )
    parser.add_argument("--dpi", type=int, default=200, help="OCR 前 PDF 页面渲染 DPI，默认 200。")
    parser.add_argument(
        "--min-native-chars",
        type=int,
        default=20,
        help="auto 模式下跳过 OCR 所需的最少原生非空白字符数，默认 20。",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=None,
        help="并行处理 OCR 页面的 worker 数；默认按 CPU 核心数和 OCR 页数自动调度，可用 PDF_EXTRACT_OCR_WORKERS 覆盖。",
    )
    parser.add_argument(
        "--ocr-threads",
        type=int,
        default=None,
        help="每个 OCR worker 内部 ONNXRuntime 线程数；默认 1，可用 PDF_EXTRACT_OCR_THREADS 覆盖。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dpi <= 0:
        parser.error("--dpi must be greater than 0")
    if args.min_native_chars < 0:
        parser.error("--min-native-chars must be 0 or greater")
    ocr_workers = _positive_int_option(
        parser,
        "--ocr-workers",
        args.ocr_workers,
        "PDF_EXTRACT_OCR_WORKERS",
    )
    ocr_threads = _positive_int_option(
        parser,
        "--ocr-threads",
        args.ocr_threads,
        "PDF_EXTRACT_OCR_THREADS",
    )

    options = ExtractOptions(
        ocr=OCRMode(args.ocr),
        ocr_model=OCRModel(args.ocr_model),
        dpi=args.dpi,
        min_native_chars=args.min_native_chars,
        ocr_workers=ocr_workers,
        ocr_threads=ocr_threads,
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


def _positive_int_option(
    parser: argparse.ArgumentParser,
    option_name: str,
    cli_value: int | None,
    env_name: str,
    default: int | None = None,
) -> int | None:
    value = cli_value
    if value is None:
        env_value = os.environ.get(env_name)
        if env_value:
            try:
                value = int(env_value)
            except ValueError:
                parser.error(f"{env_name} must be a positive integer")

    if value is None:
        return default
    if value < 1:
        parser.error(f"{option_name} must be greater than 0")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
