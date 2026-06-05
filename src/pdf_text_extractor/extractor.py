from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
import math
import os
from pathlib import Path
import re

from pypdf import PdfReader

from .ocr import OCRModel, OCRPageResult, RapidOCREngine
from .render import render_pdf_pages


class OCRMode(str, Enum):
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


@dataclass(frozen=True)
class ExtractOptions:
    ocr: OCRMode = OCRMode.AUTO
    ocr_model: OCRModel = OCRModel.AUTO
    dpi: int = 200
    min_native_chars: int = 20
    ocr_workers: int | None = None
    ocr_threads: int = 1


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    source: str
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class _OCRChunkTask:
    path: Path
    page_indexes: tuple[int, ...]
    dpi: int
    model: OCRModel
    threads: int


_PROCESS_OCR_ENGINE: RapidOCREngine | None = None
_PROCESS_OCR_ENGINE_CONFIG: tuple[OCRModel, int, int] | None = None


def extract_pdf(pdf_path: Path, options: ExtractOptions | None = None) -> list[ExtractedPage]:
    options = options or ExtractOptions()
    _validate_options(options)
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")
    if not path.is_file():
        raise ValueError(f"不是文件: {path}")

    reader = PdfReader(str(path))
    pages: list[ExtractedPage | None] = []
    ocr_page_indexes: list[int] = []

    for page_index, page in enumerate(reader.pages):
        native_text = page.extract_text() or ""
        if _should_use_native_text(native_text, options):
            pages.append(
                ExtractedPage(
                    page_number=page_index + 1,
                    source="native",
                    text=_normalize_text(native_text),
                    confidence=None,
                )
            )
            continue

        if options.ocr == OCRMode.NEVER:
            pages.append(
                ExtractedPage(
                    page_number=page_index + 1,
                    source="none",
                    text=_normalize_text(native_text),
                    confidence=None,
                )
            )
            continue

        pages.append(None)
        ocr_page_indexes.append(page_index)

    if ocr_page_indexes:
        workers = _resolve_ocr_workers(options.ocr_workers, len(ocr_page_indexes))
        threads = _resolve_ocr_threads(options.ocr_threads)
        for page_index, page in _extract_ocr_pages(path, ocr_page_indexes, options, workers, threads):
            pages[page_index] = page

    extracted_pages: list[ExtractedPage] = []
    for page in pages:
        if page is None:
            raise RuntimeError("OCR page extraction did not produce a result")
        extracted_pages.append(page)
    return extracted_pages


def pages_to_text(pages: list[ExtractedPage]) -> str:
    return "\n\n".join(page.text for page in pages)


def _should_use_native_text(text: str, options: ExtractOptions) -> bool:
    if options.ocr == OCRMode.ALWAYS:
        return False
    if options.ocr == OCRMode.NEVER:
        return bool(_compact_text(text))
    return len(_compact_text(text)) >= options.min_native_chars


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def _page_from_ocr(page_number: int, result: OCRPageResult) -> ExtractedPage:
    return ExtractedPage(
        page_number=page_number,
        source="ocr",
        text=result.text,
        confidence=result.confidence,
    )


def _validate_options(options: ExtractOptions) -> None:
    if options.ocr_workers is not None and options.ocr_workers < 1:
        raise ValueError("ocr_workers must be greater than 0")
    if options.ocr_threads is not None and options.ocr_threads < 1:
        raise ValueError("ocr_threads must be greater than 0")


def _extract_ocr_pages(
    path: Path,
    page_indexes: list[int],
    options: ExtractOptions,
    workers: int,
    threads: int,
) -> list[tuple[int, ExtractedPage]]:
    if workers == 1:
        engine = RapidOCREngine(
            model=options.ocr_model,
            ocr_threads=threads,
        )
        return [
            (page_index, _page_from_ocr(page_index + 1, engine.recognize(image)))
            for page_index, image in render_pdf_pages(path, page_indexes, options.dpi)
        ]

    results: list[tuple[int, ExtractedPage]] = []
    tasks = [
        _OCRChunkTask(
            path=path,
            page_indexes=chunk,
            dpi=options.dpi,
            model=options.ocr_model,
            threads=threads,
        )
        for chunk in _build_ocr_page_chunks(page_indexes, workers)
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk_results in executor.map(_process_ocr_page_chunk, tasks):
            results.extend(chunk_results)

    return results


def _resolve_ocr_workers(configured_workers: int | None, ocr_page_count: int) -> int:
    if ocr_page_count <= 0:
        return 1
    if configured_workers is not None:
        return min(configured_workers, ocr_page_count)
    # Parallelize across OCR pages by default, but do not create idle workers.
    return min(max(1, os.cpu_count() or 1), ocr_page_count)


def _resolve_ocr_threads(configured_threads: int | None) -> int:
    if configured_threads is not None:
        return configured_threads
    # Avoid CPU oversubscription when multiple page workers run at the same time.
    return 1


def _build_ocr_page_chunks(page_indexes: list[int], workers: int) -> tuple[tuple[int, ...], ...]:
    if not page_indexes:
        return ()
    if workers <= 1:
        return (tuple(page_indexes),)

    chunk_count = min(len(page_indexes), workers * 2)
    chunk_size = max(1, math.ceil(len(page_indexes) / chunk_count))
    return tuple(
        tuple(page_indexes[index : index + chunk_size])
        for index in range(0, len(page_indexes), chunk_size)
    )


def _process_ocr_page_chunk(task: _OCRChunkTask) -> list[tuple[int, ExtractedPage]]:
    engine = _get_process_ocr_engine(task.model, task.threads)
    return [
        (page_index, _page_from_ocr(page_index + 1, engine.recognize(image)))
        for page_index, image in render_pdf_pages(task.path, task.page_indexes, task.dpi)
    ]


def _get_process_ocr_engine(model: OCRModel, threads: int) -> RapidOCREngine:
    global _PROCESS_OCR_ENGINE, _PROCESS_OCR_ENGINE_CONFIG

    config = (model, threads, os.getpid())
    if _PROCESS_OCR_ENGINE is None or _PROCESS_OCR_ENGINE_CONFIG != config:
        _PROCESS_OCR_ENGINE = RapidOCREngine(
            model=model,
            ocr_threads=threads,
        )
        _PROCESS_OCR_ENGINE_CONFIG = config
    return _PROCESS_OCR_ENGINE
