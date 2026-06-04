from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import threading

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

    local = threading.local()

    def recognize(page_index: int, image: object) -> tuple[int, ExtractedPage]:
        engine = getattr(local, "engine", None)
        if engine is None:
            engine = RapidOCREngine(
                model=options.ocr_model,
                ocr_threads=threads,
            )
            local.engine = engine
        return page_index, _page_from_ocr(page_index + 1, engine.recognize(image))

    results: list[tuple[int, ExtractedPage]] = []
    pending: set[Future[tuple[int, ExtractedPage]]] = set()
    max_pending = max(1, workers * 2)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for page_index, image in render_pdf_pages(path, page_indexes, options.dpi):
            pending.add(executor.submit(recognize, page_index, image))
            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                results.extend(future.result() for future in done)

        for future in pending:
            results.append(future.result())

    return results


def _resolve_ocr_workers(configured_workers: int | None, ocr_page_count: int) -> int:
    if configured_workers is not None:
        return configured_workers
    # Parallelize across PDF pages by default; each worker keeps OCR internals single-threaded.
    return max(1, os.cpu_count() or 1)


def _resolve_ocr_threads(configured_threads: int | None) -> int:
    if configured_threads is not None:
        return configured_threads
    # Avoid CPU oversubscription when multiple page workers run at the same time.
    return 1
