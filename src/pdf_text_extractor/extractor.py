from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re

from pypdf import PdfReader

from .ocr import OCRModel, OCRPageResult, RapidOCREngine, default_model_dir
from .render import render_pdf_page


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


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    source: str
    text: str
    confidence: float | None = None


def extract_pdf(pdf_path: Path, options: ExtractOptions | None = None) -> list[ExtractedPage]:
    options = options or ExtractOptions()
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")
    if not path.is_file():
        raise ValueError(f"不是文件: {path}")

    reader = PdfReader(str(path))
    ocr_engine: RapidOCREngine | None = None
    pages: list[ExtractedPage] = []

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

        if ocr_engine is None:
            ocr_engine = RapidOCREngine(model=options.ocr_model, model_dir=default_model_dir())

        image = render_pdf_page(path, page_index, options.dpi)
        ocr_result = ocr_engine.recognize(image)
        pages.append(_page_from_ocr(page_index + 1, ocr_result))

    return pages


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
