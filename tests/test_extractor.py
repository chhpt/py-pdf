from __future__ import annotations

from pathlib import Path
import time

import pytest
from pypdf import PdfWriter

from pdf_text_extractor.extractor import (
    ExtractOptions,
    OCRMode,
    _resolve_ocr_threads,
    _resolve_ocr_workers,
    extract_pdf,
    pages_to_text,
)
from pdf_text_extractor.ocr import OCRModel, OCRPageResult


def test_native_text_pdf_does_not_trigger_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "native.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    class FakePage:
        def extract_text(self) -> str:
            return "这是一个中文测试文档，包含足够的原生文字。"

    class FakeReader:
        pages = [FakePage()]

    def fail_ocr(*args: object, **kwargs: object) -> None:
        raise AssertionError("OCR should not be initialized for native text")

    monkeypatch.setattr("pdf_text_extractor.extractor.PdfReader", lambda *args: FakeReader())
    monkeypatch.setattr("pdf_text_extractor.extractor.RapidOCREngine", fail_ocr)

    pages = extract_pdf(pdf_path, ExtractOptions(ocr=OCRMode.AUTO))

    assert len(pages) == 1
    assert pages[0].source == "native"
    assert "这是一个中文测试文档" in pages[0].text
    assert pages_to_text(pages) == pages[0].text


def test_blank_page_triggers_mocked_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    _write_blank_pdf(pdf_path)

    class FakeOCR:
        def __init__(
            self,
            model: OCRModel = OCRModel.AUTO,
            ocr_threads: int = 1,
        ) -> None:
            self.ocr_threads = ocr_threads

        def recognize(self, image: object) -> OCRPageResult:
            return OCRPageResult(text="第一页扫描文字", confidence=0.91, lines=())

    monkeypatch.setattr("pdf_text_extractor.extractor.RapidOCREngine", FakeOCR)
    monkeypatch.setattr(
        "pdf_text_extractor.extractor.render_pdf_pages",
        lambda path, page_indexes, dpi: ((page_index, object()) for page_index in page_indexes),
    )

    pages = extract_pdf(pdf_path, ExtractOptions(ocr=OCRMode.AUTO))

    assert pages[0].page_number == 1
    assert pages[0].source == "ocr"
    assert pages[0].text == "第一页扫描文字"
    assert pages[0].confidence == pytest.approx(0.91)


def test_extract_options_passes_ocr_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    _write_blank_pdf(pdf_path)
    seen_models: list[OCRModel] = []

    class FakeOCR:
        def __init__(
            self,
            model: OCRModel,
            ocr_threads: int = 1,
        ) -> None:
            seen_models.append(model)

        def recognize(self, image: object) -> OCRPageResult:
            return OCRPageResult(text="v4 识别结果", confidence=0.88, lines=())

    monkeypatch.setattr("pdf_text_extractor.extractor.RapidOCREngine", FakeOCR)
    monkeypatch.setattr(
        "pdf_text_extractor.extractor.render_pdf_pages",
        lambda path, page_indexes, dpi: ((page_index, object()) for page_index in page_indexes),
    )

    pages = extract_pdf(
        pdf_path,
        ExtractOptions(ocr=OCRMode.ALWAYS, ocr_model=OCRModel.RAPIDOCR_V4_MOBILE),
    )

    assert seen_models == [OCRModel.RAPIDOCR_V4_MOBILE]
    assert pages[0].text == "v4 识别结果"


def test_ocr_never_leaves_blank_page_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    _write_blank_pdf(pdf_path)

    def fail_render(*args: object, **kwargs: object) -> None:
        raise AssertionError("render should not run when OCR is disabled")

    monkeypatch.setattr("pdf_text_extractor.extractor.render_pdf_pages", fail_render)

    pages = extract_pdf(pdf_path, ExtractOptions(ocr=OCRMode.NEVER))

    assert pages[0].source == "none"
    assert pages[0].text == ""
    assert pages[0].confidence is None


def test_multi_page_ocr_preserves_page_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    _write_blank_pdf(pdf_path, page_count=3)

    class FakeOCR:
        def __init__(
            self,
            model: OCRModel = OCRModel.AUTO,
            ocr_threads: int = 1,
        ) -> None:
            pass

        def recognize(self, image: int) -> OCRPageResult:
            if image == 0:
                time.sleep(0.02)
            return OCRPageResult(text=f"第{image + 1}页", confidence=0.9, lines=())

    monkeypatch.setattr("pdf_text_extractor.extractor.RapidOCREngine", FakeOCR)
    monkeypatch.setattr(
        "pdf_text_extractor.extractor.render_pdf_pages",
        lambda path, page_indexes, dpi: ((page_index, page_index) for page_index in page_indexes),
    )

    pages = extract_pdf(pdf_path, ExtractOptions(ocr=OCRMode.ALWAYS, ocr_workers=3))

    assert [page.page_number for page in pages] == [1, 2, 3]
    assert [page.text for page in pages] == ["第1页", "第2页", "第3页"]


def test_ocr_threads_are_passed_to_worker_engines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    _write_blank_pdf(pdf_path, page_count=2)
    seen_threads: list[int] = []

    class FakeOCR:
        def __init__(
            self,
            model: OCRModel = OCRModel.AUTO,
            ocr_threads: int = 1,
        ) -> None:
            seen_threads.append(ocr_threads)

        def recognize(self, image: object) -> OCRPageResult:
            return OCRPageResult(text="识别结果", confidence=0.9, lines=())

    monkeypatch.setattr("pdf_text_extractor.extractor.RapidOCREngine", FakeOCR)
    monkeypatch.setattr(
        "pdf_text_extractor.extractor.render_pdf_pages",
        lambda path, page_indexes, dpi: ((page_index, object()) for page_index in page_indexes),
    )

    pages = extract_pdf(
        pdf_path,
        ExtractOptions(ocr=OCRMode.ALWAYS, ocr_workers=1, ocr_threads=2),
    )

    assert len(pages) == 2
    assert seen_threads == [2]


def test_missing_pdf_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF 文件不存在"):
        extract_pdf(missing)


def test_invalid_ocr_worker_option_raises(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    _write_blank_pdf(pdf_path)

    with pytest.raises(ValueError, match="ocr_workers must be greater than 0"):
        extract_pdf(pdf_path, ExtractOptions(ocr_workers=0))


def test_default_ocr_workers_uses_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pdf_text_extractor.extractor.os.cpu_count", lambda: 8)

    assert _resolve_ocr_workers(None, ocr_page_count=2) == 8


def test_default_ocr_threads_uses_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pdf_text_extractor.extractor.os.cpu_count", lambda: 8)

    assert _resolve_ocr_threads(None) == 1


def _write_blank_pdf(path: Path, page_count: int = 1) -> None:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=300, height=200)
    with path.open("wb") as output:
        writer.write(output)
