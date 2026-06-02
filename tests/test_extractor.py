from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from pdf_text_extractor.extractor import ExtractOptions, OCRMode, extract_pdf, pages_to_text
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
            model_dir: Path | None = None,
        ) -> None:
            self.model_dir = model_dir

        def recognize(self, image: object) -> OCRPageResult:
            return OCRPageResult(text="第一页扫描文字", confidence=0.91, lines=())

    monkeypatch.setattr("pdf_text_extractor.extractor.RapidOCREngine", FakeOCR)
    monkeypatch.setattr("pdf_text_extractor.extractor.render_pdf_page", lambda *args: object())

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
        def __init__(self, model: OCRModel, model_dir: Path | None = None) -> None:
            seen_models.append(model)

        def recognize(self, image: object) -> OCRPageResult:
            return OCRPageResult(text="v4 识别结果", confidence=0.88, lines=())

    monkeypatch.setattr("pdf_text_extractor.extractor.RapidOCREngine", FakeOCR)
    monkeypatch.setattr("pdf_text_extractor.extractor.render_pdf_page", lambda *args: object())

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

    monkeypatch.setattr("pdf_text_extractor.extractor.render_pdf_page", fail_render)

    pages = extract_pdf(pdf_path, ExtractOptions(ocr=OCRMode.NEVER))

    assert pages[0].source == "none"
    assert pages[0].text == ""
    assert pages[0].confidence is None


def test_missing_pdf_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF 文件不存在"):
        extract_pdf(missing)


def _write_blank_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=200)
    with path.open("wb") as output:
        writer.write(output)
