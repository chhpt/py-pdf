from __future__ import annotations

from pathlib import Path

import pytest

from pdf_text_extractor import cli
from pdf_text_extractor.extractor import ExtractOptions, ExtractedPage
from pdf_text_extractor.ocr import OCRModel


def test_cli_writes_text_and_json(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    text_path = tmp_path / "out.txt"
    json_path = tmp_path / "out.json"

    monkeypatch.setattr(
        cli,
        "extract_pdf",
        lambda *args, **kwargs: [
            ExtractedPage(page_number=1, source="native", text="中文内容", confidence=None)
        ],
    )

    exit_code = cli.main([str(pdf_path), "-o", str(text_path), "--json", str(json_path)])

    assert exit_code == 0
    assert text_path.read_text(encoding="utf-8") == "中文内容"
    assert '"text": "中文内容"' in json_path.read_text(encoding="utf-8")


def test_cli_returns_error_for_missing_file(capsys) -> None:
    exit_code = cli.main(["/path/does/not/exist.pdf"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PDF 文件不存在" in captured.err


def test_cli_accepts_ocr_model(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    seen_options: list[ExtractOptions] = []

    def fake_extract_pdf(path: Path, options: ExtractOptions) -> list[ExtractedPage]:
        seen_options.append(options)
        return [ExtractedPage(page_number=1, source="ocr", text="识别结果", confidence=0.9)]

    monkeypatch.setattr(cli, "extract_pdf", fake_extract_pdf)

    exit_code = cli.main([str(pdf_path), "--ocr-model", "rapidocr-v4-mobile"])

    assert exit_code == 0
    assert seen_options[0].ocr_model == OCRModel.RAPIDOCR_V4_MOBILE


def test_cli_accepts_ocr_worker_and_thread_options(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    seen_options: list[ExtractOptions] = []

    def fake_extract_pdf(path: Path, options: ExtractOptions) -> list[ExtractedPage]:
        seen_options.append(options)
        return [ExtractedPage(page_number=1, source="ocr", text="识别结果", confidence=0.9)]

    monkeypatch.setattr(cli, "extract_pdf", fake_extract_pdf)

    exit_code = cli.main([str(pdf_path), "--ocr-workers", "3", "--ocr-threads", "2"])

    assert exit_code == 0
    assert seen_options[0].ocr_workers == 3
    assert seen_options[0].ocr_threads == 2


def test_cli_reads_ocr_worker_and_thread_env(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    seen_options: list[ExtractOptions] = []

    def fake_extract_pdf(path: Path, options: ExtractOptions) -> list[ExtractedPage]:
        seen_options.append(options)
        return [ExtractedPage(page_number=1, source="ocr", text="识别结果", confidence=0.9)]

    monkeypatch.setenv("PDF_EXTRACT_OCR_WORKERS", "4")
    monkeypatch.setenv("PDF_EXTRACT_OCR_THREADS", "2")
    monkeypatch.setattr(cli, "extract_pdf", fake_extract_pdf)

    exit_code = cli.main([str(pdf_path), "--ocr-workers", "1"])

    assert exit_code == 0
    assert seen_options[0].ocr_workers == 1
    assert seen_options[0].ocr_threads == 2


def test_cli_rejects_invalid_ocr_workers(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["input.pdf", "--ocr-workers", "0"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "--ocr-workers must be greater than 0" in captured.err
