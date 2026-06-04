from __future__ import annotations

import sys
from pathlib import Path

from pdf_text_extractor.render import render_pdf_pages


def test_render_pdf_pages_opens_document_once(monkeypatch, tmp_path: Path) -> None:
    open_calls: list[Path] = []
    loaded_pages: list[int] = []

    class FakePixmap:
        width = 1
        height = 1
        samples = b"\x00\x00\x00"

    class FakePage:
        def get_pixmap(self, matrix: object, alpha: bool) -> FakePixmap:
            return FakePixmap()

    class FakeDocument:
        def __enter__(self) -> "FakeDocument":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            pass

        def load_page(self, page_index: int) -> FakePage:
            loaded_pages.append(page_index)
            return FakePage()

    class FakePyMuPDF:
        class Matrix:
            def __init__(self, x: float, y: float) -> None:
                self.x = x
                self.y = y

        @staticmethod
        def open(path: Path) -> FakeDocument:
            open_calls.append(path)
            return FakeDocument()

    monkeypatch.setitem(sys.modules, "pymupdf", FakePyMuPDF)

    pdf_path = tmp_path / "input.pdf"
    rendered = list(render_pdf_pages(pdf_path, [2, 0, 1], 200))

    assert open_calls == [pdf_path]
    assert loaded_pages == [2, 0, 1]
    assert [page_index for page_index, image in rendered] == [2, 0, 1]
    assert all(image.mode == "RGB" for page_index, image in rendered)
