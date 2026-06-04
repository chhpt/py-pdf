from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int) -> Image.Image:
    """Render one PDF page to an RGB Pillow image."""
    return next(render_pdf_pages(pdf_path, [page_index], dpi))[1]


def render_pdf_pages(
    pdf_path: Path,
    page_indexes: Iterable[int],
    dpi: int,
) -> Iterator[tuple[int, Image.Image]]:
    """Render PDF pages to RGB Pillow images while keeping the PDF open once."""
    import pymupdf

    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    with pymupdf.open(pdf_path) as document:
        for page_index in page_indexes:
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            yield page_index, image
