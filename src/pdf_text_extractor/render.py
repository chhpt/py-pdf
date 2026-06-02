from __future__ import annotations

from pathlib import Path

from PIL import Image


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int) -> Image.Image:
    """Render one PDF page to an RGB Pillow image."""
    import pymupdf

    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    with pymupdf.open(pdf_path) as document:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
