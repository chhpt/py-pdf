from __future__ import annotations

from pathlib import Path

import pytest

from pdf_text_extractor.ocr import (
    OCRModel,
    OCRModelError,
    _create_rapidocr_engine,
    _extract_lines,
    _model_params,
    _v4_model_params,
)


class FakeRapidResult:
    boxes = [
        [(20, 30), (30, 30), (30, 40), (20, 40)],
        [(5, 10), (15, 10), (15, 20), (5, 20)],
    ]
    txts = ("第二行", "第一行")
    scores = (0.8, 0.9)


def test_extract_lines_normalizes_rapidocr_result() -> None:
    lines = _extract_lines(FakeRapidResult())

    assert len(lines) == 2
    assert lines[0].text == "第二行"
    assert lines[0].confidence == 0.8
    assert lines[0].box[0] == (20.0, 30.0)


def test_model_params_prefers_v5_server_files(tmp_path: Path) -> None:
    for filename in [
        "ch_PP-OCRv5_det_server.onnx",
        "ch_PP-OCRv5_rec_server.onnx",
        "ppocrv5_dict.txt",
    ]:
        (tmp_path / filename).write_text("placeholder", encoding="utf-8")

    params = _model_params(tmp_path)

    assert params["Det.limit_side_len"] == 960
    assert params["Global.use_cls"] is False
    assert params["Rec.model_path"].endswith("ch_PP-OCRv5_rec_server.onnx")


def test_v4_model_params_use_rapidocr_bundled_files() -> None:
    params = _v4_model_params()

    assert params["Det.model_path"].endswith("ch_PP-OCRv4_det_infer.onnx")
    assert params["Cls.model_path"].endswith("ch_ppocr_mobile_v2.0_cls_infer.onnx")
    assert params["Rec.model_path"].endswith("ch_PP-OCRv4_rec_infer.onnx")
    assert params["Rec.rec_keys_path"].endswith("ppocr_keys_v1.txt")


def test_auto_model_falls_back_to_v4_when_v5_fails(tmp_path: Path) -> None:
    for filename in [
        "ch_PP-OCRv5_det_server.onnx",
        "ch_PP-OCRv5_rec_server.onnx",
        "ppocrv5_dict.txt",
    ]:
        (tmp_path / filename).write_text("placeholder", encoding="utf-8")

    calls: list[dict[str, object]] = []

    class FakeRapidOCR:
        def __init__(self, params: dict[str, object]) -> None:
            calls.append(params)
            if str(params["Rec.model_path"]).endswith("ch_PP-OCRv5_rec_server.onnx"):
                raise RuntimeError("unsupported model ir")

    _create_rapidocr_engine(FakeRapidOCR, OCRModel.AUTO, tmp_path, 0.7)

    assert len(calls) == 2
    assert calls[0]["Rec.model_path"].endswith("ch_PP-OCRv5_rec_server.onnx")
    assert calls[1]["Rec.model_path"].endswith("ch_PP-OCRv4_rec_infer.onnx")
    assert calls[1]["Global.text_score"] == 0.7


def test_explicit_v5_failure_mentions_centos7_hint(tmp_path: Path) -> None:
    for filename in [
        "ch_PP-OCRv5_det_server.onnx",
        "ch_PP-OCRv5_rec_server.onnx",
        "ppocrv5_dict.txt",
    ]:
        (tmp_path / filename).write_text("placeholder", encoding="utf-8")

    class FailingRapidOCR:
        def __init__(self, params: dict[str, object]) -> None:
            raise RuntimeError("unsupported model ir")

    with pytest.raises(OCRModelError, match="CentOS 7"):
        _create_rapidocr_engine(FailingRapidOCR, OCRModel.RAPIDOCR_V5_SERVER, tmp_path, 0.5)
