from __future__ import annotations

from pdf_text_extractor.ocr import (
    OCRModel,
    _create_engine_with_params,
    _create_rapidocr_engine,
    _extract_lines,
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


def test_v4_model_params_use_rapidocr_bundled_files() -> None:
    params = _v4_model_params()

    assert params["Det.model_path"].endswith("ch_PP-OCRv4_det_infer.onnx")
    assert params["Cls.model_path"].endswith("ch_ppocr_mobile_v2.0_cls_infer.onnx")
    assert params["Rec.model_path"].endswith("ch_PP-OCRv4_rec_infer.onnx")
    assert params["Rec.rec_keys_path"].endswith("ppocr_keys_v1.txt")


def test_auto_model_uses_rapidocr_bundled_v4() -> None:
    calls: list[dict[str, object]] = []

    class FakeRapidOCR:
        def __init__(self, params: dict[str, object]) -> None:
            calls.append(params)

    _create_rapidocr_engine(FakeRapidOCR, OCRModel.AUTO, 0.7)

    assert len(calls) == 1
    assert calls[0]["Rec.model_path"].endswith("ch_PP-OCRv4_rec_infer.onnx")
    assert calls[0]["Global.text_score"] == 0.7


def test_engine_params_include_onnxruntime_threads() -> None:
    calls: list[dict[str, object]] = []

    class FakeRapidOCR:
        def __init__(self, params: dict[str, object]) -> None:
            calls.append(params)

    _create_engine_with_params(FakeRapidOCR, {"Det.model_path": "det.onnx"}, 0.7, 2)

    assert calls[0]["Global.text_score"] == 0.7
    assert calls[0]["EngineConfig.onnxruntime.intra_op_num_threads"] == 2
    assert calls[0]["EngineConfig.onnxruntime.inter_op_num_threads"] == 2
