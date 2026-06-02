from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    box: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class OCRPageResult:
    text: str
    confidence: float | None
    lines: tuple[OCRLine, ...]


class OCRModel(str, Enum):
    AUTO = "auto"
    RAPIDOCR_V4_MOBILE = "rapidocr-v4-mobile"
    RAPIDOCR_V5_SERVER = "rapidocr-v5-server"


class OCRModelError(RuntimeError):
    pass


class RapidOCREngine:
    def __init__(
        self,
        model: OCRModel = OCRModel.AUTO,
        model_dir: Path | None = None,
        text_score: float = 0.5,
    ) -> None:
        from rapidocr import RapidOCR

        self._engine = _create_rapidocr_engine(RapidOCR, model, model_dir, text_score)

    def recognize(self, image: Image.Image) -> OCRPageResult:
        result = self._engine(image)
        lines = _extract_lines(result)
        sorted_lines = tuple(sorted(lines, key=_line_sort_key))
        text = "\n".join(line.text for line in sorted_lines)
        confidence = _average_confidence(line.confidence for line in sorted_lines)
        return OCRPageResult(text=text, confidence=confidence, lines=sorted_lines)


def default_model_dir() -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "vendor" / "models" / "rapidocr-v5-server"
    return model_dir if model_dir.exists() else None


def _create_rapidocr_engine(
    rapidocr_class: type[Any],
    model: OCRModel,
    model_dir: Path | None,
    text_score: float,
) -> Any:
    if model == OCRModel.AUTO:
        try:
            return _create_engine_with_params(
                rapidocr_class,
                _v5_model_params(model_dir or default_model_dir()),
                text_score,
            )
        except Exception:
            return _create_engine_with_params(
                rapidocr_class,
                _v4_model_params(),
                text_score,
            )

    if model == OCRModel.RAPIDOCR_V4_MOBILE:
        return _create_engine_with_params(rapidocr_class, _v4_model_params(), text_score)

    if model == OCRModel.RAPIDOCR_V5_SERVER:
        try:
            return _create_engine_with_params(
                rapidocr_class,
                _v5_model_params(model_dir or default_model_dir()),
                text_score,
            )
        except Exception as exc:
            raise OCRModelError(
                "OCR model rapidocr-v5-server failed to initialize. "
                "CentOS 7 should use --ocr-model rapidocr-v4-mobile or the default auto mode."
            ) from exc

    raise ValueError(f"Unsupported OCR model: {model}")


def _create_engine_with_params(
    rapidocr_class: type[Any],
    model_params: dict[str, Any],
    text_score: float,
) -> Any:
    params: dict[str, Any] = {
        "Global.text_score": text_score,
    }
    params.update(model_params)
    return rapidocr_class(params=params)


def _model_params(model_dir: Path | None) -> dict[str, Any]:
    return _v5_model_params(model_dir)


def _v5_model_params(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None or not model_dir.exists():
        raise OCRModelError("OCR model rapidocr-v5-server files are missing.")

    paths = {
        "Det.model_path": model_dir / "ch_PP-OCRv5_det_server.onnx",
        "Rec.model_path": model_dir / "ch_PP-OCRv5_rec_server.onnx",
        "Rec.rec_keys_path": model_dir / "ppocrv5_dict.txt",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise OCRModelError(f"OCR model rapidocr-v5-server files are missing: {missing_names}")
    bundled = _v4_model_params()
    return {
        "Det.limit_side_len": 960,
        "Global.use_cls": False,
        "Cls.model_path": bundled["Cls.model_path"],
        "Rec.rec_img_shape": [3, 48, 320],
        **{key: str(path) for key, path in paths.items()},
    }


def _v4_model_params() -> dict[str, Any]:
    model_root = resources.files("rapidocr").joinpath("models")
    paths = {
        "Det.model_path": model_root.joinpath("ch_PP-OCRv4_det_infer.onnx"),
        "Cls.model_path": model_root.joinpath("ch_ppocr_mobile_v2.0_cls_infer.onnx"),
        "Rec.model_path": model_root.joinpath("ch_PP-OCRv4_rec_infer.onnx"),
        "Rec.rec_keys_path": model_root.joinpath("ppocr_keys_v1.txt"),
    }
    if not all(path.is_file() for path in paths.values()):
        raise OCRModelError("RapidOCR bundled PP-OCRv4 mobile model files are missing.")
    return {key: str(path) for key, path in paths.items()}


def _bundled_model_params() -> dict[str, Any]:
    return _v4_model_params()


def _extract_lines(result: Any) -> tuple[OCRLine, ...]:
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or texts is None or scores is None:
        return ()

    lines: list[OCRLine] = []
    for box, text, score in zip(boxes, texts, scores):
        clean_text = str(text).strip()
        if not clean_text:
            continue
        lines.append(
            OCRLine(
                text=clean_text,
                confidence=float(score),
                box=_normalize_box(box),
            )
        )
    return tuple(lines)


def _normalize_box(box: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in box)


def _line_sort_key(line: OCRLine) -> tuple[float, float]:
    if not line.box:
        return (0.0, 0.0)
    top = min(point[1] for point in line.box)
    left = min(point[0] for point in line.box)
    return (top, left)


def _average_confidence(scores: Iterable[float]) -> float | None:
    values = tuple(scores)
    if not values:
        return None
    return sum(values) / len(values)
