# PDF 中文文字提取

一个用于从 PDF 提取中文文字的 Python CLI 项目。它会先读取 PDF 内置文字；如果页面看起来像扫描件或文字太少，会把页面渲染成图片并使用 RapidOCR + ONNXRuntime 做中文 OCR。

## 功能

- 支持 macOS 和 Linux/CentOS。
- 支持原生文字 PDF 和扫描版 PDF。
- 默认输出 UTF-8 文本，可选 JSON 明细。
- OCR 模型支持 `auto`、`rapidocr-v4-mobile`、`rapidocr-v5-server`；默认 `auto` 会优先尝试本地 PP-OCRv5 server，失败后回退 RapidOCR wheel 内置的 PP-OCRv4 mobile。
- 不依赖系统 Tesseract、Poppler 或在线模型下载。
- `numpy` 固定为 `1.26.4`，Linux 使用 `opencv-python 4.11.0.86` 和 `onnxruntime 1.16.3`，macOS 使用 `opencv-python 4.10.0.84`。
- CentOS 7 免编译安装以 PP-OCRv4 mobile 为兼容基线；PP-OCRv5 server 需要支持 ONNX IR 10 的新版 ONNXRuntime，不保证在 CentOS 7 可用。

## 环境要求

- Python 3.10
- CentOS 建议使用 Python 3.10 虚拟环境。

## 在线安装和运行

```bash
cd /Users/chaihe/Desktop/py-pdf
uv sync --python 3.10
uv run pdf-extract input.pdf -o output.txt --json output.json --dpi 200 --ocr auto
```

如果只想打印到终端：

```bash
uv run pdf-extract input.pdf
```

## OCR 模型下载

`auto` 模式会优先使用 PP-OCRv5 server 的检测和识别模型，方向分类关闭，适合普通正向 PDF 文档。首次拉取 v5 模型：

```bash
uv run python scripts/download_assets.py
```

模型会下载到 `vendor/models/rapidocr-v5-server`。该目录需要随项目一起保留。

RapidOCR wheel 已内置 PP-OCRv4 mobile 模型，CentOS 7 通常不需要额外下载。下面命令只用于检查或解包 wheel 内置模型：

```bash
uv run python scripts/download_assets.py --legacy-v4
```

## CLI 参数

```bash
uv run pdf-extract input.pdf -o output.txt --json output.json --dpi 200 --ocr auto
```

- `--ocr auto`：默认模式，优先使用 PDF 内置文字，文字过少时 OCR。
- `--ocr always`：所有页面都 OCR。
- `--ocr never`：只提取 PDF 内置文字。
- `--ocr-model auto`：默认模型模式，优先 PP-OCRv5 server，失败后回退 PP-OCRv4 mobile。
- `--ocr-model rapidocr-v4-mobile`：固定使用 RapidOCR wheel 内置 PP-OCRv4 mobile，适合 CentOS 7 免编译安装路径。
- `--ocr-model rapidocr-v5-server`：固定使用 `vendor/models/rapidocr-v5-server`，如果运行时不支持会快速报错。
- `--dpi 200`：OCR 渲染分辨率，扫描件较模糊时可尝试 `300`。
- `--min-native-chars 20`：auto 模式下跳过 OCR 的最小非空白字符数。

JSON 输出字段：

- `page_number`：页码，从 1 开始。
- `source`：`native`、`ocr` 或 `none`。
- `text`：该页文字。
- `confidence`：OCR 平均置信度；原生文字为 `null`。

## 测试

```bash
uv run pytest
```
