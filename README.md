# PDF 中文文字提取

一个用于从 PDF 提取中文文字的 Python CLI 项目。它会先读取 PDF 内置文字；如果页面看起来像扫描件或文字太少，会把页面渲染成图片并使用 RapidOCR + ONNXRuntime 做中文 OCR。

## 功能

- 支持 macOS 和 Linux/CentOS。
- 支持原生文字 PDF 和扫描版 PDF。
- 默认输出 UTF-8 文本，可选 JSON 明细。
- OCR 默认使用 RapidOCR wheel 内置的 PP-OCRv4 mobile 模型，兼容 CentOS 7 等老服务器环境。
- 不依赖系统 Tesseract、Poppler 或在线模型下载。
- `numpy` 固定为 `1.26.4`，Linux 使用 `opencv-python 4.11.0.86` 和 `onnxruntime 1.16.3`，macOS 使用 `opencv-python 4.10.0.84`。
- CentOS 7 免编译安装以 PP-OCRv4 mobile 为兼容基线。

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

## OCR 模型

RapidOCR wheel 已内置 PP-OCRv4 mobile 模型，通常不需要额外下载。下面命令只用于检查或解包 wheel 内置模型：

```bash
uv run python scripts/download_assets.py
```

## CLI 参数

```bash
uv run pdf-extract input.pdf -o output.txt --json output.json --dpi 200 --ocr auto
```

基础输出参数：

- `pdf`：必填，输入 PDF 文件路径。支持原生文字 PDF 和扫描图片型 PDF。
- `-o, --output`：可选，输出 UTF-8 纯文本文件路径；不传时直接打印到终端。
- `--json`：可选，输出逐页 JSON 明细，包含页码、来源、文字和 OCR 平均置信度。

OCR 触发策略：

- `--ocr auto`：默认模式。每页先尝试读取 PDF 内置文字；如果非空白字符数少于 `--min-native-chars`，再渲染页面并 OCR。
- `--ocr always`：所有页面都走 OCR。适合 PDF 内置文字乱码、顺序错误或需要统一扫描件识别结果的情况。
- `--ocr never`：完全禁用 OCR，只提取 PDF 内置文字。适合确认 PDF 是原生文本、并希望最快速度处理的情况。
- `--min-native-chars 20`：`auto` 模式下跳过 OCR 的最小原生非空白字符数。调大后会有更多页面进入 OCR；调小后会更倾向直接使用 PDF 内置文字。

OCR 模型参数：

- `--ocr-model auto`：默认模型模式，直接使用 RapidOCR wheel 内置 PP-OCRv4 mobile。
- `--ocr-model rapidocr-v4-mobile`：显式使用 PP-OCRv4 mobile。兼容性最好，适合 CentOS 7、老服务器或不想依赖新版 ONNXRuntime 的部署。

渲染和并发参数：

- `--dpi 200`：OCR 前把 PDF 页面渲染成图片的分辨率。默认 `200`；模糊扫描件或小字号可以尝试 `300`，但会显著增加 CPU、内存和耗时。
- `--ocr-workers N`：同时 OCR 的页面 worker 数。默认使用 CPU 核心数，至少为 `1`；多页扫描 PDF 通常提高该值会更快，但也会增加内存占用。
- `--ocr-threads N`：每个 OCR worker 内部 ONNXRuntime 使用的线程数。默认 `1`；多页并发时建议保持 `1`，避免 `ocr-workers * ocr-threads` 过大导致 CPU 线程超卖。

也可以通过环境变量设置服务器默认值，CLI 参数优先级更高：

```bash
PDF_EXTRACT_OCR_WORKERS=2 PDF_EXTRACT_OCR_THREADS=1 uv run pdf-extract input.pdf
```

服务器部署建议：

```bash
# 单文件低延迟：提高同一 PDF 内的 OCR 页并发
uv run pdf-extract input.pdf -o output.txt --ocr-workers 4 --ocr-threads 1

# 多请求并发服务：限制每个请求占用，避免整机 CPU/内存被打满
PDF_EXTRACT_OCR_WORKERS=2 PDF_EXTRACT_OCR_THREADS=1 uv run pdf-extract input.pdf -o output.txt
```

JSON 输出字段：

- `page_number`：页码，从 1 开始。
- `source`：`native`、`ocr` 或 `none`。
- `text`：该页文字。
- `confidence`：OCR 平均置信度；原生文字为 `null`。

## 测试

```bash
uv run pytest
```
