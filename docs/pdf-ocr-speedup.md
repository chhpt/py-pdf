# 多页 PDF OCR 解析提速说明

本文总结本项目针对多页扫描 PDF 的核心提速改动，以及服务器部署时相关参数的推荐配置。

## 核心思路

多页扫描 PDF 的耗时主要集中在两步：把 PDF 页面渲染成图片，以及对图片执行 OCR。优化后的流程会先判断每页是否真的需要 OCR，只把扫描页或原生文字过少的页面送入 OCR，并通过页面级并发提升整体速度。

整体流程：

1. 使用 `pypdf` 逐页尝试读取 PDF 内置文字。
2. 原生文字足够的页面直接返回，跳过渲染和 OCR。
3. 需要 OCR 的页面集中进入批量渲染和并发识别流程。
4. OCR 结果按原始页码写回，保证最终输出顺序稳定。

## 主要提速点

### 1. 只对必要页面做 OCR

`--ocr auto` 模式下，每页会先读取 PDF 内置文字。如果非空白字符数达到 `--min-native-chars`，该页直接使用原生文字结果。

收益：

- 原生文字 PDF 基本不进入 OCR，速度更快。
- 混合型 PDF 只 OCR 扫描页，避免整份文件无差别 OCR。
- `--ocr never` 可用于确认是原生文本的场景，完全跳过 OCR。

### 2. 批量渲染时只打开一次 PDF

旧逻辑是每个 OCR 页面都重新打开一次 PDF，再渲染单页。多页扫描件会重复执行文件打开、解析和关闭。

现在使用 `render_pdf_pages(...)`，对一组需要 OCR 的页只打开一次 PDF，然后按页渲染图片。

收益：

- 减少重复打开 PDF 的开销。
- 对 8 页、10 页以上扫描件更明显。
- 渲染接口仍按页产出图片，不会一次性把所有页面图片塞进内存。

### 3. 按页面并发 OCR

需要 OCR 的页面会使用 `ProcessPoolExecutor` 并发处理。默认 `ocr-workers` 会按 CPU 核心数和实际 OCR 页数自动取较小值，也可以通过 CLI 参数或环境变量覆盖。

收益：

- 多页扫描 PDF 可以同时渲染和 OCR 多页。
- 单个 PDF 的等待时间通常明显下降。
- 页面结果会按页码写回，不会因为并发完成顺序不同而打乱输出。

### 4. 每个 worker 使用独立 OCR 引擎

并发 OCR 时，每个进程会持有自己的 `RapidOCREngine`，避免多个 worker 共享同一个 RapidOCR 实例。

收益：

- 避免共享 OCR 引擎可能带来的并发安全问题。
- 每个 worker 初始化一次后复用，减少同一进程内重复初始化成本。

### 5. 按页数自动拆分任务

并发路径不会按单页无限拆分任务，而是最多拆成约 `ocr-workers * 2` 个页面块。

收益：

- 避免创建太多细碎进程任务。
- 降低多页大 PDF 的内存峰值。
- 更适合服务器部署。

### 6. 默认避免 OCR 内部线程超卖

当前默认策略：

- `ocr-workers`：默认按 CPU 核心数和实际 OCR 页数自动取较小值，用于页面级并发。
- `ocr-threads`：默认 `1`，用于每个 OCR worker 内部的 ONNXRuntime 线程数。

这样做的原因是多页 PDF 的主要收益来自“多页同时处理”。如果每个页面 worker 内部再开很多 ONNXRuntime 线程，实际线程数会接近：

```text
ocr-workers * ocr-threads
```

例如 8 核机器上，如果 `ocr-workers=8` 且 `ocr-threads=8`，可能会产生大量 CPU 竞争，反而比 `ocr-threads=1` 更慢。

## 参数建议

### 单个 PDF 尽快返回

```bash
uv run pdf-extract input.pdf \
  -o output.txt \
  --json output.json \
  --ocr-model rapidocr-v4-mobile \
  --ocr-workers 8 \
  --ocr-threads 1
```

适合独占机器或低并发服务。`ocr-workers` 可按 CPU 核数设置，`ocr-threads` 通常保持 `1`。

### 多请求服务器部署

```bash
PDF_EXTRACT_OCR_WORKERS=2 PDF_EXTRACT_OCR_THREADS=1 \
uv run pdf-extract input.pdf -o output.txt --json output.json
```

适合多个请求同时处理的服务。限制每个请求占用的 worker 数，避免整机 CPU 和内存被少数请求打满。

### 原生文字 PDF

```bash
uv run pdf-extract input.pdf -o output.txt --ocr never
```

如果确认 PDF 不是扫描件，禁用 OCR 可以获得最快速度。

## 调优原则

- 优先调 `ocr-workers`：多页扫描件通常先增加页面并发。
- 谨慎调 `ocr-threads`：默认 `1` 更稳；只有页面数少、单页很重、且机器空闲时才建议尝试 `2` 或更高。
- DPI 越高越慢：`--dpi 200` 是默认平衡值；`300` 可能提高小字识别效果，但会增加 CPU、内存和耗时。
- 多请求服务要保守：不要让每个请求都使用全部 CPU 核心数，否则并发请求会互相抢资源。

## 相关实现位置

- `src/pdf_text_extractor/extractor.py`：页面判定、OCR 进程池并发、worker/thread 默认值解析。
- `src/pdf_text_extractor/render.py`：批量渲染页面，避免每页重复打开 PDF。
- `src/pdf_text_extractor/ocr.py`：将 `ocr_threads` 传给 RapidOCR / ONNXRuntime。
- `src/pdf_text_extractor/cli.py`：CLI 参数和环境变量入口。
