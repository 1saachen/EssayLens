# EssayLens

一个用于**英文作文自动评分**的实验性项目，当前仓库主要包含三部分能力：

- `ocr/`：作文图片 OCR 识别（PaddleOCR）
- `grade/`：作文自动评分（FusionGraphBERT + 图结构 + 特征工程）
- `comment/`：基于通义千问生成中英文评语

> 当前仓库内容以脚本与模型代码为主，未包含完整前端工程代码。

---

## 目录结构

```text
EssayLens/
├── README.md
├── requirements.txt
├── requirements-grade.txt
├── ocr/
│   ├── main.py
│   └── ocr_config_example.py
├── grade/
│   ├── grade_run.py
│   └── training/train.py
└── comment/
    ├── comment_generator.py
    ├── config.py
    ├── start.py
    └── test.py
```

---

## 环境准备

### 1) Python 版本

建议 Python `3.10+`。

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含 OCR 与评语模块的基础依赖。使用评分或训练模块时执行：

```bash
python -m pip install -r requirements-grade.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

PyTorch 的 CPU/GPU 版本应按机器环境安装；上述依赖范围用于限制主要版本，尚不是经过完整训练验证的锁定环境。

### 3) 安装 PaddlePaddle（用于 OCR）

CPU（示例）：

```bash
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

GPU（示例，CUDA 12.9）：

```bash
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

---

## 快速开始

以下命令均在仓库根目录执行。

### OCR：识别作文图片

```bash
python ocr/main.py ocr/essay2.jpg --out ocr/ocr_output --lang en
```

可选参数：

- `--gpu`：启用 GPU（环境支持时）
- `--debug-structure`：打印 PaddleOCR 返回结构

输出会包含：

- 提取文本 `ocr_result.json`
- 标注图片 `*_annotated.jpg`

### 评分：单篇作文打分

```bash
python grade/grade_run.py
```

该脚本默认从当前目录读取以下训练产物，**仓库没有附带训练权重，首次使用需先训练或恢复文件**：

- `fusion_graphbert_best.pth`：最佳模型权重。
- `feature_scaler.pkl`、`target_scaler.pkl`：训练时拟合的特征与分数缩放器。
- `model_config.json`：模型结构、分数范围等配置。
- `tokenizer/`：与训练匹配的分词器。

重新训练时，把具有 `full_text`、`score`、`assignment` 列的 `ASAP2_train_sourcetexts.csv` 放在仓库根目录，再执行：

```bash
python grade/training/train.py
```

训练会在当前目录导出上述文件及 `fusion_graphbert_full.pth`。数据清洗、分数范围和训练参数见训练脚本，其他数据集需先适配。模型配置统一使用小写字段（如 `model_name`、`graph_out`、`score_min`），推理也兼容旧大写字段与 `min_score` / `max_score`。这些文件应来自同一次训练；旧版本误导出的目标缩放器不能仅靠修改代码修复，需要重新训练或从原始数据恢复。

### 评语生成：基于分数与分析结果生成双语反馈

先设置 API Key：

```bash
export QIANWEN_API_KEY=your_real_api_key
```

Windows PowerShell：

```powershell
$env:QIANWEN_API_KEY="your_real_api_key"
```

检查环境变量：

```bash
python comment/test.py
```

运行示例：

```bash
python comment/start.py
```

---

## 模块说明

### `ocr/`

- `main.py`：OCR 主程序，支持单图/目录批量处理
- `ocr_config_example.py`：配置示例

### `grade/`

- `training/train.py`：训练脚本
- `grade_run.py`：推理/评分脚本（含示例）

### `comment/`

- `comment_generator.py`：评语生成核心逻辑（Prompt 构建、解析、质量检查、降级策略）
- `start.py`：评语生成测试入口
- `config.py`：通义千问参数配置

---

## 注意事项

- `grade/` 模块依赖较重，首次运行可能需要下载模型与 NLP 资源（如 tokenizer、spaCy 模型、nltk 数据）。
- 请勿将真实 API Key 写入代码文件，建议仅通过环境变量注入。

## 无模型回归检查

```bash
python -m unittest discover -s tests -v
```

这些检查不下载模型、不调用付费 API；验证配置兼容、缩放器导出和融合层维度。真实推理和训练仍需完整环境、数据与模型文件。
