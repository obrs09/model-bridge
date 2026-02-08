# Model Bridge - GitHub Copilot Instructions

## 项目概述
Model Bridge 是一个统一的本地 AI 模型管理器，旨在扫描、索引和管理来自各种来源的 AI 模型(HF, GGUF, VLLM, ComfyUI, etc)。
使用conda basic1环境。
本文档只记录最重要的设计原则和代码规范，详细的实现细节请参阅CHANGELOG.md, README.md, 和源代码和注释。
更新需求，请直接修改CHANGELOG.md和README.md，不要修改这个文件。

## 1. Project Context
- **Core Stack:** Python 3.10+, `click` (CLI), `rich` (UI), `pydantic` (Schema), `huggingface_hub`.
- **Architecture:**
  1. **Scanner:** Strategy pattern to crawl directories & parse metadata (GGUF headers, HF cache).
  2. **Registry:** Singleton managing a JSON index of local models.
  3. **Interface:** CLI (`mb`), Python API, and `@smart_load` decorator.

## 2. Coding Standards (STRICT)
- **Type Hinting:** MUST use `typing` (List, Dict, Optional, etc.) for all function arguments and returns.
- **Path Handling:** EXCLUSIVELY use `pathlib.Path`. DO NOT use `os.path.join`.
- **Docstrings:** Use Google-style docstrings for all classes and public functions.
- **Error Handling:** Use `try/except` blocks for file I/O and parsing; fail gracefully with `rich.console.print`.
- **Style:** Follow PEP 8. Organize imports: Standard Lib > Third Party > Local.

## 3. Module Structure
```text
src/model_bridge/
├── config.py          # ConfigManager (Paths, Env Vars)
├── core.py            # ModelRegistry (Singleton, JSON Ops)
├── scanner/           # [Strategy Pattern]
│   ├── base.py        # Abstract Base Class
│   ├── strategies.py  # Concrete impls (GGUF, HF, TensorRT)
│   └── engine.py      # Execution Engine
├── decorator.py       # @smart_load implementation
└── cli.py             # Click commands & Rich output