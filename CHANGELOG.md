# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 🏷️ **元数据提取与关键词系统 (Metadata Extractor)**
  - `metadata.py`: ModelParser 类，从模型文件名解析结构化元数据
    - **关键词层次体系**:
      1. 模型家族: qwen, llama, deepseek, mistral, phi, gemma, yi, falcon...
      2. 子家族版本: qwen2.5, llama3.1, mistral-nemo, deepseek-v3...
      3. 架构/变体: instruct, chat, base, vision, multimodal, tts, asr, moe...
      4. 量化/格式: gguf, q4_k_m, q5_k_s, q8_0, fp16, gptq, awq, exl2...
      5. 参数量: 7b, 70b, 1.8b, tiny, small, medium, large...
      6. 特殊标记: v1, v2, turbo, 128k, moe, a3b...
      7. 领域/语言: chinese, multilingual, coding, math, roleplay...
      8. 来源/平台: huggingface, ollama, thebloke, bartowski, comfyui...
    - **量化评级系统**: QUANT_LEVELS 映射表 (IQ1=10 ~ FP32=320)
    - `parse()`: 解析文件名，返回 family/subfamily/size/quant/quant_score/keywords
    - `get_quant_score()`: 获取量化精度评分
    - `is_quant_above()`: 检查量化是否达到阈值 (支持 "Q4以上" 查询)
  - Scanner 策略集成 ModelParser:
    - HuggingFaceStrategy: 解析 repo_id 和目录名
    - GGUFStrategy: 解析 GGUF 文件名 + GGUF header 元数据
    - SafetensorsStrategy: 解析 Safetensors 文件名
    - ComfyUIStrategy: 解析模型文件名 + category 标签
  - JSON 结构新增 metadata.keywords 字段，支持多维度搜索

- 🔍 **增强搜索功能**
  - Ranker 升级: 使用 metadata.keywords 进行关键词匹配
    - 新增 `keyword_match` 权重 (+80 per match)
    - 新增 `family_match` 权重 (+100)
    - 新增 `size_match` 权重 (+50)
  - CLI `mb find` 新增 `--min-quant/-q` 选项:
    - `mb find qwen -q q4`: 只显示 Q4 及以上精度
    - `mb find llama -q q8`: 只显示 Q8 及以上精度
  - Registry.find() 新增 min_quant 参数支持

- 🔌 **策略模式扫描层 (Strategy Pattern Scanner Layer)**
  - `config.py`: ConfigManager 配置管理器
    - 支持自定义搜索路径
    - 支持修改 HF_HOME 环境变量
    - 支持 vLLM/ComfyUI/Ollama 路径配置
  - `scanner/base.py`: ScanStrategy 抽象基类和 ModelInfo 数据类
  - `scanner/strategies.py`: 具体策略实现
    - `HuggingFaceStrategy`: HF 缓存 + 本地 HF 模型
    - `GGUFStrategy`: GGUF 文件扫描 (支持 llama.cpp, Ollama)
    - `TensorRTStrategy`: TensorRT-LLM .engine/.plan 文件
    - `ComfyUIStrategy`: ComfyUI 模型目录结构
    - `SafetensorsStrategy`: 独立 Safetensors 文件
  - `scanner/engine.py`: ScannerEngine 扫描调度器
    - 统一运行所有策略
    - 自动去重
    - 支持 verbose 模式

- 🎯 **分层加权模糊匹配系统 (ModelRanker)**
  - `ranker.py`: 智能模型搜索排序
    - `parse_features()`: 从模型名解析 size/quant/instruct/moe 特征
    - Hard Filters: 尺寸、MoE、格式等硬性过滤
    - Token Match Quality Gate: 短查询需100%匹配，长查询≥30%匹配
    - Soft Scoring: 多维度评分系统
      - Token 匹配: 每 token +100
      - Exact ID 匹配: +500
      - Instruct 偏好: +50
      - 版本加权: qwen2.5 > qwen2 > qwen (×10)
      - 量化偏好: q4_k_m/q5_k_m +30, q6_k/q8_0 +20
      - GGUF 格式: +10
    - `explain_score()`: 调试用评分解释

- ✨ **@smart_load 装饰器 (Interface Layer)**
  - `decorator.py`: 自动模型路径解析魔法
    - 拦截函数参数，自动替换为本地路径
    - 支持位置参数和关键字参数
    - 兼容 transformers/diffusers/llama-cpp/ollama
    - 惰性初始化 Registry
    - `smart_load_v2`: 高级版本，支持 strict/silent/fallback
    - `ModelNotFoundError`: 严格模式异常

- 🖥️ **CLI 命令行界面 (Interface Layer)**
  - `cli.py`: Click + Rich 实现的命令行工具
    - `mb scan`: 扫描模型目录
    - `mb list`: 列出所有模型（支持 --type/--format/--limit）
    - `mb find <query>`: 模糊搜索（支持 --explain 显示评分）
    - `mb get <query>`: 快速获取模型路径（便于脚本调用）
    - `mb info <query>`: 显示模型详细信息
    - `mb stats`: 显示统计信息
    - `mb config show/add-path/set-hf-home/reset`: 配置管理
    - `mb clear`: 清空注册表

- 🧠 **核心层 ModelRegistry (Registry Layer)**
  - `core.py`: ModelRegistry 单例模式实现
    - JSON 持久化存储 (`~/.config/model_bridge/registry.json`)
    - 使用 ModelRanker 进行智能搜索排序
    - `find()`: 智能模糊查询，支持 top_k
    - `find_all()`: 返回所有匹配结果
    - `get_path()`: 便捷方法，直接返回模型路径
    - `get_by_type()`: 按模型类型过滤
    - `get_by_engine()`: 按推理引擎过滤 (vllm, llama.cpp 等)
    - `stats()`: 统计信息 (总数、大小、按类型分布)
    - 自动检测过期缓存 (>7天提示刷新)

### Changed

- 重构 scanner 模块为目录结构，采用策略模式
- 更新 `__init__.py` 导出新的 API
- 重写 `core.py`，从简单 Registry 升级为完整 ModelRegistry

---

## [0.1.0] - 2026-02-07

### Added

- 🎉 **项目初始化**
  - 建立标准 Python 包结构 (src layout)
  - 创建 `pyproject.toml` 配置文件
  - 添加 MIT 开源协议

- 📦 **核心模块**
  - `core.py`: Registry 单例模式实现，模型索引管理
  - `utils.py`: 工具函数 (哈希计算, 文件大小格式化等)
  - `decorator.py`: `@smart_load` 装饰器实现
  - `cli.py`: Click 命令行界面

- 🧪 **测试框架**
  - `test_scanner.py`: 扫描器单元测试
  - `test_decorator.py`: 装饰器单元测试

- 📝 **文档**
  - `README.md`: 项目说明书
  - `.github/copilot-instructions.md`: 项目架构说明
  - `CHANGELOG.md`: 变更记录

### Technical Details

- 使用 conda basic1 作为运行环境
- 依赖: click>=8.0.0, rich>=13.0.0
- 开发依赖: pytest>=7.0.0, pytest-cov>=4.0.0
