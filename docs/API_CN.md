# Model Bridge API Reference

完整的 Python API 参考文档。

## 目录

- [快速开始](#快速开始)
- [核心类](#核心类)
  - [ModelRegistry](#modelregistry)
  - [ConfigManager](#configmanager)
  - [ModelRanker](#modelranker)
  - [ModelParser](#modelparser)
- [扫描器](#扫描器)
  - [ScannerEngine](#scannerengine)
  - [扫描策略](#扫描策略)
  - [ModelInfo](#modelinfo)
- [装饰器](#装饰器)
  - [@smart_load](#smart_load)
- [便捷函数](#便捷函数)
- [CLI 命令](#cli-命令)

---

## 快速开始

```python
from model_bridge import (
    ModelRegistry,      # 核心注册表
    find_model,         # 快速查找
    smart_load,         # 魔法装饰器
    get_config,         # 获取配置
    quick_scan,         # 快速扫描
)

# 1. 初始化并扫描
registry = ModelRegistry()
registry.scan()  # 增量扫描（只添加新模型）

# 2. 查找模型
model = registry.find("qwen 7b instruct")
print(model['path'])  # 模型路径

# 3. 使用装饰器自动解析
@smart_load
def load_model(model_path):
    return MyModel.load(model_path)

load_model("qwen")  # 自动替换为本地路径
```

---

## 核心类

### ModelRegistry

**模型注册表** - Model Bridge 的核心类，负责模型索引的持久化和查询。

```python
from model_bridge import ModelRegistry

registry = ModelRegistry()  # 单例模式
```

#### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `models` | `List[Dict]` | 所有已索引模型的列表 |
| `count` | `int` | 已索引模型数量 |
| `is_empty` | `bool` | 注册表是否为空 |
| `last_scan` | `float` | 上次扫描的时间戳 |
| `registry_path` | `Path` | 注册表 JSON 文件路径 |

#### 方法

##### `scan(force_refresh=False, verbose=True) -> Dict[str, int]`

扫描模型目录并更新索引。

**参数:**
- `force_refresh` (bool): 
  - `False` (默认): 增量模式，只添加新发现的模型
  - `True`: 完全刷新，删除现有索引并重建
- `verbose` (bool): 是否打印进度信息

**返回:** 扫描统计字典
```python
{
    'scanned': 104,   # 本次扫描发现的模型数
    'added': 5,       # 新添加的模型数
    'skipped': 99,    # 已存在跳过的模型数
    'removed': 0,     # 被移除的模型数 (仅 force_refresh)
    'total': 104      # 最终索引中的模型总数
}
```

**示例:**
```python
# 增量扫描（推荐日常使用）
stats = registry.scan()
print(f"新增 {stats['added']} 个模型")

# 完全刷新（重建索引）
stats = registry.scan(force_refresh=True)

# 静默扫描
stats = registry.scan(verbose=False)
```

---

##### `find(query, top_k=1, min_quant=None) -> Union[Dict, List[Dict], None]`

使用智能模糊匹配查找模型。

**参数:**
- `query` (str): 搜索查询，如 `"qwen 7b instruct"`
- `top_k` (int): 返回结果数量
  - `top_k=1`: 返回单个最佳匹配 (Dict) 或 None
  - `top_k>1`: 返回匹配列表 (List[Dict])
- `min_quant` (str, optional): 最低量化等级过滤，如 `"q4"`, `"q8"`

**返回:** 模型字典或列表

**示例:**
```python
# 查找最佳匹配
best = registry.find("qwen 7b")
if best:
    print(best['path'])

# 获取 Top 5
results = registry.find("llama", top_k=5)
for m in results:
    print(m['id'], m['path'])

# 只要 Q5 及以上精度
results = registry.find("qwen 7b", top_k=10, min_quant="q5")

# 只要 Q8 及以上
results = registry.find("llama", top_k=5, min_quant="q8")
```

---

##### `find_all(query, limit=10) -> List[Dict]`

查找所有匹配的模型（`find` 的别名，始终返回列表）。

```python
models = registry.find_all("qwen", limit=20)
```

---

##### `get_path(query) -> Optional[str]`

快速获取模型路径。

```python
path = registry.get_path("qwen 7b instruct")
if path:
    model = load_from_path(path)
```

---

##### `get_by_type(model_type) -> List[Dict]`

按模型类型过滤。

**类型值:**
- `"gguf"` - GGUF 文件
- `"hf"` - HuggingFace 缓存
- `"hf_local"` - 本地 HuggingFace 模型
- `"safetensors"` - Safetensors 文件
- `"tensorrt"` - TensorRT 引擎
- `"comfyui_checkpoint"` / `"comfyui_lora"` / `"comfyui_vae"`

```python
gguf_models = registry.get_by_type("gguf")
hf_models = registry.get_by_type("hf_local")
```

---

##### `get_by_engine(engine) -> List[Dict]`

按推理引擎过滤。

**引擎值:**
- `"llama.cpp"`, `"ollama"` - GGUF 模型
- `"transformers"`, `"vllm"`, `"sglang"` - HuggingFace 模型
- `"tensorrt_llm"`, `"triton"` - TensorRT 模型
- `"comfyui"`, `"diffusers"` - 图像生成模型

```python
# 获取所有可用于 vllm 的模型
vllm_models = registry.get_by_engine("vllm")

# 获取 llama.cpp 可用模型
llamacpp_models = registry.get_by_engine("llama.cpp")
```

---

##### `stats() -> Dict[str, Any]`

获取统计信息。

```python
stats = registry.stats()
print(f"总模型数: {stats['total_models']}")
print(f"总大小: {stats['total_size_gb']:.2f} GB")
print(f"按类型: {stats['by_type']}")
```

**返回:**
```python
{
    'total_models': 104,
    'total_size_gb': 554.35,
    'by_type': {
        'gguf': 65,
        'hf_local': 30,
        'safetensors': 9
    },
    'last_scan': 1707350400.0,
    'registry_path': '~/.config/model_bridge/registry.json'
}
```

---

##### `clear() -> None`

清空注册表。

```python
registry.clear()
assert registry.is_empty
```

---

##### `save() -> None`

手动保存注册表到磁盘（通常自动调用）。

---

### ConfigManager

**配置管理器** - 管理搜索路径和环境变量。

```python
from model_bridge import get_config, ConfigManager

config = get_config()  # 获取全局配置
```

#### 方法

##### `get_search_paths() -> List[Path]`

获取当前配置的搜索路径。

```python
paths = config.get_search_paths()
for p in paths:
    print(p)
```

---

##### `add_search_path(path) -> None`

添加搜索路径。

```python
config.add_search_path("D:/MyModels")
config.add_search_path("/data/llm-models")
```

---

##### `remove_search_path(path) -> bool`

移除搜索路径。

```python
removed = config.remove_search_path("D:/OldModels")
```

---

##### `set_hf_home(path) -> None`

设置 HuggingFace 缓存目录。

```python
config.set_hf_home("E:/HFCache")
# 等同于设置 HF_HOME 环境变量
```

---

##### `get_hf_home() -> Optional[Path]`

获取当前 HuggingFace 缓存目录。

---

##### `reset() -> None`

重置所有配置为默认值。

---

### ModelRanker

**模型排序器** - 实现分层加权模糊匹配算法。

```python
from model_bridge import ModelRanker
```

#### 类方法

##### `parse_features(text) -> Dict[str, Any]`

从文本中提取特征。

```python
features = ModelRanker.parse_features("qwen2.5-7b-instruct-q4_k_m.gguf")
# {
#     'size': '7',
#     'quant': 'q4_k_m',
#     'format': 'gguf',
#     'is_instruct': True,
#     'is_base': False,
#     'is_moe': False,
#     'tokens': {'qwen2', '5', '7b', 'instruct', 'q4', 'k', 'm', 'gguf'}
# }
```

---

##### `rank(query, candidates, min_quant=None) -> List[Dict]`

对候选模型进行排序。

```python
# 内部使用，通常不需要直接调用
sorted_models = ModelRanker.rank("qwen 7b", all_models, min_quant="q4")
```

---

##### `explain_score(query, model) -> Dict`

解释评分（调试用）。

```python
explanation = ModelRanker.explain_score("qwen 7b", model_dict)
print(explanation)
# {
#     'model_id': 'qwen2-7b-instruct-q5_k_m',
#     'scores': {
#         'token_matches': 200,
#         'instruct_bonus': 50,
#         'version_bonus': 25,
#         'quant_bonus': 30
#     },
#     'total': 305
# }
```

---

### ModelParser

**元数据解析器** - 从文件名提取结构化元数据。

```python
from model_bridge.metadata import ModelParser
```

#### 类方法

##### `parse(filename, parent_folder="") -> Dict[str, Any]`

解析模型文件名。

```python
meta = ModelParser.parse("qwen2.5-7b-instruct-q5_k_m.gguf", "Qwen2.5-7B-Instruct-GGUF")

# 返回:
{
    'family': 'qwen',           # 模型家族
    'subfamily': 'qwen2.5',     # 子家族/版本
    'size': 7.0,                # 参数量 (B)
    'size_raw': '7b',           # 原始大小字符串
    'quant': 'q5_k_m',          # 量化类型
    'quant_score': 50,          # 量化分数 (用于过滤)
    'format': 'gguf',           # 文件格式
    'architecture': ['instruct'],  # 架构标签
    'is_instruct': True,        # 是否指令微调
    'is_moe': False,            # 是否 MoE
    'special_tags': [],         # 特殊标签
    'domain_tags': [],          # 领域标签
    'source': None,             # 来源平台
    'keywords': ['7b', 'gguf', 'instruct', 'q5', 'q5_k_m', 'qwen', 'qwen2.5']
}
```

---

##### `get_quant_score(quant) -> int`

获取量化精度分数。

```python
ModelParser.get_quant_score("q4_k_m")  # 40
ModelParser.get_quant_score("q8_0")    # 80
ModelParser.get_quant_score("fp16")    # 160
```

**量化分数表:**
| 量化 | 分数 |
|------|------|
| IQ1 | 10 |
| Q2 | 20 |
| Q3 | 30 |
| Q4 | 40 |
| Q5 | 50 |
| Q6 | 60 |
| Q8 | 80 |
| FP16/BF16 | 160 |
| FP32 | 320 |

---

##### `is_quant_above(quant, threshold) -> bool`

检查量化是否达到阈值。

```python
ModelParser.is_quant_above("q5_k_m", "q4")  # True (Q5 >= Q4)
ModelParser.is_quant_above("q3_k_s", "q4")  # False (Q3 < Q4)
ModelParser.is_quant_above("fp16", "q8")    # True (FP16 >= Q8)
```

---

## 扫描器

### ScannerEngine

**扫描引擎** - 协调所有扫描策略。

```python
from model_bridge import ScannerEngine, get_config

config = get_config()
engine = ScannerEngine(config)
models = engine.run(verbose=True)  # 返回 List[ModelInfo]
```

#### 方法

##### `run(verbose=True) -> List[ModelInfo]`

运行所有扫描策略。

##### `add_strategy(strategy) -> None`

添加自定义扫描策略。

##### `remove_strategy(strategy_name) -> bool`

移除扫描策略。

##### `strategy_names -> List[str]`

获取所有策略名称。

---

### 扫描策略

所有策略都继承自 `ScanStrategy` 抽象基类。

```python
from model_bridge import (
    HuggingFaceStrategy,  # HuggingFace 缓存 + 本地模型
    GGUFStrategy,         # GGUF 文件
    TensorRTStrategy,     # TensorRT 引擎
    ComfyUIStrategy,      # ComfyUI 模型目录
)
from model_bridge.scanner.strategies import SafetensorsStrategy
```

#### 自定义策略

```python
from model_bridge import ScanStrategy, ModelInfo
from pathlib import Path
from typing import List

class MyCustomStrategy(ScanStrategy):
    @property
    def name(self) -> str:
        return "my_format"
    
    def scan(self, paths: List[Path]) -> List[ModelInfo]:
        models = []
        for path in paths:
            for file in path.rglob("*.myformat"):
                models.append(ModelInfo(
                    id=file.stem,
                    path=str(file),
                    type="my_format",
                    engine_support=["my_engine"],
                    metadata={"custom": "data"},
                    size_bytes=file.stat().st_size,
                ))
        return models

# 使用自定义策略
engine = ScannerEngine(config)
engine.add_strategy(MyCustomStrategy())
```

---

### ModelInfo

**模型信息数据类** - 所有扫描策略返回的标准格式。

```python
from model_bridge import ModelInfo

model = ModelInfo(
    id="qwen2-7b-instruct-q5_k_m",
    path="/models/qwen2-7b-instruct-q5_k_m.gguf",
    type="gguf",
    engine_support=["llama.cpp", "ollama"],
    metadata={
        "family": "qwen",
        "size": 7.0,
        "quant": "q5_k_m",
        "keywords": ["qwen", "7b", "instruct", "q5_k_m"]
    },
    size_bytes=5123456789,
)

# 转换为字典
model_dict = model.to_dict()

# 从字典创建
model2 = ModelInfo.from_dict(model_dict)
```

#### 字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `id` | str | 模型唯一标识符 |
| `path` | str | 模型文件/目录的绝对路径 |
| `type` | str | 模型类型 (gguf, hf, safetensors, etc.) |
| `engine_support` | List[str] | 支持的推理引擎 |
| `metadata` | Dict | 额外元数据 (family, quant, keywords, etc.) |
| `size_bytes` | int | 模型大小（字节） |

---

## 装饰器

### @smart_load

**智能加载装饰器** - 自动将模型名称替换为本地路径。

```python
from model_bridge import smart_load

@smart_load
def load_model(model_path, device="cuda"):
    return AutoModel.from_pretrained(model_path).to(device)

# 使用时：
model = load_model("qwen 7b")  # 自动查找并替换为本地路径
model = load_model("llama-3-8b-instruct")
```

#### 工作流程

1. **拦截**: 捕获第一个参数（通常是 model_id）
2. **查询**: 向 ModelRegistry 查询是否有本地匹配
3. **注入**: 如果找到，替换为本地路径；否则传递原值
4. **执行**: 运行原始函数

#### 支持的参数名

装饰器会自动识别以下参数名：
- `pretrained_model_name_or_path` (transformers)
- `model_id` (diffusers)
- `repo_id` (huggingface_hub)
- `model_name`, `model_path`, `model`
- `checkpoint`, `ckpt_path`

#### 高级版本

```python
from model_bridge.decorator import smart_load_v2

@smart_load_v2(strict=True)  # 找不到模型时抛出异常
def load_strict(model_path):
    pass

@smart_load_v2(silent=True)  # 不打印日志
def load_silent(model_path):
    pass

@smart_load_v2(fallback="default-model")  # 找不到时使用默认值
def load_with_fallback(model_path):
    pass
```

---

## 便捷函数

### find_model

快速查找模型路径。

```python
from model_bridge import find_model

path = find_model("qwen 7b instruct")
if path:
    print(f"Found: {path}")
```

---

### get_registry

获取全局 Registry 实例。

```python
from model_bridge import get_registry

registry = get_registry()
# 等同于 ModelRegistry()，但更明确表达意图
```

---

### quick_scan

快速扫描（不保存到 registry）。

```python
from model_bridge import quick_scan

# 扫描默认路径
models = quick_scan()

# 扫描指定路径
models = quick_scan(paths=["D:/Models", "E:/LLM"], verbose=True)

for m in models:
    print(m.id, m.path)
```

---

## CLI 命令

### 扫描

```bash
mb scan              # 增量扫描
mb scan --force      # 完全刷新
mb scan -p D:/Models # 添加搜索路径
mb scan --verbose    # 详细输出
```

### 列表

```bash
mb list              # 列出所有模型
mb list -t gguf      # 按类型过滤
mb list -n 20        # 限制数量
mb list -f json      # JSON 格式输出
mb list -f paths     # 只输出路径
```

### 搜索

```bash
mb find qwen                    # 模糊搜索
mb find qwen 7b instruct        # 多词搜索
mb find llama -n 5              # 返回 Top 5
mb find qwen -q q4              # Q4 或更高精度
mb find llama --min-quant q8    # Q8 或更高精度
mb find qwen --explain          # 显示评分详情
```

### 获取路径

```bash
mb get qwen              # 直接输出路径
model_path=$(mb get qwen)  # 用于脚本
```

### 详情

```bash
mb info qwen             # 显示模型详情
```

### 统计

```bash
mb stats                 # 显示统计信息
```

### 配置

```bash
mb config show                   # 显示当前配置
mb config add-path D:/NewModels  # 添加搜索路径
mb config set-hf-home E:/HFCache # 设置 HF 缓存目录
mb config reset                  # 重置配置
```

### 清空

```bash
mb clear                 # 清空注册表
```

---

## 数据存储

配置和索引文件位置：

```
~/.config/model_bridge/
├── config.json      # 搜索路径、环境变量配置
└── registry.json    # 模型索引缓存
```

- **Windows**: `C:\Users\<username>\.config\model_bridge\`
- **Linux/macOS**: `~/.config/model_bridge/`

---

## 模型元数据结构

每个模型在 registry 中的完整结构：

```json
{
  "id": "qwen2.5-7b-instruct-q5_k_m-00001-of-00002",
  "path": "D:\\LLM\\Qwen2.5-7B-Instruct-GGUF\\qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
  "type": "gguf",
  "engine_support": ["llama.cpp", "ollama"],
  "size_bytes": 5476083712,
  "metadata": {
    "family": "qwen",
    "subfamily": "qwen2.5",
    "size": 7.0,
    "size_raw": "7b",
    "quant": "q5_k_m",
    "quant_score": 50,
    "format": "gguf",
    "architecture": ["instruct"],
    "is_instruct": true,
    "is_moe": false,
    "special_tags": [],
    "domain_tags": [],
    "source": null,
    "keywords": ["7b", "gguf", "instruct", "q5", "q5_k_m", "qwen", "qwen2.5"],
    "shards": 2
  }
}
```
