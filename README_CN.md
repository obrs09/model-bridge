# Model Bridge 🌉

A unified model manager for local AI models.

## Features

- 🔍 **Unified Scanning**: Automatically discover models from HuggingFace Cache, GGUF directories, ComfyUI, and more
- 📋 **Smart Registry**: JSON-based index for fast model lookup
- 🎯 **Intelligent Search**: Layered weighted fuzzy matching with hard filters and soft scoring
- 🖥️ **CLI Tools**: Simple commands like `mb scan` and `mb list`
- 🔌 **Developer API**: Programmatic access via `find_model()`

## Data Storage

配置和索引文件存储在用户配置目录：

```
~/.config/model_bridge/
├── config.json      # 搜索路径、环境变量配置
└── registry.json    # 模型索引缓存
```

- **Windows**: `C:\Users\<username>\.config\model_bridge\`
- **Linux/macOS**: `~/.config/model_bridge/`

## Installation

```bash
pip install model-bridge
```

## Quick Start

### CLI Usage

```bash
# Scan for models
mb scan
mb scan -p D:/MyModels --verbose

# List all models
mb list
mb list -t gguf -n 20
mb list -f json > models.json

# Find models (fuzzy search)
mb find qwen
mb find qwen 7b instruct --explain
mb find llama gguf q4_k_m

# Get model path (for scripting)
mb get llama
model_path=$(mb get qwen 7b)

# Show model details
mb info qwen

# Statistics
mb stats

# Configuration
mb config show
mb config add-path D:/NewModels
mb config set-hf-home E:/HFCache
```

### Python API

```python
from model_bridge import ModelRegistry, find_model, smart_load

# Initialize registry
registry = ModelRegistry()

# Scan for models (incremental by default - only adds new models)
registry.scan()

# Force full refresh (rebuild registry from scratch)
registry.scan(force_refresh=True)

# Smart fuzzy search with ranking
result = registry.find("qwen 7b instruct")  # Returns best match
print(result['path'])  # Model file path

# Get top-k results with min quantization filter
top_5 = registry.find("llama", top_k=5)
q8_plus = registry.find("qwen 7b", top_k=10, min_quant="q8")  # Q8 or above

# Filter by type
gguf_models = registry.get_by_type("gguf")
hf_models = registry.get_by_type("hf_local")

# Quick path lookup
model_path = find_model("llama-7b")

# Use smart_load decorator
@smart_load
def load_my_model(model_path):
    # model_path will be automatically resolved
    pass
```

### Search Query Examples

| Query | Behavior |
|-------|----------|
| `qwen` | Best Qwen model (prefers Instruct, newer version) |
| `qwen 7b` | Only 7B size models |
| `qwen instruct` | Prioritize instruction-tuned |
| `llama gguf` | Only GGUF format |
| `q4_k_m` | Prefer Q4_K_M quantization |
| `qwen -q q4` | Q4 or above precision only |
| `llama --min-quant q8` | Q8 or above precision only |
| `whisper` | Search audio models by family |
| `sdxl` | Search diffusion models |

## Metadata & Keywords System

Model Bridge automatically extracts structured metadata from model filenames:

```python
# Example: "qwen2.5-7b-instruct-q5_k_m.gguf"
metadata = {
    "family": "qwen",
    "subfamily": "qwen2.5",
    "size": 7.0,
    "quant": "q5_k_m",
    "quant_score": 50,  # Used for "Q4 or above" filtering
    "is_instruct": True,
    "keywords": ["7b", "gguf", "instruct", "q5", "q5_k_m", "qwen", "qwen2.5"]
}
```

### Keyword Categories

| Priority | Category | Examples |
|----------|----------|----------|
| 1 | Model Family | qwen, llama, deepseek, mistral, phi, gemma |
| 2 | Sub-family | qwen2.5, llama3.1, deepseek-v3, phi-4 |
| 3 | Architecture | instruct, chat, base, vision, moe, tts |
| 4 | Quantization | gguf, q4_k_m, q8_0, fp16, gptq, awq |
| 5 | Size | 7b, 70b, 1.8b, tiny, small, large |
| 6 | Special Tags | v1, turbo, 128k, moe |
| 7 | Domain | chinese, coding, math, roleplay |
| 8 | Source | huggingface, ollama, thebloke, bartowski |

### Quantization Filtering

Filter models by minimum precision level:

```bash
# CLI
mb find qwen 7b -q q4      # Q4 or above
mb find llama --min-quant q8  # Q8 or above

# Python API
registry.find("qwen 7b", min_quant="q4")
registry.find("llama", top_k=10, min_quant="q8")
```

Quantization scores (higher = more precision):
- IQ1=10, Q2=20, Q3=30, Q4=40, Q5=50, Q6=60, Q8=80, FP16=160, FP32=320

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Interface Layer                       │
│           CLI (mb) │ Python API │ @smart_load            │
├─────────────────────────────────────────────────────────┤
│                    Registry Layer                        │
│   ModelRegistry (Singleton) + ModelRanker (Scoring)      │
├─────────────────────────────────────────────────────────┤
│                    Scanner Layer                         │
│   HF │ GGUF │ TensorRT │ ComfyUI │ Safetensors           │
└─────────────────────────────────────────────────────────┘
```

## License

MIT License
