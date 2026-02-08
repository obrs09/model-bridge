# Model Bridge API Reference

Complete Python API reference documentation.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Classes](#core-classes)
  - [ModelRegistry](#modelregistry)
  - [ConfigManager](#configmanager)
  - [ModelRanker](#modelranker)
  - [ModelParser](#modelparser)
- [Scanners](#scanners)
  - [ScannerEngine](#scannerengine)
  - [Scan Strategies](#scan-strategies)
  - [ModelInfo](#modelinfo)
- [Decorators](#decorators)
  - [@smart_load](#smart_load)
- [Convenience Functions](#convenience-functions)
- [CLI Commands](#cli-commands)

---

## Quick Start

```python
from model_bridge import (
    ModelRegistry,      # Core registry
    find_model,         # Quick find
    smart_load,         # Magic decorator
    get_config,         # Get configuration
    quick_scan,         # Quick scan
)

# 1. Initialize and scan
registry = ModelRegistry()
registry.scan()  # Incremental scan (only adds new models)

# 2. Find model
model = registry.find("qwen 7b instruct")
print(model['path'])  # Model path

# 3. Use decorator for automatic parsing
@smart_load
def load_model(model_path):
    return MyModel.load(model_path)

load_model("qwen")  # Automatically replaced with local path
```

---

## Core Classes

### ModelRegistry

**Model Registry** - The core class of Model Bridge, responsible for persistence and querying of model indices.

```python
from model_bridge import ModelRegistry

registry = ModelRegistry()  # Singleton pattern
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `models` | `List[Dict]` | List of all indexed models |
| `count` | `int` | Number of indexed models |
| `is_empty` | `bool` | Whether the registry is empty |
| `last_scan` | `float` | Timestamp of last scan |
| `registry_path` | `Path` | Path to registry JSON file |

#### Methods

##### `scan(force_refresh=False, verbose=True) -> Dict[str, int]`

Scan model directories and update index.

**Parameters:**
- `force_refresh` (bool): 
  - `False` (default): Incremental mode, only adds newly discovered models
  - `True`: Full refresh, deletes existing index and rebuilds
- `verbose` (bool): Whether to print progress information

**Returns:** Scan statistics dictionary
```python
{
    'scanned': 104,   # Number of models discovered in this scan
    'added': 5,       # Number of newly added models
    'skipped': 99,    # Number of existing models skipped
    'removed': 0,     # Number of models removed (force_refresh only)
    'total': 104      # Total models in final index
}
```

**Example:**
```python
# Incremental scan (recommended for daily use)
stats = registry.scan()
print(f"Added {stats['added']} models")

# Full refresh (rebuild index)
stats = registry.scan(force_refresh=True)

# Silent scan
stats = registry.scan(verbose=False)
```

---

##### `find(query, top_k=1, min_quant=None) -> Union[Dict, List[Dict], None]`

Find models using intelligent fuzzy matching.

**Parameters:**
- `query` (str): Search query, e.g., `"qwen 7b instruct"`
- `top_k` (int): Number of results to return
  - `top_k=1`: Returns single best match (Dict) or None
  - `top_k>1`: Returns list of matches (List[Dict])
- `min_quant` (str, optional): Minimum quantization level filter, e.g., `"q4"`, `"q8"`

**Returns:** Model dictionary or list

**Example:**
```python
# Find best match
best = registry.find("qwen 7b")
if best:
    print(best['path'])

# Get Top 5
results = registry.find("llama", top_k=5)
for m in results:
    print(m['id'], m['path'])

# Q5 and higher precision only
results = registry.find("qwen 7b", top_k=10, min_quant="q5")

# Q8 and higher only
results = registry.find("llama", top_k=5, min_quant="q8")
```

---

##### `find_all(query, limit=10) -> List[Dict]`

Find all matching models (alias for `find`, always returns a list).

```python
models = registry.find_all("qwen", limit=20)
```

---

##### `get_path(query) -> Optional[str]`

Quick get model path.

```python
path = registry.get_path("qwen 7b instruct")
if path:
    model = load_from_path(path)
```

---

##### `get_by_type(model_type) -> List[Dict]`

Filter by model type.

**Type values:**
- `"gguf"` - GGUF files
- `"hf"` - HuggingFace cache
- `"hf_local"` - Local HuggingFace models
- `"safetensors"` - Safetensors files
- `"tensorrt"` - TensorRT engines
- `"comfyui_checkpoint"` / `"comfyui_lora"` / `"comfyui_vae"`

```python
gguf_models = registry.get_by_type("gguf")
hf_models = registry.get_by_type("hf_local")
```

---

##### `get_by_engine(engine) -> List[Dict]`

Filter by inference engine.

**Engine values:**
- `"llama.cpp"`, `"ollama"` - GGUF models
- `"transformers"`, `"vllm"`, `"sglang"` - HuggingFace models
- `"tensorrt_llm"`, `"triton"` - TensorRT models
- `"comfyui"`, `"diffusers"` - Image generation models

```python
# Get all models available for vllm
vllm_models = registry.get_by_engine("vllm")

# Get llama.cpp compatible models
llamacpp_models = registry.get_by_engine("llama.cpp")
```

---

##### `stats() -> Dict[str, Any]`

Get statistics.

```python
stats = registry.stats()
print(f"Total models: {stats['total_models']}")
print(f"Total size: {stats['total_size_gb']:.2f} GB")
print(f"By type: {stats['by_type']}")
```

**Returns:**
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

Clear the registry.

```python
registry.clear()
assert registry.is_empty
```

---

##### `save() -> None`

Manually save registry to disk (usually called automatically).

---

### ConfigManager

**Configuration Manager** - Manages search paths and environment variables.

```python
from model_bridge import get_config, ConfigManager

config = get_config()  # Get global configuration
```

#### Methods

##### `get_search_paths() -> List[Path]`

Get currently configured search paths.

```python
paths = config.get_search_paths()
for p in paths:
    print(p)
```

---

##### `add_search_path(path) -> None`

Add a search path.

```python
config.add_search_path("D:/MyModels")
config.add_search_path("/data/llm-models")
```

---

##### `remove_search_path(path) -> bool`

Remove a search path.

```python
removed = config.remove_search_path("D:/OldModels")
```

---

##### `set_hf_home(path) -> None`

Set HuggingFace cache directory.

```python
config.set_hf_home("E:/HFCache")
# Equivalent to setting HF_HOME environment variable
```

---

##### `get_hf_home() -> Optional[Path]`

Get current HuggingFace cache directory.

---

##### `reset() -> None`

Reset all configurations to default values.

---

### ModelRanker

**Model Ranker** - Implements hierarchical weighted fuzzy matching algorithm.

```python
from model_bridge import ModelRanker
```

#### Class Methods

##### `parse_features(text) -> Dict[str, Any]`

Extract features from text.

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

Rank candidate models.

```python
# Internal use, usually no need to call directly
sorted_models = ModelRanker.rank("qwen 7b", all_models, min_quant="q4")
```

---

##### `explain_score(query, model) -> Dict`

Explain scoring (for debugging).

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

**Metadata Parser** - Extracts structured metadata from filenames.

```python
from model_bridge.metadata import ModelParser
```

#### Class Methods

##### `parse(filename, parent_folder="") -> Dict[str, Any]`

Parse model filename.

```python
meta = ModelParser.parse("qwen2.5-7b-instruct-q5_k_m.gguf", "Qwen2.5-7B-Instruct-GGUF")

# Returns:
{
    'family': 'qwen',           # Model family
    'subfamily': 'qwen2.5',     # Subfamily/version
    'size': 7.0,                # Parameter count (B)
    'size_raw': '7b',           # Raw size string
    'quant': 'q5_k_m',          # Quantization type
    'quant_score': 50,          # Quantization score (for filtering)
    'format': 'gguf',           # File format
    'architecture': ['instruct'],  # Architecture tags
    'is_instruct': True,        # Instruction-tuned
    'is_moe': False,            # Mixture of Experts
    'special_tags': [],         # Special tags
    'domain_tags': [],          # Domain tags
    'source': None,             # Source platform
    'keywords': ['7b', 'gguf', 'instruct', 'q5', 'q5_k_m', 'qwen', 'qwen2.5']
}
```

---

##### `get_quant_score(quant) -> int`

Get quantization precision score.

```python
ModelParser.get_quant_score("q4_k_m")  # 40
ModelParser.get_quant_score("q8_0")    # 80
ModelParser.get_quant_score("fp16")    # 160
```

**Quantization Score Table:**
| Quantization | Score |
|--------------|-------|
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

Check if quantization meets threshold.

```python
ModelParser.is_quant_above("q5_k_m", "q4")  # True (Q5 >= Q4)
ModelParser.is_quant_above("q3_k_s", "q4")  # False (Q3 < Q4)
ModelParser.is_quant_above("fp16", "q8")    # True (FP16 >= Q8)
```

---

## Scanners

### ScannerEngine

**Scanner Engine** - Coordinates all scanning strategies.

```python
from model_bridge import ScannerEngine, get_config

config = get_config()
engine = ScannerEngine(config)
models = engine.run(verbose=True)  # Returns List[ModelInfo]
```

#### Methods

##### `run(verbose=True) -> List[ModelInfo]`

Run all scanning strategies.

##### `add_strategy(strategy) -> None`

Add custom scanning strategy.

##### `remove_strategy(strategy_name) -> bool`

Remove a scanning strategy.

##### `strategy_names -> List[str]`

Get all strategy names.

---

### Scan Strategies

All strategies inherit from the `ScanStrategy` abstract base class.

```python
from model_bridge import (
    HuggingFaceStrategy,  # HuggingFace cache + local models
    GGUFStrategy,         # GGUF files
    TensorRTStrategy,     # TensorRT engines
    ComfyUIStrategy,      # ComfyUI model directories
)
from model_bridge.scanner.strategies import SafetensorsStrategy
```

#### Custom Strategies

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

# Use custom strategy
engine = ScannerEngine(config)
engine.add_strategy(MyCustomStrategy())
```

---

### ModelInfo

**Model Information Dataclass** - Standard format returned by all scanning strategies.

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

# Convert to dictionary
model_dict = model.to_dict()

# Create from dictionary
model2 = ModelInfo.from_dict(model_dict)
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique model identifier |
| `path` | str | Absolute path to model file/directory |
| `type` | str | Model type (gguf, hf, safetensors, etc.) |
| `engine_support` | List[str] | Supported inference engines |
| `metadata` | Dict | Additional metadata (family, quant, keywords, etc.) |
| `size_bytes` | int | Model size in bytes |

---

## Decorators

### @smart_load

**Smart Load Decorator** - Automatically replaces model names with local paths.

```python
from model_bridge import smart_load

@smart_load
def load_model(model_path, device="cuda"):
    return AutoModel.from_pretrained(model_path).to(device)

# Usage:
model = load_model("qwen 7b")  # Automatically finds and replaces with local path
model = load_model("llama-3-8b-instruct")
```

#### Workflow

1. **Intercept**: Captures the first argument (usually model_id)
2. **Query**: Queries ModelRegistry for local matches
3. **Inject**: Replaces with local path if found; otherwise passes original value
4. **Execute**: Runs original function

#### Supported Parameter Names

The decorator automatically recognizes these parameter names:
- `pretrained_model_name_or_path` (transformers)
- `model_id` (diffusers)
- `repo_id` (huggingface_hub)
- `model_name`, `model_path`, `model`
- `checkpoint`, `ckpt_path`

#### Advanced Version

```python
from model_bridge.decorator import smart_load_v2

@smart_load_v2(strict=True)  # Raises exception if model not found
def load_strict(model_path):
    pass

@smart_load_v2(silent=True)  # No logging
def load_silent(model_path):
    pass

@smart_load_v2(fallback="default-model")  # Uses default if not found
def load_with_fallback(model_path):
    pass
```

---

## Convenience Functions

### find_model

Quick find model path.

```python
from model_bridge import find_model

path = find_model("qwen 7b instruct")
if path:
    print(f"Found: {path}")
```

---

### get_registry

Get global Registry instance.

```python
from model_bridge import get_registry

registry = get_registry()
# Equivalent to ModelRegistry(), but more explicit
```

---

### quick_scan

Quick scan (doesn't save to registry).

```python
from model_bridge import quick_scan

# Scan default paths
models = quick_scan()

# Scan specific paths
models = quick_scan(paths=["D:/Models", "E:/LLM"], verbose=True)

for m in models:
    print(m.id, m.path)
```

---

## CLI Commands

### Scan

```bash
mb scan              # Incremental scan
mb scan --force      # Full refresh
mb scan -p D:/Models # Add search path
mb scan --verbose    # Verbose output
```

### List

```bash
mb list              # List all models
mb list -t gguf      # Filter by type
mb list -n 20        # Limit count
mb list -f json      # JSON format output
mb list -f paths     # Output paths only
```

### Search

```bash
mb find qwen                    # Fuzzy search
mb find qwen 7b instruct        # Multi-word search
mb find llama -n 5              # Return Top 5
mb find qwen -q q4              # Q4 or higher precision
mb find llama --min-quant q8    # Q8 or higher precision
mb find qwen --explain          # Show scoring details
```

### Get Path

```bash
mb get qwen              # Output path directly
model_path=$(mb get qwen)  # Use in scripts
```

### Details

```bash
mb info qwen             # Show model details
```

### Statistics

```bash
mb stats                 # Show statistics
```

### Configuration

```bash
mb config show                   # Show current configuration
mb config add-path D:/NewModels  # Add search path
mb config set-hf-home E:/HFCache # Set HF cache directory
mb config reset                  # Reset configuration
```

### Clear

```bash
mb clear                 # Clear registry
```

---

## Data Storage

Configuration and index file locations:

```
~/.config/model_bridge/
├── config.json      # Search paths, environment variable configuration
└── registry.json    # Model index cache
```

- **Windows**: `C:\Users\<username>\.config\model_bridge\`
- **Linux/macOS**: `~/.config/model_bridge/`

---

## Model Metadata Structure

Complete structure of each model in the registry:

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
