"""
Model Bridge - A unified model manager for local AI models.

This package provides tools to scan, index, and manage AI models from various sources:
- HuggingFace Cache (Transformers, vLLM, SGLang)
- GGUF files (llama.cpp, Ollama)
- TensorRT-LLM compiled engines
- ComfyUI models (checkpoints, LoRAs, VAEs)
- Safetensors files

Architecture:
- Scanner Layer: Strategy pattern for extensible format support
- Registry Layer: Singleton for model indexing and lookup
- Interface Layer: CLI, Python API, and @smart_load decorator
"""

__version__ = "0.1.0"

# Configuration
from .config import ConfigManager, get_config

# Core registry
from .core import ModelRegistry, Registry, find_model, get_registry

# Ranker (Fuzzy Matching)
from .ranker import ModelRanker

# Scanner (Strategy Pattern)
from .scanner import (
    ScanStrategy,
    ModelInfo,
    ScannerEngine,
    HuggingFaceStrategy,
    GGUFStrategy,
    TensorRTStrategy,
    ComfyUIStrategy,
)
from .scanner.engine import quick_scan

# Decorator
from .decorator import smart_load

__all__ = [
    # Config
    "ConfigManager",
    "get_config",
    # Registry
    "ModelRegistry",
    "Registry",
    "find_model",
    "get_registry",
    # Ranker
    "ModelRanker",
    # Scanner
    "ScanStrategy",
    "ModelInfo",
    "ScannerEngine",
    "HuggingFaceStrategy",
    "GGUFStrategy",
    "TensorRTStrategy",
    "ComfyUIStrategy",
    "quick_scan",
    # Decorator
    "smart_load",
    # Version
    "__version__",
]
