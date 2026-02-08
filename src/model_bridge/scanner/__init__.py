"""
Scanner module for Model Bridge.

This module implements the Strategy Pattern for scanning different model formats:
- HuggingFace (Transformers, vLLM, SGLang)
- GGUF (llama.cpp, Ollama)
- TensorRT-LLM
- ComfyUI (checkpoints, LoRAs, VAEs)

Each format is implemented as a separate strategy that can be extended without
modifying the core scanning engine.
"""

from .base import ScanStrategy, ModelInfo, ModelType
from .strategies import (
    ScanUtils,
    HuggingFaceStrategy,
    GGUFStrategy,
    TensorRTStrategy,
    ComfyUIStrategy,
    SafetensorsStrategy,
)
from .engine import ScannerEngine, quick_scan

__all__ = [
    "ScanStrategy",
    "ModelInfo",
    "ModelType",
    "ScanUtils",
    "HuggingFaceStrategy",
    "GGUFStrategy",
    "TensorRTStrategy",
    "ComfyUIStrategy",
    "SafetensorsStrategy",
    "ScannerEngine",
    "quick_scan",
]
