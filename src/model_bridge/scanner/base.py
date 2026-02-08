"""
Base classes for the scanner module.

Defines the abstract interface (ScanStrategy) that all model format scanners
must implement. This enables the Strategy Pattern for extensible scanning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum


class ModelType(Enum):
    """Enumeration of supported model types."""
    HUGGINGFACE = "hf"
    HUGGINGFACE_LOCAL = "hf_local"
    GGUF = "gguf"
    TENSORRT = "tensorrt"
    SAFETENSORS = "safetensors"
    COMFYUI_CHECKPOINT = "comfyui_checkpoint"
    COMFYUI_LORA = "comfyui_lora"
    COMFYUI_VAE = "comfyui_vae"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """
    Standardized model information container.
    
    All scanners should return models in this format for consistency
    across the registry and API.
    
    Attributes:
        id: Unique identifier for the model (usually model name)
        path: Absolute path to the model file or directory
        type: Model type/format (hf, gguf, tensorrt, etc.)
        engine_support: List of inference engines that can load this model
        metadata: Additional format-specific metadata
        size_bytes: Total size in bytes (optional)
    """
    id: str
    path: str
    type: str
    engine_support: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    size_bytes: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the model info
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """
        Create ModelInfo from a dictionary.
        
        Args:
            data: Dictionary with model information
            
        Returns:
            ModelInfo instance
        """
        return cls(
            id=data.get("id", "unknown"),
            path=data.get("path", ""),
            type=data.get("type", "unknown"),
            engine_support=data.get("engine_support", []),
            metadata=data.get("metadata", {}),
            size_bytes=data.get("size_bytes"),
        )


class ScanStrategy(ABC):
    """
    Abstract base class for all model format scanners.
    
    Implement this interface to add support for new model formats.
    Each strategy is responsible for:
    1. Finding model files/directories in given paths
    2. Parsing metadata from the model format
    3. Returning standardized ModelInfo objects
    
    Example:
        class MyCustomStrategy(ScanStrategy):
            @property
            def name(self) -> str:
                return "my_format"
            
            def scan(self, paths: List[Path]) -> List[ModelInfo]:
                models = []
                for path in paths:
                    # ... scanning logic ...
                return models
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of this scanning strategy.
        
        Returns:
            Strategy name (e.g., "huggingface", "gguf", "tensorrt")
        """
        pass
    
    @abstractmethod
    def scan(self, paths: List[Path]) -> List[ModelInfo]:
        """
        Scan the given paths for models of this format.
        
        Args:
            paths: List of root directories to scan
            
        Returns:
            List of ModelInfo objects for discovered models
            
        Note:
            - Should handle non-existent paths gracefully
            - Should catch and log parsing errors, not crash
            - Should use recursive scanning where appropriate
        """
        pass
    
    def supports_path(self, path: Path) -> bool:
        """
        Check if this strategy can scan a given path.
        
        Override this for strategies that only work with specific paths
        (e.g., HuggingFace cache has a specific structure).
        
        Args:
            path: Path to check
            
        Returns:
            True if this strategy can scan the path
        """
        return path.exists() and path.is_dir()
    
    def get_model_size(self, path: Path) -> Optional[int]:
        """
        Calculate total size of a model (file or directory).
        
        Args:
            path: Path to model file or directory
            
        Returns:
            Size in bytes, or None if cannot be determined
        """
        try:
            if path.is_file():
                return path.stat().st_size
            elif path.is_dir():
                return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        except OSError:
            pass
        return None
