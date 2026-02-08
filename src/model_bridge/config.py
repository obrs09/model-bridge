"""
Configuration management for Model Bridge.

Handles path configuration, environment variables, and user settings.
Supports modifying HuggingFace cache path, vLLM cache, and custom search paths.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


# Default configuration file location
CONFIG_DIR = Path.home() / ".config" / "model_bridge"
CONFIG_FILE = CONFIG_DIR / "config.json"


class ConfigManager:
    """
    Centralized configuration manager for Model Bridge.
    
    Manages:
    - Custom model search paths
    - HuggingFace cache location (HF_HOME)
    - vLLM cache location
    - Other environment variables
    
    Example:
        >>> config = ConfigManager()
        >>> config.add_search_path("/my/models")
        >>> config.set_hf_home("/new/cache/path")
    """
    
    _instance: Optional["ConfigManager"] = None
    
    def __new__(cls) -> "ConfigManager":
        """Singleton pattern - ensure only one config manager exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config: Dict[str, Any] = self._load_config()
        self._apply_env_vars()
        self._initialized = True
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Return default configuration values.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "search_paths": [],  # User-defined local model repositories
            "hf_home": str(Path.home() / ".cache" / "huggingface"),  # Default HF path
            "vllm_cache": str(Path.home() / ".cache" / "vllm"),  # Reserved for vLLM
            "comfyui_path": None,  # ComfyUI models directory
            "ollama_models": None,  # Ollama models directory
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from disk, merging with defaults.
        
        Returns:
            Configuration dictionary
        """
        if not CONFIG_FILE.exists():
            return self._default_config()
        
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            # Merge with defaults (user config takes precedence)
            return {**self._default_config(), **user_config}
        except (json.JSONDecodeError, IOError):
            return self._default_config()
    
    def save(self) -> None:
        """Save current configuration to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
    
    def _apply_env_vars(self) -> None:
        """
        Apply environment variables based on configuration.
        
        This affects the behavior of Transformers, vLLM, and other libraries
        by setting their cache paths.
        """
        # HuggingFace / Transformers cache
        if self.config.get("hf_home"):
            hf_home = self.config["hf_home"]
            os.environ["HF_HOME"] = hf_home
            # Compatibility with older transformers versions
            os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "hub")
            os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        
        # vLLM cache
        if self.config.get("vllm_cache"):
            os.environ["VLLM_CACHE_ROOT"] = self.config["vllm_cache"]
    
    def add_search_path(self, path: str) -> None:
        """
        Add a directory to the list of model search paths.
        
        Args:
            path: Path to add (will be resolved to absolute path)
        """
        resolved = str(Path(path).resolve())
        if resolved not in self.config["search_paths"]:
            self.config["search_paths"].append(resolved)
            self.save()
    
    def remove_search_path(self, path: str) -> bool:
        """
        Remove a directory from the search paths.
        
        Args:
            path: Path to remove
            
        Returns:
            True if path was removed, False if not found
        """
        resolved = str(Path(path).resolve())
        if resolved in self.config["search_paths"]:
            self.config["search_paths"].remove(resolved)
            self.save()
            return True
        return False
    
    def get_search_paths(self) -> List[Path]:
        """
        Get all configured search paths as Path objects.
        
        Returns:
            List of Path objects
        """
        return [Path(p) for p in self.config["search_paths"]]
    
    def set_hf_home(self, path: str) -> None:
        """
        Set the HuggingFace cache directory.
        
        This affects where models are downloaded and cached.
        
        Args:
            path: New cache path
        """
        self.config["hf_home"] = str(Path(path).resolve())
        self.save()
        self._apply_env_vars()  # Apply immediately for current process
    
    def get_hf_home(self) -> Path:
        """
        Get the HuggingFace cache directory.
        
        Returns:
            Path to HF cache
        """
        return Path(self.config["hf_home"])
    
    def set_vllm_cache(self, path: str) -> None:
        """
        Set the vLLM cache directory.
        
        Args:
            path: New cache path
        """
        self.config["vllm_cache"] = str(Path(path).resolve())
        self.save()
        self._apply_env_vars()
    
    def set_comfyui_path(self, path: str) -> None:
        """
        Set the ComfyUI models directory.
        
        Args:
            path: Path to ComfyUI installation or models folder
        """
        self.config["comfyui_path"] = str(Path(path).resolve())
        self.save()
    
    def get_comfyui_path(self) -> Optional[Path]:
        """
        Get the ComfyUI models directory.
        
        Returns:
            Path to ComfyUI models, or None if not set
        """
        if self.config.get("comfyui_path"):
            return Path(self.config["comfyui_path"])
        return None
    
    def set_ollama_models(self, path: str) -> None:
        """
        Set the Ollama models directory.
        
        Args:
            path: Path to Ollama models folder
        """
        self.config["ollama_models"] = str(Path(path).resolve())
        self.save()
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        self.config = self._default_config()
        self.save()
        self._apply_env_vars()
    
    @property
    def config_file_path(self) -> Path:
        """Return the path to the configuration file."""
        return CONFIG_FILE


def get_config() -> ConfigManager:
    """
    Get the global ConfigManager instance.
    
    Returns:
        ConfigManager singleton instance
    """
    return ConfigManager()
