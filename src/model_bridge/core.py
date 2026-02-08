"""
Core module for Model Bridge.

Contains the ModelRegistry singleton - the "brain" of Model Bridge.
Responsible for:
- Persistence: Store/load model index to/from JSON
- Indexing & Query: Fast model lookup with fuzzy matching
- Singleton: Ensure single instance across the application
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from rich.console import Console

from .config import ConfigManager, get_config, CONFIG_DIR
from .scanner.engine import ScannerEngine
from .scanner.base import ModelInfo
from .ranker import ModelRanker


console = Console()


class ModelRegistry:
    """
    Central registry for managing model index.
    
    This is the "brain" of Model Bridge:
    - Loads/saves model index from/to JSON for fast access
    - Provides fuzzy search capabilities
    - Integrates with ScannerEngine for discovery
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.scan()  # Discover models
        >>> model = registry.find("qwen")  # Search
        >>> path = registry.get_path("llama-3")  # Get path directly
    """
    
    _instance: Optional["ModelRegistry"] = None
    
    def __new__(cls) -> "ModelRegistry":
        """Singleton pattern - ensure only one registry exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config = get_config()
        self.registry_path = CONFIG_DIR / "registry.json"
        self.models: List[Dict[str, Any]] = []
        self.last_scan: float = 0
        self._initialized = True
        
        # Load existing registry
        self._load()
    
    def _load(self) -> None:
        """
        Load model index from JSON file.
        
        If file doesn't exist or is corrupted, initializes empty registry.
        """
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.models = data.get("models", [])
                    self.last_scan = data.get("last_scan", 0)
                    
                    # Check if scan is outdated (> 7 days)
                    days_since_scan = (time.time() - self.last_scan) / 86400
                    if days_since_scan > 7 and self.models:
                        console.print(
                            f"[yellow]⚠️ Registry is {days_since_scan:.0f} days old. "
                            f"Consider running 'mb scan' to refresh.[/yellow]"
                        )
            except (json.JSONDecodeError, IOError, KeyError) as e:
                console.print(f"[red]⚠️ Failed to load registry: {e}[/red]")
                self.models = []
                self.last_scan = 0
        else:
            self.models = []
            self.last_scan = 0
    
    def save(self) -> None:
        """
        Save model index to JSON file.
        
        Stores models list, timestamp, and count for quick access.
        """
        data = {
            "last_scan": time.time(),
            "count": len(self.models),
            "models": self.models,
        }
        
        try:
            # Ensure directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            console.print(f"[dim]Registry saved to {self.registry_path}[/dim]")
        except (IOError, OSError) as e:
            console.print(f"[red]❌ Failed to save registry: {e}[/red]")
    
    def scan(self, force_refresh: bool = False, verbose: bool = True) -> Dict[str, int]:
        """
        Execute model scan using ScannerEngine.
        
        By default uses incremental mode (only adds new models).
        Use force_refresh=True to rebuild from scratch.
        
        Args:
            force_refresh: If True, replace entire index. If False (default), 
                          only add models not already in registry.
            verbose: Whether to print progress
            
        Returns:
            Dictionary with scan statistics:
            - 'scanned': Total models found during scan
            - 'added': New models added to registry
            - 'skipped': Models already in registry (skipped)
            - 'total': Final registry count
        """
        scanner = ScannerEngine(self.config)
        
        if verbose:
            mode_str = "full refresh" if force_refresh else "incremental"
            console.print(f"[green]🔄 Scanning models ({mode_str})...[/green]")
        
        # Run scanner (returns List[ModelInfo])
        new_models = scanner.run(verbose=verbose)
        
        # Convert ModelInfo objects to dictionaries
        model_dicts = [m.to_dict() for m in new_models]
        
        stats = {
            'scanned': len(model_dicts),
            'added': 0,
            'skipped': 0,
            'removed': 0,
            'total': 0,
        }
        
        if force_refresh:
            # Full refresh: replace everything
            stats['added'] = len(model_dicts)
            stats['removed'] = len(self.models)
            self.models = model_dicts
        else:
            # Incremental update: only add new models
            # Build index of existing models by path AND id for fast lookup
            existing_paths = {m["path"] for m in self.models}
            existing_ids = {m["id"] for m in self.models}
            
            for model in model_dicts:
                # Skip if path already exists
                if model["path"] in existing_paths:
                    stats['skipped'] += 1
                    continue
                
                # Skip if same model ID exists (same model, different path)
                # This handles cases where model was moved
                if model["id"] in existing_ids:
                    stats['skipped'] += 1
                    continue
                
                self.models.append(model)
                stats['added'] += 1
        
        stats['total'] = len(self.models)
        self.last_scan = time.time()
        self.save()
        
        if verbose:
            console.print(
                f"[bold green]✅ Scan complete![/bold green] "
                f"Found {stats['scanned']}, Added {stats['added']}, "
                f"Skipped {stats['skipped']}, Total: {stats['total']}"
            )
        
        return stats
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return all models, optionally filtered by type.
        
        Args:
            model_type: Filter by type (e.g., "gguf", "hf", "safetensors")
            
        Returns:
            List of model dictionaries
        """
        if model_type is None:
            return self.models
        
        return [m for m in self.models if m.get("type") == model_type]
    
    def find(self, query: str, top_k: int = 1, 
             min_quant: Optional[str] = None) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Find model(s) using intelligent weighted fuzzy matching.
        
        Uses ModelRanker for layered scoring:
        1. Hard filters: Size (7B/72B), MoE, Format, min_quant requirements
        2. Token matching: Query words in model name
        3. Keyword matching: From parsed metadata
        4. Heuristic bonuses: Instruct preference, version, quantization
        
        Args:
            query: Search term (e.g., "qwen 7b instruct")
            top_k: Number of results to return
                   - top_k=1: Returns single best match (or None)
                   - top_k>1: Returns list of top matches
            min_quant: Minimum quantization level (e.g., "q4" for Q4 or above)
            
        Returns:
            If top_k=1: Best matching model dict, or None
            If top_k>1: List of top matching models
        """
        if not query or not query.strip():
            return None if top_k == 1 else []
        
        # Use ModelRanker for intelligent sorting
        sorted_models = ModelRanker.rank(query, self.models, min_quant=min_quant)
        
        if not sorted_models:
            return None if top_k == 1 else []
        
        if top_k == 1:
            return sorted_models[0]
        else:
            return sorted_models[:top_k]
    
    def find_all(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find all models matching query using ModelRanker.
        
        Args:
            query: Search term
            limit: Maximum results to return
            
        Returns:
            List of matching models, sorted by relevance
        """
        if not query or not query.strip():
            return []
        
        # Use ModelRanker for intelligent sorting
        sorted_models = ModelRanker.rank(query, self.models)
        
        return sorted_models[:limit]
    
    def get_path(self, query: str) -> Optional[str]:
        """
        Convenience method: Get model path by query.
        
        Args:
            query: Model name/ID to search for
            
        Returns:
            Absolute path to model, or None if not found
        """
        model = self.find(query)
        return model.get("path") if model else None
    
    def get_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get all models of a specific type.
        
        Args:
            model_type: Type to filter by (gguf, hf, hf_local, etc.)
            
        Returns:
            List of matching models
        """
        return [m for m in self.models if m.get("type") == model_type]
    
    def get_by_engine(self, engine: str) -> List[Dict[str, Any]]:
        """
        Get all models supporting a specific inference engine.
        
        Args:
            engine: Engine name (vllm, transformers, llama.cpp, etc.)
            
        Returns:
            List of compatible models
        """
        return [
            m for m in self.models
            if engine in m.get("engine_support", [])
        ]
    
    def remove(self, query: str) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            query: Model ID or path to remove
            
        Returns:
            True if removed, False if not found
        """
        query_lower = query.lower()
        
        for i, model in enumerate(self.models):
            if (model.get("id", "").lower() == query_lower or
                model.get("path", "").lower() == query_lower):
                self.models.pop(i)
                return True
        
        return False
    
    def clear(self) -> None:
        """Clear all models from the registry."""
        self.models = []
        self.last_scan = 0
    
    @property
    def count(self) -> int:
        """Return total number of registered models."""
        return len(self.models)
    
    @property
    def is_empty(self) -> bool:
        """Check if registry is empty."""
        return len(self.models) == 0
    
    def stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with count by type, total size, etc.
        """
        type_counts: Dict[str, int] = {}
        total_size = 0
        
        for model in self.models:
            model_type = model.get("type", "unknown")
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
            total_size += model.get("size_bytes", 0) or 0
        
        return {
            "total_models": len(self.models),
            "by_type": type_counts,
            "total_size_gb": total_size / (1024 ** 3),
            "last_scan": self.last_scan,
            "registry_path": str(self.registry_path),
        }


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """
    Get the global ModelRegistry instance.
    
    Returns:
        ModelRegistry singleton
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def find_model(query: str) -> Optional[str]:
    """
    Convenience function: Find a model and return its path.
    
    This is the main entry point for @smart_load decorator.
    
    Args:
        query: Model name/ID to search for
        
        
    Returns:
        Path to model, or None if not found
    """
    registry = get_registry()
    return registry.get_path(query)


# Backwards compatibility alias
Registry = ModelRegistry
