"""
Scanner execution engine.

Orchestrates all scanning strategies and provides a unified interface
for discovering models across all supported formats.
"""

from pathlib import Path
from typing import List, Dict, Optional, Type

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .base import ScanStrategy, ModelInfo
from .strategies import (
    HuggingFaceStrategy,
    GGUFStrategy,
    TensorRTStrategy,
    ComfyUIStrategy,
    SafetensorsStrategy,
)
from ..config import ConfigManager, get_config


console = Console()


class ScannerEngine:
    """
    Central scanning engine that coordinates all format strategies.
    
    The engine:
    1. Loads strategies (HF, GGUF, TensorRT, etc.)
    2. Reads search paths from configuration
    3. Runs all strategies and collects results
    4. Deduplicates and returns unified model list
    
    Example:
        >>> from model_bridge.config import get_config
        >>> from model_bridge.scanner import ScannerEngine
        >>> 
        >>> config = get_config()
        >>> engine = ScannerEngine(config)
        >>> models = engine.run()
    """
    
    # Default strategies to use
    DEFAULT_STRATEGIES: List[Type[ScanStrategy]] = [
        HuggingFaceStrategy,
        GGUFStrategy,
        TensorRTStrategy,
        ComfyUIStrategy,
        SafetensorsStrategy,
    ]
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        strategies: Optional[List[ScanStrategy]] = None,
    ):
        """
        Initialize the scanner engine.
        
        Args:
            config: Configuration manager instance (uses global if None)
            strategies: Custom list of strategies (uses defaults if None)
        """
        self.config = config or get_config()
        
        # Initialize strategies
        if strategies is not None:
            self.strategies = strategies
        else:
            self.strategies = [cls() for cls in self.DEFAULT_STRATEGIES]
    
    def run(self, verbose: bool = True) -> List[ModelInfo]:
        """
        Run all scanning strategies and return discovered models.
        
        Args:
            verbose: Whether to print progress to console
            
        Returns:
            List of all discovered models (deduplicated)
        """
        all_models: List[ModelInfo] = []
        
        # Get search paths from config
        search_paths = self.config.get_search_paths()
        
        if verbose:
            console.print(f"[bold blue]🔍 Scanning paths:[/bold blue]")
            for p in search_paths:
                console.print(f"   • {p}")
            console.print(f"   • [dim]+ Default HuggingFace Cache[/dim]")
            console.print()
        
        # Run each strategy
        if verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                for strategy in self.strategies:
                    task = progress.add_task(
                        f"Running {strategy.name} scanner...",
                        total=None,
                    )
                    
                    try:
                        found = strategy.scan(search_paths)
                        all_models.extend(found)
                        progress.update(
                            task,
                            description=f"[green]✓[/green] {strategy.name}: found {len(found)} models",
                        )
                    except Exception as e:
                        progress.update(
                            task,
                            description=f"[red]✗[/red] {strategy.name} failed: {e}",
                        )
        else:
            # Quiet mode
            for strategy in self.strategies:
                try:
                    found = strategy.scan(search_paths)
                    all_models.extend(found)
                except Exception:
                    pass
        
        # Deduplicate by path
        unique_models = self._deduplicate(all_models)
        
        if verbose:
            console.print()
            console.print(f"[bold green]✅ Found {len(unique_models)} unique models[/bold green]")
        
        return unique_models
    
    def run_single(self, strategy_name: str) -> List[ModelInfo]:
        """
        Run a single scanning strategy by name.
        
        Args:
            strategy_name: Name of the strategy to run
            
        Returns:
            List of models found by that strategy
            
        Raises:
            ValueError: If strategy name is not found
        """
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                search_paths = self.config.get_search_paths()
                return strategy.scan(search_paths)
        
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def _deduplicate(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """
        Remove duplicate models based on their paths.
        
        Args:
            models: List of models to deduplicate
            
        Returns:
            List of unique models
        """
        seen_paths: set = set()
        unique_models: List[ModelInfo] = []
        
        for model in models:
            if model.path not in seen_paths:
                seen_paths.add(model.path)
                unique_models.append(model)
        
        return unique_models
    
    def add_strategy(self, strategy: ScanStrategy) -> None:
        """
        Add a custom scanning strategy.
        
        Args:
            strategy: Strategy instance to add
        """
        self.strategies.append(strategy)
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove a strategy by name.
        
        Args:
            strategy_name: Name of strategy to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, strategy in enumerate(self.strategies):
            if strategy.name == strategy_name:
                self.strategies.pop(i)
                return True
        return False
    
    @property
    def strategy_names(self) -> List[str]:
        """Get list of all registered strategy names."""
        return [s.name for s in self.strategies]


def quick_scan(paths: Optional[List[str]] = None, verbose: bool = False) -> List[ModelInfo]:
    """
    Convenience function for quick scanning.
    
    Args:
        paths: Optional list of paths to scan (uses config if None)
        verbose: Whether to print progress
        
    Returns:
        List of discovered models
    """
    config = get_config()
    
    # Temporarily add paths if provided
    added_paths = []
    if paths:
        for p in paths:
            if p not in [str(x) for x in config.get_search_paths()]:
                config.add_search_path(p)
                added_paths.append(p)
    
    try:
        engine = ScannerEngine(config)
        return engine.run(verbose=verbose)
    finally:
        # Clean up temporary paths
        for p in added_paths:
            config.remove_search_path(p)
