# src/model_bridge/decorator.py
"""
Decorator module for Model Bridge.

Contains the @smart_load decorator for automatic model path resolution.
This is the "magic" that makes Model Bridge seamless to use.

Example:
    @smart_load
    def load_model(model_id):
        # model_id will be automatically resolved to local path if available
        pass
        
    # User writes:
    load_model("qwen")
    
    # Actually executes as:
    load_model("D:/Models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
"""

import functools
import os
from typing import Callable, Any, Optional, Union, List
from pathlib import Path

from rich.console import Console

from .core import ModelRegistry

console = Console()

# 惰性初始化 Registry，避免 import 时就读硬盘
_REGISTRY: Optional[ModelRegistry] = None


def _get_registry() -> ModelRegistry:
    """Lazy initialization of the registry singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry()
    return _REGISTRY


# 常见的模型参数名称
MODEL_PARAM_NAMES: List[str] = [
    'pretrained_model_name_or_path',  # transformers
    'model_id',                       # diffusers
    'repo_id',                        # huggingface_hub
    'model_name',                     # general
    'model_path',                     # general
    'checkpoint',                     # pytorch
    'model',                          # ollama, general
    'ckpt_path',                      # pytorch lightning
]


def _is_likely_model_query(value: str) -> bool:
    """
    Check if a string looks like a model query (short name) vs a path.
    
    Returns True if it looks like a query (should be resolved).
    Returns False if it looks like a path (skip resolution).
    """
    if not isinstance(value, str):
        return False
    
    # Already an existing path
    if os.path.exists(value):
        return False
    
    # Contains path separators (likely a path)
    if "/" in value or "\\" in value:
        return False
    
    # Looks like a HuggingFace repo (org/model)
    # But we still want to try resolving it
    # if "/" in value and not os.path.isabs(value):
    #     return True
    
    return True


def smart_load(func: Callable) -> Callable:
    """
    Decorator: Automatically substitute model names with local paths.
    
    Works with transformers, diffusers, llama-cpp-python, and similar libraries.
    
    Workflow:
    1. Intercept: Capture the first argument (usually model_id)
    2. Query: Ask ModelRegistry if we have a local match
    3. Inject: If found, replace with local path; otherwise pass through
    4. Execute: Run the original function
    
    Example:
        @smart_load
        def load_model(model_id, device="cpu"):
            return MyModel.load(model_id, device)
        
        # This will auto-resolve "qwen" to the local path
        model = load_model("qwen", device="cuda")
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with automatic model resolution
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        registry = _get_registry()
        
        # === 1. Extract model name ===
        query_name: Optional[str] = None
        arg_type: Optional[str] = None  # 'pos' or 'kw'
        arg_index: int = -1
        kw_key: Optional[str] = None
        
        # Strategy A: Check first positional argument
        if args and isinstance(args[0], str):
            if _is_likely_model_query(args[0]):
                query_name = args[0]
                arg_type = 'pos'
                arg_index = 0
        
        # Strategy B: Check known keyword arguments
        if not query_name:
            for key in MODEL_PARAM_NAMES:
                if key in kwargs and isinstance(kwargs[key], str):
                    if _is_likely_model_query(kwargs[key]):
                        query_name = kwargs[key]
                        arg_type = 'kw'
                        kw_key = key
                        break
        
        # === 2. No model-like argument found -> pass through ===
        if not query_name:
            return func(*args, **kwargs)
        
        # === 3. Query the Registry ===
        match = registry.find(query_name)
        
        if match:
            local_path = match['path']
            model_id = match.get('id', 'unknown')
            
            console.print(
                f"[bold green]✨ SmartLoad:[/bold green] "
                f"'{query_name}' → [cyan]{model_id}[/cyan]"
            )
            console.print(f"   [dim]Path: {local_path}[/dim]")
            
            # === 4. Substitute the argument ===
            new_args = list(args)
            new_kwargs = kwargs.copy()
            
            if arg_type == 'pos':
                new_args[arg_index] = local_path
            elif arg_type == 'kw' and kw_key:
                new_kwargs[kw_key] = local_path
            
            # === 5. (Optional) Inject optimization hints ===
            # For HuggingFace libs, tell them not to check remote
            # Uncomment if you're sure your target functions support this
            # if 'local_files_only' not in new_kwargs:
            #     new_kwargs['local_files_only'] = True
            
            return func(*tuple(new_args), **new_kwargs)
        
        else:
            console.print(
                f"[dim]☁️ SmartLoad: '{query_name}' not found locally. "
                f"Passing through to remote.[/dim]"
            )
            return func(*args, **kwargs)
    
    return wrapper


def smart_load_v2(
    func: Optional[Callable] = None,
    *,
    param_name: Optional[str] = None,
    fallback: Optional[str] = None,
    strict: bool = False,
    silent: bool = False
) -> Callable:
    """
    Advanced decorator with more options.
    
    Args:
        func: The function to decorate (for @smart_load_v2 syntax)
        param_name: Specific parameter name to resolve (default: auto-detect)
        fallback: Default model to use if resolution fails
        strict: If True, raise error when model not found locally
        silent: If True, don't print status messages
        
    Returns:
        Decorated function
        
    Example:
        @smart_load_v2(param_name="weights", strict=True)
        def load_weights(weights: str, config: dict):
            pass
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            registry = _get_registry()
            
            # Determine which parameter to resolve
            target_params = [param_name] if param_name else MODEL_PARAM_NAMES
            
            query_name: Optional[str] = None
            arg_type: Optional[str] = None
            arg_index: int = -1
            kw_key: Optional[str] = None
            
            # Check positional args (first string arg)
            if args and isinstance(args[0], str) and not param_name:
                if _is_likely_model_query(args[0]):
                    query_name = args[0]
                    arg_type = 'pos'
                    arg_index = 0
            
            # Check kwargs
            if not query_name:
                for key in target_params:
                    if key in kwargs and isinstance(kwargs[key], str):
                        if _is_likely_model_query(kwargs[key]):
                            query_name = kwargs[key]
                            arg_type = 'kw'
                            kw_key = key
                            break
            
            # Use fallback if no query found
            if not query_name and fallback:
                query_name = fallback
                arg_type = 'fallback'
            
            if not query_name:
                return fn(*args, **kwargs)
            
            # Query registry
            match = registry.find(query_name)
            
            if match:
                local_path = match['path']
                
                if not silent:
                    console.print(
                        f"[bold green]✨ SmartLoad:[/bold green] "
                        f"'{query_name}' → [cyan]{match.get('id', 'unknown')}[/cyan]"
                    )
                
                new_args = list(args)
                new_kwargs = kwargs.copy()
                
                if arg_type == 'pos':
                    new_args[arg_index] = local_path
                elif arg_type == 'kw' and kw_key:
                    new_kwargs[kw_key] = local_path
                elif arg_type == 'fallback':
                    # Inject as first positional arg
                    new_args = [local_path] + list(args)
                
                return fn(*tuple(new_args), **new_kwargs)
            
            else:
                if strict:
                    raise ModelNotFoundError(
                        f"Model '{query_name}' not found in local registry. "
                        f"Run 'mb scan' to update the registry."
                    )
                
                if not silent:
                    console.print(
                        f"[dim]☁️ SmartLoad: '{query_name}' not found locally.[/dim]"
                    )
                
                return fn(*args, **kwargs)
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


class ModelNotFoundError(Exception):
    """Raised when a model cannot be found in the registry."""
    pass
