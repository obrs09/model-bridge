"""
Utility functions for Model Bridge.

Contains helper functions for file operations, hashing, and more.
"""

import hashlib
from pathlib import Path
from typing import Optional, Generator
import os


def calculate_hash(file_path: Path, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)
        chunk_size: Size of chunks to read (default: 8KB)
        
    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def calculate_partial_hash(
    file_path: Path, 
    algorithm: str = "sha256",
    head_bytes: int = 1024 * 1024,  # 1MB
    tail_bytes: int = 1024 * 1024   # 1MB
) -> str:
    """
    Calculate a partial hash using head and tail of file.
    
    Useful for large files where full hashing would be slow.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use
        head_bytes: Number of bytes to read from start
        tail_bytes: Number of bytes to read from end
        
    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)
    file_size = file_path.stat().st_size
    
    with open(file_path, "rb") as f:
        # Read head
        hasher.update(f.read(head_bytes))
        
        # Read tail if file is large enough
        if file_size > head_bytes + tail_bytes:
            f.seek(-tail_bytes, 2)  # Seek from end
            hasher.update(f.read(tail_bytes))
        
        # Include file size in hash
        hasher.update(str(file_size).encode())
    
    return hasher.hexdigest()


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string (e.g., "4.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_file_extension(file_path: Path) -> str:
    """Get lowercase file extension without the dot."""
    return file_path.suffix.lower().lstrip(".")


def is_model_file(file_path: Path) -> bool:
    """
    Check if a file is a known model format.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file is a known model format
    """
    model_extensions = {
        "gguf", "safetensors", "ckpt", "pt", "pth", "bin", "onnx"
    }
    return get_file_extension(file_path) in model_extensions


def walk_models(root_path: Path) -> Generator[Path, None, None]:
    """
    Walk a directory tree and yield model files.
    
    Args:
        root_path: Root directory to walk
        
    Yields:
        Paths to model files
    """
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if is_model_file(file_path):
                yield file_path


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_path(path_str: str) -> Optional[Path]:
    """
    Safely convert a string to a Path, returning None on error.
    
    Args:
        path_str: String path to convert
        
    Returns:
        Path object or None if invalid
    """
    try:
        return Path(path_str).resolve()
    except (ValueError, OSError):
        return None
