"""
Concrete scanning strategies for different model formats.

This module contains implementations for:
- HuggingFaceStrategy: HF cache and local transformer models
- GGUFStrategy: GGUF files for llama.cpp/Ollama
- TensorRTStrategy: TensorRT-LLM compiled engines
- ComfyUIStrategy: ComfyUI model directories
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import ScanStrategy, ModelInfo, ModelType
from ..metadata import ModelParser


# ============================================================
# Scan Utilities
# ============================================================

class ScanUtils:
    """
    Utility class for common scanning operations.
    """
    
    # Pattern: model-00002-of-00005.safetensors or .gguf
    RE_SHARD_OF = re.compile(r'[-_](\d+)-of-(\d+)')
    
    # Pattern: model.part2.gguf
    RE_PART = re.compile(r'\.part(\d+)\.')
    
    @classmethod
    def is_follower_shard(cls, filename: str) -> bool:
        """
        Check if a file is a follower shard (not the primary file).
        
        We keep only the first shard (part 1 or 00001) as the entry point.
        
        Examples:
            model-00001-of-00005.safetensors -> False (keep as entry)
            model-00002-of-00005.safetensors -> True (skip)
            model.part1.gguf -> False (keep)
            model.part2.gguf -> True (skip)
            regular-model.gguf -> False (keep)
        
        Args:
            filename: The filename to check
            
        Returns:
            True if this is a follower shard that should be skipped
        """
        # Check for pattern: -00002-of-00005
        shard_match = cls.RE_SHARD_OF.search(filename)
        if shard_match:
            current_part = int(shard_match.group(1))
            return current_part > 1
        
        # Check for pattern: .part2.
        part_match = cls.RE_PART.search(filename)
        if part_match:
            return int(part_match.group(1)) > 1
        
        return False
    
    @classmethod
    def get_shard_base_name(cls, filename: str) -> str:
        """
        Get the base name for a sharded model (without shard suffix).
        
        This helps identify related shards.
        
        Args:
            filename: The filename to process
            
        Returns:
            Base name without shard indicators
        """
        # Remove -00001-of-00005 pattern
        name = cls.RE_SHARD_OF.sub('', filename)
        # Remove .part1 pattern
        name = cls.RE_PART.sub('.', name)
        return name
    
    @classmethod
    def count_shards(cls, model_path: Path) -> int:
        """
        Count the total number of shards for a model.
        
        Args:
            model_path: Path to the primary shard file
            
        Returns:
            Number of shard files found
        """
        if not model_path.exists():
            return 0
        
        base_name = cls.get_shard_base_name(model_path.name)
        parent_dir = model_path.parent
        
        count = 0
        for f in parent_dir.iterdir():
            if f.is_file() and cls.get_shard_base_name(f.name) == base_name:
                if f.suffix == model_path.suffix:
                    count += 1
        
        return count if count > 1 else 1


class HuggingFaceStrategy(ScanStrategy):
    """
    Scanner for HuggingFace models.
    
    Supports:
    - Official HuggingFace cache (using huggingface_hub.scan_cache_dir)
    - Local unpacked HF models (directories with config.json)
    
    Engine support: transformers, vllm, sglang
    """
    
    @property
    def name(self) -> str:
        return "huggingface"
    
    def scan(self, paths: List[Path]) -> List[ModelInfo]:
        """
        Scan HuggingFace cache and custom paths for models.
        
        Args:
            paths: Additional paths to scan for local HF models
            
        Returns:
            List of discovered HuggingFace models
        """
        models = []
        
        # 1. Scan official HuggingFace cache
        models.extend(self._scan_hf_cache())
        
        # 2. Scan custom paths for local HF models
        for root_path in paths:
            if not root_path.exists():
                continue
            models.extend(self._scan_local_hf_models(root_path))
        
        return models
    
    def _scan_hf_cache(self) -> List[ModelInfo]:
        """
        Scan the official HuggingFace cache directory.
        
        Returns:
            List of models found in HF cache
        """
        models = []
        
        try:
            from huggingface_hub import scan_cache_dir
            
            hf_cache_info = scan_cache_dir()
            
            for repo in hf_cache_info.repos:
                for revision in repo.revisions:
                    # Parse semantic metadata from repo_id
                    # repo_id format: "organization/model-name"
                    repo_name = repo.repo_id.split("/")[-1] if "/" in repo.repo_id else repo.repo_id
                    parsed_meta = ModelParser.parse(repo_name, "")
                    
                    # Add HF-specific metadata
                    parsed_meta["revision"] = revision.commit_hash
                    parsed_meta["repo_type"] = repo.repo_type
                    parsed_meta["last_accessed"] = revision.last_accessed
                    parsed_meta["repo_id"] = repo.repo_id
                    
                    # Add huggingface to keywords
                    keywords = list(parsed_meta.get("keywords", []))
                    if "huggingface" not in keywords:
                        keywords.append("huggingface")
                    parsed_meta["keywords"] = sorted(keywords)
                    
                    model = ModelInfo(
                        id=repo.repo_id,
                        path=str(revision.snapshot_path),
                        type=ModelType.HUGGINGFACE.value,
                        engine_support=["transformers", "vllm", "sglang"],
                        metadata=parsed_meta,
                        size_bytes=revision.size_on_disk,
                    )
                    models.append(model)
                    
        except ImportError:
            # huggingface_hub not installed
            pass
        except Exception:
            # Cache directory doesn't exist or other error
            pass
        
        return models
    
    def _scan_local_hf_models(self, root_path: Path) -> List[ModelInfo]:
        """
        Scan for local unpacked HuggingFace models.
        
        A directory is considered an HF model if it contains config.json.
        
        Args:
            root_path: Root directory to scan
            
        Returns:
            List of local HF models found
        """
        models = []
        
        # Find directories containing config.json (HF model marker)
        for config_file in root_path.rglob("config.json"):
            model_dir = config_file.parent
            
            # Skip if this is inside HF cache structure (already scanned)
            if ".cache/huggingface" in str(model_dir):
                continue
            
            # Try to extract model name from config
            model_name = self._extract_model_name(model_dir)
            
            # Parse semantic metadata from name
            parsed_meta = ModelParser.parse(model_name, model_dir.parent.name)
            
            model = ModelInfo(
                id=model_name,
                path=str(model_dir),
                type=ModelType.HUGGINGFACE_LOCAL.value,
                engine_support=["transformers", "vllm", "sglang"],
                metadata=parsed_meta,
                size_bytes=self.get_model_size(model_dir),
            )
            models.append(model)
        
        return models
    
    def _extract_model_name(self, model_dir: Path) -> str:
        """
        Extract model name from directory or config.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            Model name/identifier
        """
        try:
            import json
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                # Try common fields
                if "_name_or_path" in config:
                    return config["_name_or_path"]
        except Exception:
            pass
        
        # Fall back to directory name
        return model_dir.name


class GGUFStrategy(ScanStrategy):
    """
    Scanner for GGUF model files.
    
    GGUF is the format used by llama.cpp and Ollama.
    Parses GGUF headers to extract architecture and quantization info.
    
    Engine support: llama.cpp, ollama
    """
    
    @property
    def name(self) -> str:
        return "gguf"
    
    def scan(self, paths: List[Path]) -> List[ModelInfo]:
        """
        Scan directories for GGUF files.
        
        Filters out follower shards (keeps only the first shard as entry point).
        
        Args:
            paths: Directories to scan recursively
            
        Returns:
            List of discovered GGUF models
        """
        models = []
        
        for root_path in paths:
            if not root_path.exists():
                continue
            
            # Recursively find .gguf files
            for gguf_file in root_path.rglob("*.gguf"):
                # Skip follower shards (00002-of-00005, part2, etc.)
                if ScanUtils.is_follower_shard(gguf_file.name):
                    continue
                
                try:
                    model = self._parse_gguf(gguf_file)
                    if model:
                        models.append(model)
                except Exception as e:
                    # Log but don't crash on broken files
                    print(f"⚠️ Skipping broken GGUF {gguf_file}: {e}")
        
        return models
    
    def _get_shard_total_size(self, primary_file: Path) -> tuple:
        """
        Calculate total size of all shards for a model.
        
        Args:
            primary_file: Path to the primary shard (00001-of-xxxx)
            
        Returns:
            Tuple of (total_size_bytes, shard_count)
        """
        filename = primary_file.name
        parent_dir = primary_file.parent
        
        # Check if this is a sharded model
        shard_match = ScanUtils.RE_SHARD_OF.search(filename)
        if shard_match:
            # Get the base pattern to find all shards
            # e.g., qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf
            # -> find all files matching qwen2.5-7b-instruct-q5_k_m-*-of-*.gguf
            base_pattern = ScanUtils.RE_SHARD_OF.sub('', filename)
            total_shards = int(shard_match.group(2))
            
            total_size = 0
            found_count = 0
            
            for f in parent_dir.iterdir():
                if f.is_file() and f.suffix == '.gguf':
                    f_base = ScanUtils.RE_SHARD_OF.sub('', f.name)
                    if f_base == base_pattern:
                        total_size += f.stat().st_size
                        found_count += 1
            
            return total_size, found_count
        
        # Check for .part1. pattern
        part_match = ScanUtils.RE_PART.search(filename)
        if part_match:
            base_pattern = ScanUtils.RE_PART.sub('.', filename)
            
            total_size = 0
            found_count = 0
            
            for f in parent_dir.iterdir():
                if f.is_file() and f.suffix == primary_file.suffix:
                    f_base = ScanUtils.RE_PART.sub('.', f.name)
                    if f_base == base_pattern:
                        total_size += f.stat().st_size
                        found_count += 1
            
            return total_size, found_count
        
        # Not a sharded model
        return primary_file.stat().st_size, 1
    
    def _parse_gguf(self, gguf_file: Path) -> Optional[ModelInfo]:
        """
        Parse a GGUF file and extract metadata.
        
        Args:
            gguf_file: Path to GGUF file
            
        Returns:
            ModelInfo if successful, None otherwise
        """
        # Parse semantic metadata from filename first
        parsed_meta = ModelParser.parse(gguf_file.name, gguf_file.parent.name)
        metadata: Dict[str, Any] = dict(parsed_meta)  # Copy to avoid mutation
        arch = "unknown"
        
        try:
            from gguf import GGUFReader
            
            reader = GGUFReader(str(gguf_file), mode="r")
            
            # Extract architecture
            if "general.architecture" in reader.fields:
                val = reader.fields["general.architecture"].parts[-1]
                if isinstance(val, (bytes, bytearray)):
                    arch = bytes(val).decode("utf-8")
                else:
                    arch = str(val)
                metadata["architecture"] = arch
            
            # Extract model name if available
            if "general.name" in reader.fields:
                val = reader.fields["general.name"].parts[-1]
                if isinstance(val, (bytes, bytearray)):
                    metadata["name"] = bytes(val).decode("utf-8")
            
            # Extract quantization type if available
            if "general.file_type" in reader.fields:
                metadata["file_type"] = int(reader.fields["general.file_type"].parts[-1])
                
        except ImportError:
            # gguf library not installed - just use file info
            pass
        except Exception:
            # Parsing failed - still record the file
            pass
        
        # Calculate total size (including all shards)
        total_size, shard_count = self._get_shard_total_size(gguf_file)
        
        if shard_count > 1:
            metadata["shards"] = shard_count
        
        # Ensure format is set to gguf
        metadata["format"] = "gguf"
        if "gguf" not in metadata.get("keywords", []):
            keywords = list(metadata.get("keywords", []))
            keywords.append("gguf")
            metadata["keywords"] = sorted(keywords)
        
        return ModelInfo(
            id=gguf_file.stem,
            path=str(gguf_file),
            type=ModelType.GGUF.value,
            engine_support=["llama.cpp", "ollama"],
            metadata=metadata,
            size_bytes=total_size,
        )


class TensorRTStrategy(ScanStrategy):
    """
    Scanner for TensorRT-LLM compiled engines.
    
    TensorRT-LLM compiles models to .engine or .plan files for
    optimized inference on NVIDIA GPUs.
    
    Engine support: tensorrt_llm, triton
    """
    
    @property
    def name(self) -> str:
        return "tensorrt"
    
    def scan(self, paths: List[Path]) -> List[ModelInfo]:
        """
        Scan directories for TensorRT engine files.
        
        Args:
            paths: Directories to scan
            
        Returns:
            List of discovered TensorRT models
        """
        models = []
        
        for root_path in paths:
            if not root_path.exists():
                continue
            
            # Find .engine files
            for engine_file in root_path.rglob("*.engine"):
                model = self._parse_engine(engine_file)
                models.append(model)
            
            # Find .plan files (alternative TensorRT format)
            for plan_file in root_path.rglob("*.plan"):
                model = self._parse_engine(plan_file)
                models.append(model)
        
        return models
    
    def _parse_engine(self, engine_file: Path) -> ModelInfo:
        """
        Parse a TensorRT engine file.
        
        Args:
            engine_file: Path to engine file
            
        Returns:
            ModelInfo for the engine
        """
        return ModelInfo(
            id=engine_file.stem,
            path=str(engine_file),
            type=ModelType.TENSORRT.value,
            engine_support=["tensorrt_llm", "triton"],
            metadata={
                "format": engine_file.suffix.lstrip("."),
            },
            size_bytes=engine_file.stat().st_size,
        )


class ComfyUIStrategy(ScanStrategy):
    """
    Scanner for ComfyUI model directories.
    
    Scans the standard ComfyUI models folder structure:
    - checkpoints/
    - loras/
    - vae/
    - embeddings/
    - controlnet/
    - upscale_models/
    
    Engine support: comfyui, diffusers
    """
    
    # Standard ComfyUI model subdirectories
    SUBDIRS = {
        "checkpoints": ModelType.COMFYUI_CHECKPOINT,
        "loras": ModelType.COMFYUI_LORA,
        "vae": ModelType.COMFYUI_VAE,
        "embeddings": ModelType.SAFETENSORS,
        "controlnet": ModelType.SAFETENSORS,
        "upscale_models": ModelType.SAFETENSORS,
    }
    
    # Valid model file extensions
    EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
    
    @property
    def name(self) -> str:
        return "comfyui"
    
    def scan(self, paths: List[Path]) -> List[ModelInfo]:
        """
        Scan ComfyUI model directories.
        
        Args:
            paths: Paths to ComfyUI installations or model folders
            
        Returns:
            List of discovered ComfyUI models
        """
        models = []
        
        for root_path in paths:
            if not root_path.exists():
                continue
            
            # Check if this is a ComfyUI root or models folder
            models_path = root_path / "models"
            if not models_path.exists():
                models_path = root_path
            
            # Scan each standard subdirectory
            for subdir, model_type in self.SUBDIRS.items():
                subdir_path = models_path / subdir
                if not subdir_path.exists():
                    continue
                
                for model_file in subdir_path.rglob("*"):
                    if model_file.is_file() and model_file.suffix.lower() in self.EXTENSIONS:
                        # Parse semantic metadata from filename
                        parsed_meta = ModelParser.parse(model_file.name, model_file.parent.name)
                        parsed_meta["category"] = subdir
                        parsed_meta["extension"] = model_file.suffix.lower()
                        
                        # Add comfyui to keywords
                        keywords = list(parsed_meta.get("keywords", []))
                        if "comfyui" not in keywords:
                            keywords.append("comfyui")
                        if subdir not in keywords:
                            keywords.append(subdir)
                        parsed_meta["keywords"] = sorted(keywords)
                        
                        model = ModelInfo(
                            id=model_file.stem,
                            path=str(model_file),
                            type=model_type.value,
                            engine_support=["comfyui", "diffusers"],
                            metadata=parsed_meta,
                            size_bytes=model_file.stat().st_size,
                        )
                        models.append(model)
        
        return models


class SafetensorsStrategy(ScanStrategy):
    """
    Generic scanner for Safetensors files.
    
    Used as a fallback for .safetensors files that aren't in
    a recognized directory structure.
    
    Engine support: transformers, diffusers
    """
    
    @property
    def name(self) -> str:
        return "safetensors"
    
    def scan(self, paths: List[Path]) -> List[ModelInfo]:
        """
        Scan directories for standalone Safetensors files.
        
        Filters out follower shards and files inside HF model directories.
        
        Args:
            paths: Directories to scan
            
        Returns:
            List of discovered Safetensors models
        """
        models = []
        
        for root_path in paths:
            if not root_path.exists():
                continue
            
            for st_file in root_path.rglob("*.safetensors"):
                # Skip files that are likely part of a larger model
                # (e.g., inside an HF model directory)
                if (st_file.parent / "config.json").exists():
                    continue
                
                # Skip follower shards (00002-of-00005, etc.)
                if ScanUtils.is_follower_shard(st_file.name):
                    continue
                
                # Calculate total size including all shards
                total_size, shard_count = self._get_shard_total_size(st_file)
                
                # Parse semantic metadata from filename
                parsed_meta = ModelParser.parse(st_file.name, st_file.parent.name)
                
                if shard_count > 1:
                    parsed_meta["shards"] = shard_count
                
                # Ensure format is set to safetensors
                parsed_meta["format"] = "safetensors"
                if "safetensors" not in parsed_meta.get("keywords", []):
                    keywords = list(parsed_meta.get("keywords", []))
                    keywords.append("safetensors")
                    parsed_meta["keywords"] = sorted(keywords)
                
                model = ModelInfo(
                    id=st_file.stem,
                    path=str(st_file),
                    type=ModelType.SAFETENSORS.value,
                    engine_support=["transformers", "diffusers"],
                    metadata=parsed_meta,
                    size_bytes=total_size,
                )
                models.append(model)
        
        return models
    
    def _get_shard_total_size(self, primary_file: Path) -> tuple:
        """
        Calculate total size of all shards for a safetensors model.
        
        Args:
            primary_file: Path to the primary shard (00001-of-xxxx)
            
        Returns:
            Tuple of (total_size_bytes, shard_count)
        """
        filename = primary_file.name
        parent_dir = primary_file.parent
        
        # Check if this is a sharded model (pattern: -00001-of-00005)
        shard_match = ScanUtils.RE_SHARD_OF.search(filename)
        if shard_match:
            base_pattern = ScanUtils.RE_SHARD_OF.sub('', filename)
            
            total_size = 0
            found_count = 0
            
            for f in parent_dir.iterdir():
                if f.is_file() and f.suffix == '.safetensors':
                    f_base = ScanUtils.RE_SHARD_OF.sub('', f.name)
                    if f_base == base_pattern:
                        total_size += f.stat().st_size
                        found_count += 1
            
            return total_size, found_count
        
        # Not a sharded model
        return primary_file.stat().st_size, 1
