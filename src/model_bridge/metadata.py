# src/model_bridge/metadata.py
"""
Metadata extraction and parsing for AI models.

This module provides structured parsing of model filenames and paths,
extracting semantic tags like family, size, quantization, architecture, etc.

The keyword system follows a priority hierarchy:
1. Model Family (qwen, llama, deepseek, mistral, etc.)
2. Sub-family/Version (qwen2.5, llama3.1, etc.)
3. Architecture/Variant (instruct, chat, base, vision, etc.)
4. Quantization/Format (gguf, q4_k_m, awq, etc.)
5. Parameter Size (7b, 70b, 1.8b, etc.)
6. Version/Special Tags (v1, turbo, 128k, moe, etc.)
7. Domain/Language (chinese, coding, roleplay, etc.)
8. Source/Platform (huggingface, ollama, thebloke, etc.)
"""

import re
from typing import Dict, Any, List, Optional, Set
from pathlib import Path


class ModelParser:
    """
    Parser for extracting structured metadata from model names.
    
    Example:
        >>> meta = ModelParser.parse("Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        >>> meta['family']  # 'qwen'
        >>> meta['size']    # 7.0
        >>> meta['quant']   # 'q4_k_m'
        >>> meta['is_instruct']  # True
    """
    
    # ================================================================
    # Quantization Level Mapping (higher = more precision)
    # ================================================================
    QUANT_LEVELS: Dict[str, int] = {
        # Ultra-low precision
        "IQ1": 10, "IQ2": 15,
        # Low precision  
        "Q2": 20, "Q3": 30,
        # Medium precision (sweet spot for most use cases)
        "Q4": 40, "Q5": 50, "Q6": 60,
        # High precision
        "Q8": 80,
        # Full precision
        "FP16": 160, "BF16": 160, "F16": 160,
        "FP32": 320, "F32": 320,
        # Special quantizations
        "NF4": 45, "INT4": 40, "INT8": 80,
        "GPTQ": 45, "AWQ": 45, "EXL2": 50,
    }
    
    # ================================================================
    # Model Family Patterns
    # ================================================================
    FAMILY_PATTERNS: Dict[str, List[str]] = {
        # LLM Families
        "qwen": ["qwen"],
        "llama": ["llama", "meta-llama"],
        "deepseek": ["deepseek"],
        "mistral": ["mistral", "mixtral"],
        "phi": ["phi-", "phi2", "phi3", "phi4"],
        "gemma": ["gemma"],
        "yi": ["yi-", "01-ai"],
        "internlm": ["internlm"],
        "baichuan": ["baichuan"],
        "chatglm": ["chatglm", "glm-"],
        "falcon": ["falcon"],
        "mpt": ["mpt-"],
        "bloom": ["bloom"],
        "vicuna": ["vicuna"],
        "alpaca": ["alpaca"],
        "wizardlm": ["wizardlm", "wizard-"],
        "openchat": ["openchat"],
        "neural-chat": ["neural-chat"],
        "starling": ["starling"],
        "zephyr": ["zephyr"],
        "solar": ["solar"],
        "command": ["command-r", "c4ai"],
        
        # Vision/Multimodal
        "llava": ["llava"],
        "qwen-vl": ["qwen-vl", "qwen2-vl", "qwen3-vl"],
        "cogvlm": ["cogvlm"],
        "internvl": ["internvl"],
        "minicpm-v": ["minicpm-v"],
        
        # Diffusion Models
        "sdxl": ["sdxl", "stable-diffusion-xl"],
        "sd": ["stable-diffusion", "sd-", "sd1", "sd2"],
        "flux": ["flux"],
        "playground": ["playground"],
        "kandinsky": ["kandinsky"],
        
        # Audio/Speech
        "whisper": ["whisper"],
        "sovits": ["sovits", "gpt-sovits"],
        "bark": ["bark"],
        "tortoise": ["tortoise"],
        "valle": ["valle"],
        "xtts": ["xtts"],
        
        # Embedding/Encoder
        "clip": ["clip"],
        "bge": ["bge-"],
        "e5": ["e5-"],
        "gte": ["gte-"],
        "nomic": ["nomic-embed"],
        "jina": ["jina-"],
        
        # Code Models
        "codellama": ["codellama", "code-llama"],
        "starcoder": ["starcoder"],
        "deepseek-coder": ["deepseek-coder"],
        "codestral": ["codestral"],
        "qwen-coder": ["qwen2.5-coder", "qwen-coder"],
    }
    
    # Sub-family patterns (more specific versions)
    SUBFAMILY_PATTERNS: Dict[str, str] = {
        # Qwen versions
        "qwen3": "qwen3",
        "qwen2.5": "qwen2.5",
        "qwen2": "qwen2",
        "qwen1.5": "qwen1.5",
        "qwen-vl": "qwen-vl",
        
        # Llama versions
        "llama-3.3": "llama3.3",
        "llama-3.2": "llama3.2",
        "llama-3.1": "llama3.1",
        "llama-3": "llama3",
        "llama-2": "llama2",
        "llama3.3": "llama3.3",
        "llama3.2": "llama3.2",
        "llama3.1": "llama3.1",
        "llama3": "llama3",
        "llama2": "llama2",
        
        # Mistral versions
        "mixtral": "mixtral",
        "mistral-nemo": "mistral-nemo",
        "mistral-small": "mistral-small",
        "mistral-large": "mistral-large",
        
        # DeepSeek versions
        "deepseek-v3": "deepseek-v3",
        "deepseek-v2.5": "deepseek-v2.5",
        "deepseek-v2": "deepseek-v2",
        "deepseek-coder-v2": "deepseek-coder-v2",
        
        # Phi versions
        "phi-4": "phi4",
        "phi-3.5": "phi3.5",
        "phi-3": "phi3",
        "phi-2": "phi2",
        
        # Gemma versions
        "gemma-2": "gemma2",
        "gemma2": "gemma2",
        
        # SD versions
        "sdxl-turbo": "sdxl-turbo",
        "sd-turbo": "sd-turbo",
        "sd3.5": "sd3.5",
        "sd3": "sd3",
    }
    
    # ================================================================
    # Architecture/Variant Tags
    # ================================================================
    ARCHITECTURE_TAGS: Dict[str, List[str]] = {
        # Text generation variants
        "instruct": ["instruct", "-it-", "-it."],
        "chat": ["chat"],
        "base": ["base"],
        "aligned": ["aligned", "rlhf", "dpo", "sft"],
        
        # Vision/Multimodal
        "vision": ["vision", "-vl-", "-vl.", "vl-"],
        "multimodal": ["multimodal", "mm-", "omni"],
        
        # Audio
        "tts": ["tts", "text-to-speech"],
        "asr": ["asr", "speech-to-text", "stt"],
        "audio": ["audio"],
        
        # Image generation
        "diffusion": ["diffusion"],
        "inpaint": ["inpaint"],
        "img2img": ["img2img"],
        "controlnet": ["controlnet"],
        "lora": ["lora"],
        "vae": ["vae"],
        
        # Special architectures
        "moe": ["moe", "mixture-of-experts", "8x"],
        "dense": ["dense"],
    }
    
    # ================================================================
    # Quantization/Format Tags
    # ================================================================
    QUANT_PATTERNS: List[str] = [
        # GGUF quantizations (ordered by specificity)
        r"iq1_[sml]", r"iq2_[smlxk]+", r"iq3_[smlxk]+", r"iq4_[smlxknl]+",
        r"q2_k[_a-z]*", r"q3_k[_a-z]*", r"q4_k[_a-z]*", r"q5_k[_a-z]*", 
        r"q6_k[_a-z]*", r"q8_[0k][_a-z]*",
        r"q[2-8]_[0-9]",
        r"q[2-8][_k]*",
        # Precision formats
        r"fp16", r"bf16", r"fp32", r"f16", r"f32",
        r"fp8", r"int4", r"int8", r"nf4",
        # Other quantization methods
        r"gptq", r"awq", r"exl2", r"bnb",
    ]
    
    FORMAT_TAGS: Dict[str, List[str]] = {
        "gguf": ["gguf", ".gguf"],
        "safetensors": ["safetensors", ".safetensors"],
        "pytorch": ["pytorch", ".pt", ".pth", ".bin"],
        "onnx": ["onnx", ".onnx"],
        "tensorrt": ["tensorrt", ".engine", ".plan"],
        "ollama": ["ollama"],
        "hf": ["huggingface", "hf-"],
        "gptq": ["gptq"],
        "awq": ["awq"],
        "exl2": ["exl2", "exllama"],
    }
    
    # ================================================================
    # Size Patterns
    # ================================================================
    SIZE_ALIASES: Dict[str, float] = {
        "tiny": 0.5,
        "mini": 1.0,
        "small": 3.0,
        "medium": 7.0,
        "base": 7.0,
        "large": 13.0,
        "xlarge": 30.0,
        "xl": 30.0,
        "xxl": 70.0,
    }
    
    # ================================================================
    # Special Tags
    # ================================================================
    SPECIAL_TAGS: Dict[str, List[str]] = {
        # Version tags
        "v1": ["v1", "-v1"],
        "v2": ["v2", "-v2"],
        "v3": ["v3", "-v3"],
        
        # Speed/Optimization
        "turbo": ["turbo"],
        "fast": ["fast"],
        "lightning": ["lightning"],
        
        # Context length
        "long-context": ["long-context", "long_context", "longcontext"],
        "32k": ["32k", "-32k"],
        "128k": ["128k", "-128k"],
        "1m": ["1m-context", "1m_context"],
        
        # Special features
        "moe": ["moe", "-moe"],
        "a3b": ["a3b"],  # Active params (for MoE)
        
        # Year/Date
        "2024": ["2024"],
        "2025": ["2025"],
        "2026": ["2026"],
    }
    
    # ================================================================
    # Domain/Language Tags
    # ================================================================
    DOMAIN_TAGS: Dict[str, List[str]] = {
        # Languages
        "chinese": ["chinese", "zh-", "-zh", "cn-", "-cn"],
        "japanese": ["japanese", "ja-", "-ja", "jp-"],
        "korean": ["korean", "ko-", "-ko", "kr-"],
        "multilingual": ["multilingual", "multi-"],
        "english": ["english", "en-", "-en"],
        
        # Domains
        "coding": ["coding", "code", "coder", "programmer"],
        "math": ["math", "mathematical"],
        "roleplay": ["roleplay", "rp-"],
        "creative": ["creative", "story", "writing"],
        "translation": ["translation", "translate"],
        "medical": ["medical", "med-", "healthcare"],
        "legal": ["legal", "law-"],
        "finance": ["finance", "financial"],
        
        # Special purposes
        "nsfw": ["nsfw", "uncensored"],
        "safe": ["safe", "censored", "filtered"],
        "reasoning": ["reasoning", "cot", "chain-of-thought"],
    }
    
    # ================================================================
    # Source/Platform Tags
    # ================================================================
    SOURCE_TAGS: Dict[str, List[str]] = {
        "huggingface": ["huggingface", "hf-", "🤗"],
        "ollama": ["ollama"],
        "thebloke": ["thebloke"],
        "bartowski": ["bartowski"],
        "unsloth": ["unsloth"],
        "lmstudio": ["lmstudio", "lm-studio"],
        "mlx": ["mlx-"],
        "ggml": ["ggml-"],
        "comfyui": ["comfyui", "comfy-"],
    }

    # ================================================================
    # Main Parse Method
    # ================================================================
    @classmethod
    def parse(cls, filename: str, parent_folder: str = "") -> Dict[str, Any]:
        """
        Parse a model filename and return structured metadata.
        
        Args:
            filename: The model filename (e.g., "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
            parent_folder: Parent folder name for additional context
            
        Returns:
            Dictionary with structured metadata:
            - family: Model family (qwen, llama, etc.)
            - subfamily: Specific version (qwen2.5, llama3.1, etc.)
            - size: Parameter count in billions (e.g., 7.0)
            - size_raw: Raw size string (e.g., "7b")
            - quant: Quantization type (e.g., "q4_k_m")
            - quant_score: Precision score (higher = more precise)
            - format: Model format (gguf, safetensors, etc.)
            - architecture: Model architecture/variant tags
            - is_instruct: Whether it's instruction-tuned
            - is_moe: Whether it's MoE architecture
            - special_tags: Special feature tags
            - domain_tags: Domain/language tags
            - source: Source platform
            - keywords: All extracted keywords for search
        """
        # Combine filename and parent for analysis
        raw = f"{filename} {parent_folder}".lower()
        
        # Initialize metadata
        meta: Dict[str, Any] = {
            "family": "unknown",
            "subfamily": None,
            "size": None,
            "size_raw": None,
            "quant": None,
            "quant_score": 160,  # Default to FP16
            "format": None,
            "architecture": [],
            "is_instruct": False,
            "is_moe": False,
            "special_tags": [],
            "domain_tags": [],
            "source": None,
            "keywords": set(),
        }
        
        # 1. Extract Family
        meta["family"], meta["subfamily"] = cls._extract_family(raw)
        if meta["family"] != "unknown":
            meta["keywords"].add(meta["family"])
        if meta["subfamily"]:
            meta["keywords"].add(meta["subfamily"])
        
        # 2. Extract Size
        meta["size"], meta["size_raw"] = cls._extract_size(raw)
        if meta["size_raw"]:
            meta["keywords"].add(meta["size_raw"])
        
        # 3. Extract Quantization
        meta["quant"], meta["quant_score"] = cls._extract_quant(raw)
        if meta["quant"]:
            meta["keywords"].add(meta["quant"])
            # Add base quant level (e.g., "q4" from "q4_k_m")
            base_match = re.match(r'(q\d+|iq\d+|fp\d+|bf\d+)', meta["quant"])
            if base_match:
                meta["keywords"].add(base_match.group(1))
        
        # 4. Extract Format
        meta["format"] = cls._extract_format(raw)
        if meta["format"]:
            meta["keywords"].add(meta["format"])
        
        # 5. Extract Architecture Tags
        meta["architecture"] = cls._extract_tags(raw, cls.ARCHITECTURE_TAGS)
        meta["is_instruct"] = "instruct" in meta["architecture"] or "chat" in meta["architecture"]
        meta["is_moe"] = "moe" in meta["architecture"]
        meta["keywords"].update(meta["architecture"])
        
        # 6. Extract Special Tags
        meta["special_tags"] = cls._extract_tags(raw, cls.SPECIAL_TAGS)
        meta["keywords"].update(meta["special_tags"])
        
        # 7. Extract Domain Tags
        meta["domain_tags"] = cls._extract_tags(raw, cls.DOMAIN_TAGS)
        meta["keywords"].update(meta["domain_tags"])
        
        # 8. Extract Source
        meta["source"] = cls._extract_source(raw)
        if meta["source"]:
            meta["keywords"].add(meta["source"])
        
        # Convert keywords set to sorted list
        meta["keywords"] = sorted(meta["keywords"])
        
        return meta
    
    @classmethod
    def _extract_family(cls, raw: str) -> tuple:
        """Extract model family and subfamily."""
        family = "unknown"
        subfamily = None
        
        # Check subfamily first (more specific)
        for pattern, sub_name in cls.SUBFAMILY_PATTERNS.items():
            if pattern in raw:
                subfamily = sub_name
                break
        
        # Check family
        for fam_name, patterns in cls.FAMILY_PATTERNS.items():
            for pattern in patterns:
                if pattern in raw:
                    family = fam_name
                    break
            if family != "unknown":
                break
        
        return family, subfamily
    
    @classmethod
    def _extract_size(cls, raw: str) -> tuple:
        """Extract parameter size (in billions)."""
        # Try numeric pattern first (7b, 72b, 1.5b, 0.5b)
        size_match = re.search(r'(\d+(?:\.\d+)?)\s*[bt](?![a-z])', raw)
        if size_match:
            size = float(size_match.group(1))
            size_raw = f"{size_match.group(1)}b"
            return size, size_raw
        
        # Try alias patterns (small, medium, large)
        for alias, size_val in cls.SIZE_ALIASES.items():
            if alias in raw:
                return size_val, alias
        
        return None, None
    
    @classmethod
    def _extract_quant(cls, raw: str) -> tuple:
        """Extract quantization type and calculate score."""
        # Try each pattern
        for pattern in cls.QUANT_PATTERNS:
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                quant = match.group(0).lower()
                score = cls._calc_quant_score(quant)
                return quant, score
        
        # Default: if it's a GGUF file without quant info, assume Q4
        if "gguf" in raw:
            return "q4", 40
        
        # Non-GGUF files are usually FP16/BF16
        return None, 160
    
    @classmethod
    def _calc_quant_score(cls, quant: str) -> int:
        """Calculate quantization score from quant string."""
        quant_upper = quant.upper()
        
        for key, score in cls.QUANT_LEVELS.items():
            if key in quant_upper:
                return score
        
        return 160  # Default FP16
    
    @classmethod
    def _extract_format(cls, raw: str) -> Optional[str]:
        """Extract model format."""
        for fmt, patterns in cls.FORMAT_TAGS.items():
            for pattern in patterns:
                if pattern in raw:
                    return fmt
        return None
    
    @classmethod
    def _extract_tags(cls, raw: str, tag_dict: Dict[str, List[str]]) -> List[str]:
        """Extract tags from a tag dictionary."""
        found = []
        for tag_name, patterns in tag_dict.items():
            for pattern in patterns:
                if pattern in raw:
                    found.append(tag_name)
                    break
        return found
    
    @classmethod
    def _extract_source(cls, raw: str) -> Optional[str]:
        """Extract source platform."""
        for source, patterns in cls.SOURCE_TAGS.items():
            for pattern in patterns:
                if pattern in raw:
                    return source
        return None
    
    # ================================================================
    # Utility Methods
    # ================================================================
    @classmethod
    def get_quant_score(cls, quant: str) -> int:
        """
        Get the precision score for a quantization type.
        
        Higher scores mean more precision/quality.
        Useful for queries like "Q4 or above".
        
        Args:
            quant: Quantization string (e.g., "q4_k_m")
            
        Returns:
            Score (10-320)
        """
        return cls._calc_quant_score(quant)
    
    @classmethod
    def compare_quant(cls, quant1: str, quant2: str) -> int:
        """
        Compare two quantization types.
        
        Returns:
            -1 if quant1 < quant2
             0 if quant1 == quant2
             1 if quant1 > quant2
        """
        score1 = cls.get_quant_score(quant1)
        score2 = cls.get_quant_score(quant2)
        
        if score1 < score2:
            return -1
        elif score1 > score2:
            return 1
        return 0
    
    @classmethod
    def is_quant_above(cls, quant: str, threshold: str) -> bool:
        """
        Check if a quantization meets or exceeds a threshold.
        
        Example:
            is_quant_above("q5_k_m", "q4")  # True
            is_quant_above("q3_k_s", "q4")  # False
        """
        return cls.compare_quant(quant, threshold) >= 0
