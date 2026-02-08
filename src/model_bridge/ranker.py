"""
Model Ranker - Layered Weighted Fuzzy Matching System.

This module implements a sophisticated ranking algorithm for model search:
1. Feature Extraction: Parse model names into structured tags
2. Hard Filtering: Eliminate models that don't match required criteria
3. Soft Scoring: Rank remaining models by relevance and preferences

The ranking system considers:
- Token matching (words from query appearing in model name)
- Keyword matching (from parsed metadata.keywords)
- Parameter size (7B, 14B, 72B - hard filter)
- Quantization filtering (support "Q4以上" queries)
- Version preference (newer versions score higher)
- Instruct/Chat preference (dialogue models preferred)
- Quantization preference (Q4_K_M, Q5_K_M as sweet spots)
- Format preference (GGUF for local inference)
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Any

from .metadata import ModelParser


class ModelRanker:
    """
    Responsible for parsing user queries and scoring/ranking candidate models.
    
    Example:
        >>> ranker = ModelRanker()
        >>> ranked = ranker.rank("qwen 7b instruct", candidates)
        >>> best = ranked[0]  # Highest scoring match
    """
    
    # Regex patterns for feature extraction
    RE_SIZE = re.compile(r'(\d+)[bB]')           # Match 7b, 72B, 14B
    RE_QUANT = re.compile(r'[qQ]\d+[_a-zA-Z0-9]*')  # Match q4_k_m, Q8_0
    RE_VERSION = re.compile(r'v?(\d+\.?\d*)')    # Match 2.5, v1.0, 3
    RE_FORMAT = re.compile(r'\b(gguf|awq|gptq|fp16|bf16|int4|int8)\b', re.IGNORECASE)
    
    # Scoring weights
    WEIGHTS = {
        "token_match": 100,      # Per token match
        "keyword_match": 80,     # Per keyword match from metadata
        "exact_id_match": 500,   # Exact ID match bonus
        "instruct_bonus": 50,    # Model is instruct/chat
        "instruct_query": 50,    # Query explicitly asks for instruct
        "version_multiplier": 10,  # Per version point (e.g., 2.5 -> 25)
        "quant_sweet_spot": 30,  # Q4_K_M, Q5_K_M
        "quant_high": 25,        # Q8_0
        "quant_fp16": 20,        # FP16
        "quant_low_penalty": -10,  # Q2 penalty
        "gguf_bonus": 10,        # GGUF format preference
        "base_penalty": -20,     # Base model when instruct preferred
        "family_match": 100,     # Family match bonus
        "size_match": 50,        # Size match bonus
    }
    
    @classmethod
    def parse_features(cls, text: str) -> Dict[str, Any]:
        """
        Extract structured features from a text string.
        
        Args:
            text: Model name, path, or query string
            
        Returns:
            Dictionary with extracted features:
            - size: Parameter size (e.g., "7", "72")
            - quant: Quantization type (e.g., "q4_k_m")
            - format: Model format (e.g., "gguf", "awq")
            - is_instruct: Whether it's an instruction-tuned model
            - is_base: Whether it's a base model
            - is_moe: Whether it's a Mixture of Experts model
            - tokens: Set of tokens from the text
        """
        text_lower = text.lower()
        
        features = {
            "size": None,
            "quant": None,
            "format": None,
            "is_instruct": False,
            "is_base": False,
            "is_moe": False,
            "tokens": set(),
        }
        
        # Tokenize (split on common delimiters)
        tokens = set(re.split(r'[-_.\s/\\]+', text_lower))
        # Filter out empty and very short tokens
        features["tokens"] = {t for t in tokens if len(t) >= 2}
        
        # Extract parameter size (first match)
        size_match = cls.RE_SIZE.search(text_lower)
        if size_match:
            features["size"] = size_match.group(1)
        
        # Extract quantization
        quant_match = cls.RE_QUANT.search(text_lower)
        if quant_match:
            features["quant"] = quant_match.group(0).lower()
        
        # Extract format
        format_match = cls.RE_FORMAT.search(text_lower)
        if format_match:
            features["format"] = format_match.group(0).lower()
        
        # Detect instruct/chat models
        instruct_keywords = ["instruct", "chat", "aligned", "sft", "rlhf", "dpo"]
        if any(kw in text_lower for kw in instruct_keywords):
            features["is_instruct"] = True
        
        # Detect base models
        if "base" in text_lower and "base-" not in text_lower:
            features["is_base"] = True
        
        # Detect MoE models
        moe_keywords = ["moe", "mixtral", "mixture"]
        if any(kw in text_lower for kw in moe_keywords):
            features["is_moe"] = True
        
        return features
    
    @classmethod
    def extract_version(cls, text: str) -> float:
        """
        Extract the highest version number from text.
        
        Ignores numbers that look like parameter sizes.
        
        Args:
            text: Text to search for version
            
        Returns:
            Highest version number found, or 0.0
        """
        text_lower = text.lower()
        versions = cls.RE_VERSION.findall(text_lower)
        
        if not versions:
            return 0.0
        
        valid_versions = []
        for v in versions:
            try:
                num = float(v)
                # Ignore numbers that are likely param sizes (7, 8, 13, 14, 30, 70, 72, etc.)
                # Valid versions are typically < 10
                if num < 10 and num not in {7, 8}:
                    valid_versions.append(num)
            except ValueError:
                continue
        
        return max(valid_versions) if valid_versions else 0.0
    
    @classmethod
    def rank(cls, query: str, candidates: List[Dict[str, Any]], 
             min_quant: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Core ranking algorithm using layered weighted scoring.
        
        Process:
        1. Parse query features (including from ModelParser)
        2. For each candidate:
           a. Hard filter (size, MoE, format, min_quant requirements)
           b. Token + keyword matching score
           c. Heuristic bonuses (instruct, version, quantization)
        3. Sort by score descending
        
        Args:
            query: User search query (e.g., "qwen 2.5 7b")
            candidates: List of model dictionaries from registry
            min_quant: Minimum quantization threshold (e.g., "q4" for "Q4 or above")
            
        Returns:
            Sorted list of models, highest score first
        """
        if not query or not candidates:
            return candidates
        
        query = query.strip()
        q_feats = cls.parse_features(query)
        
        # Also parse query with ModelParser for richer matching
        q_meta = ModelParser.parse(query, "")
        q_keywords = set(q_meta.get("keywords", []))
        
        scored_candidates: List[tuple] = []
        
        for model in candidates:
            m_id = model.get("id", "")
            m_path = model.get("path", "")
            m_type = model.get("type", "")
            m_metadata = model.get("metadata", {})
            
            # Get model keywords from metadata
            m_keywords = set(m_metadata.get("keywords", []))
            m_quant = m_metadata.get("quant")
            m_quant_score = m_metadata.get("quant_score", 160)
            m_family = m_metadata.get("family", "unknown")
            m_size = m_metadata.get("size")
            
            # Combine ID and filename for analysis
            filename = Path(m_path).name if m_path else ""
            full_text = f"{m_id} {filename}".lower()
            m_feats = cls.parse_features(full_text)
            
            score = 0
            
            # ==========================================
            # STEP 1: Hard Filters (Must-Have Rules)
            # ==========================================
            
            # 1A. Parameter size filter
            # If query specifies size (e.g., "7b"), model must match
            if q_feats["size"]:
                if m_feats["size"] != q_feats["size"]:
                    continue  # Hard reject
            
            # 1B. MoE filter
            # If query asks for MoE, model must be MoE
            if q_feats["is_moe"] and not m_feats["is_moe"]:
                continue  # Hard reject
            
            # 1C. Format filter
            # If query specifies format (gguf, awq), model must match
            if q_feats["format"]:
                if q_feats["format"] not in full_text and q_feats["format"] not in m_type.lower():
                    continue  # Hard reject
            
            # 1D. Minimum quantization filter (e.g., "Q4 or above")
            if min_quant and m_quant:
                min_score = ModelParser.get_quant_score(min_quant)
                if m_quant_score < min_score:
                    continue  # Hard reject
            
            # ==========================================
            # STEP 2: Token & Keyword Matching (Base Score)
            # ==========================================
            
            matches = 0
            meaningful_tokens = [t for t in q_feats["tokens"] if len(t) >= 2]
            
            # Token matching against full_text
            for token in meaningful_tokens:
                if token in full_text:
                    matches += 1
            
            # Keyword matching (from parsed metadata)
            keyword_matches = len(q_keywords & m_keywords)
            
            # Must have at least one token or keyword match
            if matches == 0 and keyword_matches == 0:
                continue
            
            # Quality gate: Require at least 30% token match ratio
            # (relaxed if we have keyword matches)
            total_tokens = len(meaningful_tokens)
            if total_tokens > 0 and keyword_matches == 0:
                match_ratio = matches / total_tokens
                # For short queries (1-2 tokens), require 100% match
                # For longer queries (3+ tokens), require at least 30%
                min_ratio = 1.0 if total_tokens <= 2 else 0.3
                if match_ratio < min_ratio:
                    continue
            
            # Base score from token matches
            score += matches * cls.WEIGHTS["token_match"]
            
            # Keyword match bonus
            score += keyword_matches * cls.WEIGHTS["keyword_match"]
            
            # Family match bonus
            if q_meta.get("family") != "unknown" and m_family == q_meta.get("family"):
                score += cls.WEIGHTS["family_match"]
            
            # Size match bonus (when metadata has size)
            if q_meta.get("size") and m_size and m_size == q_meta.get("size"):
                score += cls.WEIGHTS["size_match"]
            
            # Exact ID match bonus
            if m_id.lower() == query.lower():
                score += cls.WEIGHTS["exact_id_match"]
            
            # ==========================================
            # STEP 3: Heuristic Bonuses
            # ==========================================
            
            # 3A. Instruct/Chat preference
            if m_feats["is_instruct"]:
                score += cls.WEIGHTS["instruct_bonus"]
                # Extra bonus if query explicitly asks for instruct
                if q_feats["is_instruct"]:
                    score += cls.WEIGHTS["instruct_query"]
            elif m_feats["is_base"] and not q_feats["is_base"]:
                # Penalize base models unless explicitly requested
                score += cls.WEIGHTS["base_penalty"]
            
            # 3B. Version preference (newer is better)
            version = cls.extract_version(full_text)
            if version > 0:
                score += int(version * cls.WEIGHTS["version_multiplier"])
            
            # 3C. Quantization preference (sweet spot ranking)
            if m_feats["quant"]:
                quant = m_feats["quant"]
                if "q4_k_m" in quant or "q5_k_m" in quant:
                    score += cls.WEIGHTS["quant_sweet_spot"]
                elif "q8_0" in quant or "q8" in quant:
                    score += cls.WEIGHTS["quant_high"]
                elif "fp16" in quant:
                    score += cls.WEIGHTS["quant_fp16"]
                elif "q2" in quant:
                    score += cls.WEIGHTS["quant_low_penalty"]
            
            # 3D. Format preference (GGUF for local inference)
            if "gguf" in m_type.lower():
                score += cls.WEIGHTS["gguf_bonus"]
            
            # 3E. MoE bonus when explicitly requested
            if q_feats["is_moe"] and m_feats["is_moe"]:
                score += 50
            
            scored_candidates.append((score, model))
        
        # Sort by score (descending), then by ID length (shorter = more specific)
        scored_candidates.sort(key=lambda x: (-x[0], len(x[1].get("id", ""))))
        
        return [item[1] for item in scored_candidates]
    
    @classmethod
    def explain_score(cls, query: str, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain why a model received its score (for debugging).
        
        Args:
            query: User query
            model: Model dictionary
            
        Returns:
            Dictionary with score breakdown
        """
        q_feats = cls.parse_features(query)
        
        m_id = model.get("id", "")
        m_path = model.get("path", "")
        m_type = model.get("type", "")
        filename = Path(m_path).name if m_path else ""
        full_text = f"{m_id} {filename}".lower()
        m_feats = cls.parse_features(full_text)
        
        breakdown = {
            "model_id": m_id,
            "query_features": q_feats,
            "model_features": m_feats,
            "scores": {},
            "total": 0,
            "rejected": False,
            "rejection_reason": None,
        }
        
        # Check hard filters
        if q_feats["size"] and m_feats["size"] != q_feats["size"]:
            breakdown["rejected"] = True
            breakdown["rejection_reason"] = f"Size mismatch: query={q_feats['size']}, model={m_feats['size']}"
            return breakdown
        
        if q_feats["is_moe"] and not m_feats["is_moe"]:
            breakdown["rejected"] = True
            breakdown["rejection_reason"] = "Query requires MoE but model is not MoE"
            return breakdown
        
        # Token matches
        matches = sum(1 for t in q_feats["tokens"] if len(t) >= 2 and t in full_text)
        if matches == 0:
            breakdown["rejected"] = True
            breakdown["rejection_reason"] = "No token matches"
            return breakdown
        
        breakdown["scores"]["token_matches"] = matches * cls.WEIGHTS["token_match"]
        
        if m_feats["is_instruct"]:
            breakdown["scores"]["instruct_bonus"] = cls.WEIGHTS["instruct_bonus"]
        
        version = cls.extract_version(full_text)
        if version > 0:
            breakdown["scores"]["version_bonus"] = int(version * cls.WEIGHTS["version_multiplier"])
        
        if m_feats["quant"]:
            quant = m_feats["quant"]
            if "q4_k_m" in quant or "q5_k_m" in quant:
                breakdown["scores"]["quant_bonus"] = cls.WEIGHTS["quant_sweet_spot"]
            elif "q8" in quant:
                breakdown["scores"]["quant_bonus"] = cls.WEIGHTS["quant_high"]
        
        if "gguf" in m_type.lower():
            breakdown["scores"]["gguf_bonus"] = cls.WEIGHTS["gguf_bonus"]
        
        breakdown["total"] = sum(breakdown["scores"].values())
        
        return breakdown
