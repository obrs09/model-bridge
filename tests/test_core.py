"""
Tests for the core module.

Tests ModelRegistry functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path

from model_bridge.core import ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_singleton_pattern(self):
        """Test that ModelRegistry is a singleton."""
        reg1 = ModelRegistry()
        reg2 = ModelRegistry()
        assert reg1 is reg2
    
    def test_registry_has_models_list(self):
        """Test that registry has a models list."""
        registry = ModelRegistry()
        assert hasattr(registry, 'models')
        assert isinstance(registry.models, list)
    
    def test_count_property(self):
        """Test count property returns number of models."""
        registry = ModelRegistry()
        assert registry.count == len(registry.models)
    
    def test_is_empty_property(self):
        """Test is_empty property."""
        registry = ModelRegistry()
        if len(registry.models) == 0:
            assert registry.is_empty == True
        else:
            assert registry.is_empty == False
    
    def test_find_returns_dict_for_top_k_1(self):
        """Test that find returns a dict when top_k=1."""
        registry = ModelRegistry()
        if not registry.is_empty:
            result = registry.find("model", top_k=1)
            if result is not None:
                assert isinstance(result, dict)
    
    def test_find_returns_list_for_top_k_greater_than_1(self):
        """Test that find returns a list when top_k > 1."""
        registry = ModelRegistry()
        result = registry.find("model", top_k=5)
        assert isinstance(result, list)
    
    def test_find_with_empty_query(self):
        """Test find with empty query."""
        registry = ModelRegistry()
        result = registry.find("", top_k=1)
        assert result is None
        
        result = registry.find("", top_k=5)
        assert result == []
    
    def test_find_with_min_quant(self):
        """Test find with min_quant filter."""
        registry = ModelRegistry()
        
        # This should not crash
        result = registry.find("test", top_k=5, min_quant="q4")
        assert isinstance(result, list)
    
    def test_get_path_returns_string_or_none(self):
        """Test get_path returns string or None."""
        registry = ModelRegistry()
        result = registry.get_path("test-model")
        assert result is None or isinstance(result, str)
    
    def test_get_by_type(self):
        """Test filtering by model type."""
        registry = ModelRegistry()
        
        gguf_models = registry.get_by_type("gguf")
        assert isinstance(gguf_models, list)
        
        for model in gguf_models:
            assert model.get("type") == "gguf"
    
    def test_get_by_engine(self):
        """Test filtering by engine."""
        registry = ModelRegistry()
        
        llamacpp_models = registry.get_by_engine("llama.cpp")
        assert isinstance(llamacpp_models, list)
        
        for model in llamacpp_models:
            assert "llama.cpp" in model.get("engine_support", [])
    
    def test_stats(self):
        """Test stats method returns expected keys."""
        registry = ModelRegistry()
        stats = registry.stats()
        
        assert "total_models" in stats
        assert "total_size_gb" in stats
        assert "by_type" in stats
        assert isinstance(stats["by_type"], dict)
