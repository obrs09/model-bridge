"""
Tests for the decorator module.

Tests the @smart_load decorator and model resolution.
"""

import pytest
from pathlib import Path
import tempfile

from model_bridge.decorator import smart_load, _is_likely_model_query


class TestIsLikelyModelQuery:
    """Tests for _is_likely_model_query helper."""
    
    def test_short_name_is_query(self):
        """Test that short names are considered queries."""
        assert _is_likely_model_query("qwen") == True
        assert _is_likely_model_query("llama-7b") == True
        assert _is_likely_model_query("mistral") == True
    
    def test_existing_path_is_not_query(self):
        """Test that existing paths are not considered queries."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            temp_path = f.name
            f.write(b"test")
        
        try:
            assert _is_likely_model_query(temp_path) == False
        finally:
            Path(temp_path).unlink()
    
    def test_path_like_string_is_not_query(self):
        """Test that path-like strings are not considered queries."""
        assert _is_likely_model_query("/path/to/model") == False
        assert _is_likely_model_query("C:\\Models\\llama.gguf") == False
        assert _is_likely_model_query("./models/test.bin") == False
    
    def test_non_string_is_not_query(self):
        """Test that non-strings are not considered queries."""
        assert _is_likely_model_query(None) == False
        assert _is_likely_model_query(123) == False
        assert _is_likely_model_query(["list"]) == False


class TestSmartLoadDecorator:
    """Tests for @smart_load decorator."""
    
    def test_decorator_with_existing_path(self):
        """Test decorator passes through existing paths unchanged."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            temp_path = f.name
            f.write(b"fake model")
        
        try:
            @smart_load
            def load_model(model_path: str) -> str:
                return model_path
            
            result = load_model(temp_path)
            # Path should be unchanged since it exists
            assert result == temp_path
        finally:
            Path(temp_path).unlink()
    
    def test_decorator_without_parentheses(self):
        """Test decorator can be used without parentheses."""
        @smart_load
        def my_func(model_path: str) -> str:
            return model_path
        
        # Should work - if not found, passes through original
        # This tests the decorator doesn't crash
        result = my_func("some-nonexistent-model-xyz123")
        # If not found, should pass through original value
        assert isinstance(result, str)
    
    def test_decorator_preserves_function_name(self):
        """Test decorator preserves original function metadata."""
        @smart_load
        def original_function(model_path: str) -> str:
            """Original docstring."""
            return model_path
        
        assert original_function.__name__ == "original_function"
        assert "Original docstring" in (original_function.__doc__ or "")
    
    def test_decorator_passes_extra_args(self):
        """Test decorator passes additional arguments correctly."""
        @smart_load
        def load_with_options(model_path: str, device: str = "cpu") -> tuple:
            return (model_path, device)
        
        # Use a real path to skip model resolution
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            temp_path = f.name
            f.write(b"test")
        
        try:
            result = load_with_options(temp_path, device="cuda")
            assert result[0] == temp_path
            assert result[1] == "cuda"
        finally:
            Path(temp_path).unlink()
    
    def test_decorator_with_kwargs(self):
        """Test decorator works with keyword arguments."""
        @smart_load
        def load_model(pretrained_model_name_or_path: str, **kwargs) -> dict:
            return {"path": pretrained_model_name_or_path, **kwargs}
        
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = f.name
            f.write(b"test")
        
        try:
            result = load_model(pretrained_model_name_or_path=temp_path, trust_remote_code=True)
            assert result["path"] == temp_path
            assert result["trust_remote_code"] == True
        finally:
            Path(temp_path).unlink()


class TestSmartLoadV2:
    """Tests for smart_load_v2 advanced decorator."""
    
    def test_import_smart_load_v2(self):
        """Test that smart_load_v2 can be imported."""
        from model_bridge.decorator import smart_load_v2
        assert callable(smart_load_v2)
    
    def test_silent_mode(self):
        """Test silent mode doesn't print output."""
        from model_bridge.decorator import smart_load_v2
        
        @smart_load_v2(silent=True)
        def silent_load(model_path: str) -> str:
            return model_path
        
        # Should not raise, just pass through
        result = silent_load("nonexistent-model")
        assert isinstance(result, str)
