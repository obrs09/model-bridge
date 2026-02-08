"""
Tests for the scanner module.

Tests the scanning strategies and ModelInfo dataclass.
"""

import pytest
from pathlib import Path
import tempfile
import os

from model_bridge.scanner import (
    ModelInfo,
    ScannerEngine,
    HuggingFaceStrategy,
    GGUFStrategy,
    TensorRTStrategy,
    ComfyUIStrategy,
)
from model_bridge.scanner.strategies import SafetensorsStrategy, ScanUtils
from model_bridge.config import get_config


class TestModelInfo:
    """Tests for ModelInfo dataclass."""
    
    def test_create_model_info(self):
        """Test creating a ModelInfo instance."""
        info = ModelInfo(
            id="test-model",
            path="/path/to/model.gguf",
            type="gguf",
            engine_support=["llama.cpp", "ollama"],
            metadata={"family": "qwen"},
            size_bytes=1024,
        )
        
        assert info.id == "test-model"
        assert info.path == "/path/to/model.gguf"
        assert info.type == "gguf"
        assert info.size_bytes == 1024
        assert "llama.cpp" in info.engine_support
    
    def test_to_dict(self):
        """Test converting ModelInfo to dictionary."""
        info = ModelInfo(
            id="test-model",
            path="/path/to/model.gguf",
            type="gguf",
            engine_support=["llama.cpp"],
            metadata={"key": "value"},
            size_bytes=1024,
        )
        
        result = info.to_dict()
        
        assert isinstance(result, dict)
        assert result["id"] == "test-model"
        assert result["metadata"] == {"key": "value"}
    
    def test_from_dict(self):
        """Test creating ModelInfo from dictionary."""
        data = {
            "id": "test-model",
            "path": "/path/to/model.gguf",
            "type": "gguf",
            "engine_support": ["llama.cpp"],
            "metadata": {"key": "value"},
            "size_bytes": 1024,
        }
        
        info = ModelInfo.from_dict(data)
        
        assert info.id == "test-model"
        assert info.type == "gguf"


class TestScanUtils:
    """Tests for ScanUtils shard detection."""
    
    def test_is_follower_shard_pattern1(self):
        """Test shard detection for -00002-of-00005 pattern."""
        assert ScanUtils.is_follower_shard("model-00001-of-00005.safetensors") == False
        assert ScanUtils.is_follower_shard("model-00002-of-00005.safetensors") == True
        assert ScanUtils.is_follower_shard("model-00005-of-00005.safetensors") == True
    
    def test_is_follower_shard_pattern2(self):
        """Test shard detection for .part2. pattern."""
        assert ScanUtils.is_follower_shard("model.part1.gguf") == False
        assert ScanUtils.is_follower_shard("model.part2.gguf") == True
        assert ScanUtils.is_follower_shard("model.part10.gguf") == True
    
    def test_not_a_shard(self):
        """Test that regular files are not detected as shards."""
        assert ScanUtils.is_follower_shard("regular-model.gguf") == False
        assert ScanUtils.is_follower_shard("model-v2.safetensors") == False
    
    def test_get_shard_base_name(self):
        """Test extracting base name from sharded file."""
        assert ScanUtils.get_shard_base_name("model-00001-of-00005.safetensors") == "model.safetensors"
        assert ScanUtils.get_shard_base_name("regular.gguf") == "regular.gguf"


class TestGGUFStrategy:
    """Tests for GGUFStrategy."""
    
    def test_strategy_name(self):
        """Test strategy name."""
        strategy = GGUFStrategy()
        assert strategy.name == "gguf"
    
    def test_scan_empty_directory(self):
        """Test scanning an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy = GGUFStrategy()
            results = strategy.scan([Path(tmpdir)])
            assert results == []
    
    def test_scan_with_gguf_file(self):
        """Test scanning a directory with a GGUF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake GGUF file
            gguf_path = Path(tmpdir) / "test-model.gguf"
            gguf_path.write_bytes(b"fake gguf content")
            
            strategy = GGUFStrategy()
            results = strategy.scan([Path(tmpdir)])
            
            assert len(results) == 1
            assert results[0].id == "test-model"
            assert results[0].type == "gguf"
            assert "llama.cpp" in results[0].engine_support


class TestSafetensorsStrategy:
    """Tests for SafetensorsStrategy."""
    
    def test_strategy_name(self):
        """Test strategy name."""
        strategy = SafetensorsStrategy()
        assert strategy.name == "safetensors"
    
    def test_skip_hf_model_files(self):
        """Test that files inside HF model dirs (with config.json) are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            
            # Create config.json (HF model marker)
            (model_dir / "config.json").write_text("{}")
            
            # Create a safetensors file
            (model_dir / "model.safetensors").write_bytes(b"fake safetensors")
            
            strategy = SafetensorsStrategy()
            results = strategy.scan([Path(tmpdir)])
            
            # Should skip because parent has config.json
            assert len(results) == 0


class TestScannerEngine:
    """Tests for ScannerEngine."""
    
    def test_default_strategies(self):
        """Test that default strategies are loaded."""
        config = get_config()
        engine = ScannerEngine(config)
        
        strategy_names = engine.strategy_names
        assert "huggingface" in strategy_names
        assert "gguf" in strategy_names
        assert "tensorrt" in strategy_names
        assert "comfyui" in strategy_names
        assert "safetensors" in strategy_names
    
    def test_add_custom_strategy(self):
        """Test adding a custom strategy."""
        from model_bridge.scanner.base import ScanStrategy
        
        class CustomStrategy(ScanStrategy):
            @property
            def name(self):
                return "custom"
            
            def scan(self, paths):
                return []
        
        config = get_config()
        engine = ScannerEngine(config)
        engine.add_strategy(CustomStrategy())
        
        assert "custom" in engine.strategy_names
    
    def test_remove_strategy(self):
        """Test removing a strategy."""
        config = get_config()
        engine = ScannerEngine(config)
        
        assert "gguf" in engine.strategy_names
        result = engine.remove_strategy("gguf")
        assert result == True
        assert "gguf" not in engine.strategy_names
