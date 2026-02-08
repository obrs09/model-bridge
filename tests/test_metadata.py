"""
Tests for the metadata module.

Tests ModelParser functionality.
"""

import pytest

from model_bridge.metadata import ModelParser


class TestModelParser:
    """Tests for ModelParser."""
    
    def test_parse_qwen_model(self):
        """Test parsing a Qwen model name."""
        meta = ModelParser.parse("qwen2.5-7b-instruct-q5_k_m.gguf")
        
        assert meta["family"] == "qwen"
        assert meta["subfamily"] == "qwen2.5"
        assert meta["size"] == 7.0
        assert meta["quant"] == "q5_k_m"
        assert meta["quant_score"] == 50
        assert meta["is_instruct"] == True
        assert "qwen" in meta["keywords"]
        assert "7b" in meta["keywords"]
    
    def test_parse_llama_model(self):
        """Test parsing a Llama model name."""
        meta = ModelParser.parse("Meta-Llama-3-8B-Instruct.Q8_0.gguf")
        
        assert meta["family"] == "llama"
        assert meta["subfamily"] == "llama3"
        assert meta["size"] == 8.0
        assert meta["quant"] == "q8_0"
        assert meta["quant_score"] == 80
        assert meta["is_instruct"] == True
    
    def test_parse_with_parent_folder(self):
        """Test parsing with parent folder context."""
        meta = ModelParser.parse("model.safetensors", "Qwen2.5-7B-Instruct")
        
        assert meta["family"] == "qwen"
        assert meta["is_instruct"] == True
    
    def test_parse_fp16_model(self):
        """Test parsing FP16 model."""
        meta = ModelParser.parse("qwen2-7b-instruct-fp16.gguf")
        
        assert meta["quant"] == "fp16"
        assert meta["quant_score"] == 160
    
    def test_parse_base_model(self):
        """Test parsing base (non-instruct) model."""
        meta = ModelParser.parse("llama-3-8b-base.gguf")
        
        assert meta["is_instruct"] == False
    
    def test_parse_moe_model(self):
        """Test parsing MoE model."""
        meta = ModelParser.parse("mixtral-8x7b-instruct-q4_k_m.gguf")
        
        assert meta["family"] == "mistral"
        assert meta["is_moe"] == True
    
    def test_parse_unknown_model(self):
        """Test parsing unknown model returns defaults."""
        meta = ModelParser.parse("some-random-model.bin")
        
        assert meta["family"] == "unknown"
        assert isinstance(meta["keywords"], list)
    
    def test_parse_diffusion_model(self):
        """Test parsing diffusion model."""
        meta = ModelParser.parse("sdxl-turbo-v1.safetensors")
        
        assert meta["family"] == "sdxl"
        assert "sdxl" in meta["keywords"]
    
    def test_parse_whisper_model(self):
        """Test parsing Whisper model."""
        meta = ModelParser.parse("whisper-large-v3.bin")
        
        assert meta["family"] == "whisper"


class TestQuantScoring:
    """Tests for quantization scoring."""
    
    def test_get_quant_score(self):
        """Test get_quant_score for various quantizations."""
        assert ModelParser.get_quant_score("q4_k_m") == 40
        assert ModelParser.get_quant_score("q5_k_s") == 50
        assert ModelParser.get_quant_score("q8_0") == 80
        assert ModelParser.get_quant_score("fp16") == 160
        assert ModelParser.get_quant_score("bf16") == 160
    
    def test_is_quant_above(self):
        """Test is_quant_above comparisons."""
        assert ModelParser.is_quant_above("q5_k_m", "q4") == True
        assert ModelParser.is_quant_above("q8_0", "q4") == True
        assert ModelParser.is_quant_above("fp16", "q8") == True
        assert ModelParser.is_quant_above("q3_k_s", "q4") == False
        assert ModelParser.is_quant_above("q4_k_m", "q4") == True
    
    def test_compare_quant(self):
        """Test compare_quant ordering."""
        assert ModelParser.compare_quant("q5", "q4") == 1   # q5 > q4
        assert ModelParser.compare_quant("q4", "q5") == -1  # q4 < q5
        assert ModelParser.compare_quant("q4", "q4") == 0   # equal


class TestKeywordExtraction:
    """Tests for keyword extraction."""
    
    def test_keywords_include_family(self):
        """Test that family is in keywords."""
        meta = ModelParser.parse("qwen2.5-7b-instruct.gguf")
        assert "qwen" in meta["keywords"]
    
    def test_keywords_include_size(self):
        """Test that size is in keywords."""
        meta = ModelParser.parse("llama-70b-instruct.gguf")
        assert "70b" in meta["keywords"]
    
    def test_keywords_include_quant(self):
        """Test that quantization is in keywords."""
        meta = ModelParser.parse("model-q4_k_m.gguf")
        assert "q4_k_m" in meta["keywords"]
        assert "q4" in meta["keywords"]  # Base quant should also be included
    
    def test_keywords_are_sorted(self):
        """Test that keywords are sorted."""
        meta = ModelParser.parse("zephyr-7b-alpha-q4_k_m.gguf")
        assert meta["keywords"] == sorted(meta["keywords"])
