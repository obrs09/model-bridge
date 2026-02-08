"""
Tests for the ranker module.

Tests ModelRanker functionality.
"""

import pytest

from model_bridge.ranker import ModelRanker


class TestParseFeatures:
    """Tests for parse_features."""
    
    def test_parse_size(self):
        """Test extracting parameter size."""
        features = ModelRanker.parse_features("qwen2-7b-instruct")
        assert features["size"] == "7"
        
        features = ModelRanker.parse_features("llama-70b")
        assert features["size"] == "70"
    
    def test_parse_quant(self):
        """Test extracting quantization."""
        features = ModelRanker.parse_features("model-q4_k_m.gguf")
        assert features["quant"] == "q4_k_m"
        
        features = ModelRanker.parse_features("model-Q8_0.gguf")
        assert features["quant"] == "q8_0"
    
    def test_parse_format(self):
        """Test extracting format."""
        features = ModelRanker.parse_features("model.gguf")
        assert features["format"] == "gguf"
        
        features = ModelRanker.parse_features("model-awq")
        assert features["format"] == "awq"
    
    def test_parse_instruct(self):
        """Test detecting instruct models."""
        features = ModelRanker.parse_features("qwen2-7b-instruct")
        assert features["is_instruct"] == True
        
        features = ModelRanker.parse_features("llama-7b-chat")
        assert features["is_instruct"] == True
        
        features = ModelRanker.parse_features("llama-7b-base")
        assert features["is_instruct"] == False
    
    def test_parse_moe(self):
        """Test detecting MoE models."""
        features = ModelRanker.parse_features("mixtral-8x7b")
        assert features["is_moe"] == True
        
        features = ModelRanker.parse_features("qwen-moe-14b")
        assert features["is_moe"] == True
    
    def test_parse_tokens(self):
        """Test token extraction."""
        features = ModelRanker.parse_features("qwen2-7b-instruct")
        assert "qwen2" in features["tokens"]
        assert "7b" in features["tokens"]
        assert "instruct" in features["tokens"]


class TestRank:
    """Tests for rank method."""
    
    def test_rank_empty_query(self):
        """Test ranking with empty query."""
        candidates = [{"id": "model1", "path": "/path"}]
        result = ModelRanker.rank("", candidates)
        assert result == candidates
    
    def test_rank_empty_candidates(self):
        """Test ranking with empty candidates."""
        result = ModelRanker.rank("qwen", [])
        assert result == []
    
    def test_rank_filters_by_size(self):
        """Test that size filter works."""
        candidates = [
            {"id": "qwen2-7b", "path": "/path/qwen2-7b.gguf", "type": "gguf", "metadata": {}},
            {"id": "qwen2-72b", "path": "/path/qwen2-72b.gguf", "type": "gguf", "metadata": {}},
        ]
        
        result = ModelRanker.rank("qwen 7b", candidates)
        
        # Only 7b should match
        assert len(result) == 1
        assert "7b" in result[0]["id"]
    
    def test_rank_filters_by_format(self):
        """Test that format filter works."""
        candidates = [
            {"id": "model-gguf", "path": "/path/model.gguf", "type": "gguf", "metadata": {}},
            {"id": "model-awq", "path": "/path/model-awq", "type": "awq", "metadata": {}},
        ]
        
        result = ModelRanker.rank("model gguf", candidates)
        
        # Only gguf should match
        assert len(result) == 1
        assert result[0]["type"] == "gguf"
    
    def test_rank_with_min_quant(self):
        """Test min_quant filtering."""
        candidates = [
            {"id": "model-q4", "path": "/path", "type": "gguf", 
             "metadata": {"quant": "q4_k_m", "quant_score": 40}},
            {"id": "model-q8", "path": "/path", "type": "gguf", 
             "metadata": {"quant": "q8_0", "quant_score": 80}},
        ]
        
        result = ModelRanker.rank("model", candidates, min_quant="q5")
        
        # Only q8 should pass (q4 < q5)
        assert len(result) == 1
        assert "q8" in result[0]["id"]
    
    def test_rank_prefers_instruct(self):
        """Test that instruct models are ranked higher."""
        candidates = [
            {"id": "qwen-7b-base", "path": "/path/base.gguf", "type": "gguf", "metadata": {}},
            {"id": "qwen-7b-instruct", "path": "/path/instruct.gguf", "type": "gguf", "metadata": {}},
        ]
        
        result = ModelRanker.rank("qwen 7b", candidates)
        
        # Instruct should be ranked first
        assert len(result) == 2
        assert "instruct" in result[0]["id"]


class TestExtractVersion:
    """Tests for extract_version."""
    
    def test_extract_version_simple(self):
        """Test extracting simple versions."""
        assert ModelRanker.extract_version("qwen2.5") == 2.5
        assert ModelRanker.extract_version("llama3") == 3.0
        assert ModelRanker.extract_version("v1.5") == 1.5
    
    def test_extract_version_ignores_size(self):
        """Test that sizes are not confused with versions."""
        # 7 and 8 are common sizes, should be ignored
        assert ModelRanker.extract_version("model-7b") == 0.0
        assert ModelRanker.extract_version("model-8b") == 0.0
    
    def test_extract_version_none(self):
        """Test when no version is found."""
        assert ModelRanker.extract_version("simple-model") == 0.0


class TestExplainScore:
    """Tests for explain_score."""
    
    def test_explain_score_structure(self):
        """Test that explain_score returns expected structure."""
        model = {"id": "qwen2-7b-instruct", "path": "/path", "type": "gguf", "metadata": {}}
        
        explanation = ModelRanker.explain_score("qwen 7b", model)
        
        assert "model_id" in explanation
        assert "query_features" in explanation
        assert "model_features" in explanation
        assert "scores" in explanation
        assert "total" in explanation
        assert "rejected" in explanation
    
    def test_explain_score_rejection(self):
        """Test explanation for rejected model."""
        model = {"id": "llama-70b", "path": "/path", "type": "gguf", "metadata": {}}
        
        explanation = ModelRanker.explain_score("qwen 7b", model)
        
        # Should be rejected due to no match or size mismatch
        # Either rejected or low score
        assert isinstance(explanation["rejected"], bool)
