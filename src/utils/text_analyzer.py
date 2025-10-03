#!/usr/bin/env python3
"""
📝 Text Analyzer - Mock Version for Testing
"""

import re
import numpy as np

class TextAnalyzer:
    """Mock text analyzer for testing"""

    def tokenize_name(self, name):
        """Mock name tokenization"""
        if not name:
            return []
        # Simple tokenization
        return re.findall(r'\w+', name.lower())

    async def get_text_embedding(self, text):
        """Mock text embedding generation"""
        if not text:
            return np.zeros(384)  # Return zero vector for empty text

        # Mock embedding based on text hash
        text_hash = hash(text) % (2**32)
        np.random.seed(text_hash)
        return np.random.rand(384)  # Mock 384-dimensional embedding

    def extract_linguistic_features(self, text):
        """Mock linguistic feature extraction"""
        if not text:
            return {}

        return {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(text.split('.')),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }

    def calculate_name_similarity(self, tokens1, tokens2):
        """Mock name similarity calculation"""
        if not tokens1 or not tokens2:
            return 0.0

        set1, set2 = set(tokens1), set(tokens2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0