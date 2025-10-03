#!/usr/bin/env python3
"""
📊 Similarity Calculator - Mock Version for Testing
"""

class SimilarityCalculator:
    """Mock similarity calculator for testing"""

    def calculate_name_similarity(self, tokens1, tokens2):
        """Mock name similarity calculation"""
        if not tokens1 or not tokens2:
            return 0.0

        # Simple intersection-based similarity
        common = set(tokens1) & set(tokens2)
        total = set(tokens1) | set(tokens2)

        return len(common) / len(total) if total else 0.0

    def calculate_location_similarity(self, loc1, loc2):
        """Mock location similarity"""
        if not loc1 or not loc2:
            return 0.0

        return 0.8 if loc1.lower() == loc2.lower() else 0.3

    def calculate_temporal_similarity(self, temp1, temp2):
        """Mock temporal similarity"""
        return 0.7  # Mock value

    def calculate_username_cluster_similarity(self, usernames):
        """Mock username cluster similarity"""
        return 0.6  # Mock value

    def calculate_temporal_consistency(self, dates):
        """Mock temporal consistency"""
        return 0.8  # Mock value