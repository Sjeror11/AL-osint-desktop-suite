#!/usr/bin/env python3
"""
🧪 AI Enhancement Phase 3 - Comprehensive Test Suite
OSINT Desktop Suite - Testing
LakyLuk Enhanced Edition - 4.10.2025

Tests:
- EnhancedConfidenceScorer functionality
- AI Voting System strategies
- AI Performance Analytics tracking
- Integration testing
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.ai_confidence_scorer import (
    EnhancedConfidenceScorer,
    ConfidenceFactors,
    ConfidenceMetric
)
from core.ai_voting_system import (
    AIVotingSystem,
    AIVote,
    VotingStrategy
)
from core.ai_performance_analytics import (
    AIPerformanceAnalytics,
    PerformanceMetric
)


class TestEnhancedConfidenceScorer(unittest.TestCase):
    """Test Enhanced Confidence Scorer"""

    def setUp(self):
        """Setup test environment"""
        self.scorer = EnhancedConfidenceScorer()

    def test_basic_confidence_calculation(self):
        """Test basic confidence score calculation"""
        score = self.scorer.calculate_enhanced_confidence(
            model_name="test_model",
            intrinsic_confidence=0.8,
            other_confidences=[0.75, 0.85],
            context={}
        )

        self.assertIsNotNone(score)
        self.assertTrue(0 <= score.overall_score <= 1)
        self.assertIn(score.certainty_level, ["very_low", "low", "medium", "high", "very_high"])

    def test_confidence_with_high_consensus(self):
        """Test confidence scoring with high consensus"""
        score = self.scorer.calculate_enhanced_confidence(
            model_name="test_model",
            intrinsic_confidence=0.85,
            other_confidences=[0.83, 0.87],  # High agreement
            context={'data_sources': ['official_records', 'government_database']}
        )

        # High consensus should give high consensus_agreement
        self.assertGreater(score.confidence_factors.consensus_agreement, 0.8)

    def test_confidence_with_low_consensus(self):
        """Test confidence scoring with low consensus"""
        score = self.scorer.calculate_enhanced_confidence(
            model_name="test_model",
            intrinsic_confidence=0.8,
            other_confidences=[0.4, 0.9],  # Low agreement
            context={}
        )

        # Low consensus should reduce consensus_agreement
        self.assertLess(score.confidence_factors.consensus_agreement, 0.7)

    def test_confidence_interval(self):
        """Test confidence interval calculation"""
        score = self.scorer.calculate_enhanced_confidence(
            model_name="test_model",
            intrinsic_confidence=0.75,
            other_confidences=[0.7, 0.8],
            context={}
        )

        lower, upper = score.confidence_interval

        # Interval should contain overall score
        self.assertLessEqual(lower, score.overall_score)
        self.assertLessEqual(score.overall_score, upper)

        # Interval should be within bounds
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation"""
        # Low quality data should increase uncertainty
        score_low_quality = self.scorer.calculate_enhanced_confidence(
            model_name="test_model",
            intrinsic_confidence=0.7,
            other_confidences=[0.6, 0.8],
            context={'data_completeness': 0.3}  # Low quality
        )

        # High quality data should decrease uncertainty
        score_high_quality = self.scorer.calculate_enhanced_confidence(
            model_name="test_model",
            intrinsic_confidence=0.7,
            other_confidences=[0.6, 0.8],
            context={'data_completeness': 0.9}  # High quality
        )

        self.assertGreater(
            score_low_quality.uncertainty_estimate,
            score_high_quality.uncertainty_estimate
        )

    def test_performance_history_update(self):
        """Test performance history updating"""
        model_name = "test_model"

        # Record a successful prediction
        self.scorer.update_performance_history(
            model_name=model_name,
            claimed_confidence=0.8,
            actual_outcome=True,
            investigation_type="person"
        )

        # Check history was updated
        self.assertIn(model_name, self.scorer.performance_history)
        self.assertEqual(
            self.scorer.performance_history[model_name]['total_predictions'],
            1
        )


class TestAIVotingSystem(unittest.TestCase):
    """Test AI Voting System"""

    def setUp(self):
        """Setup test environment"""
        self.voting_system = AIVotingSystem()

        # Create sample votes
        self.votes = [
            AIVote(
                model_name="Model1",
                recommendation="option_a",
                confidence=0.85,
                reasoning="Strong indicators"
            ),
            AIVote(
                model_name="Model2",
                recommendation="option_a",
                confidence=0.80,
                reasoning="Good signals"
            ),
            AIVote(
                model_name="Model3",
                recommendation="option_b",
                confidence=0.75,
                reasoning="Alternative approach"
            )
        ]

    def test_majority_voting(self):
        """Test simple majority voting"""
        result = self.voting_system.conduct_vote(
            self.votes,
            strategy=VotingStrategy.MAJORITY
        )

        self.assertEqual(result.winner, "option_a")  # 2 votes vs 1
        self.assertEqual(result.strategy_used, VotingStrategy.MAJORITY)

    def test_weighted_voting(self):
        """Test confidence-weighted voting"""
        result = self.voting_system.conduct_vote(
            self.votes,
            strategy=VotingStrategy.WEIGHTED
        )

        self.assertIsNotNone(result.winner)
        self.assertEqual(result.strategy_used, VotingStrategy.WEIGHTED)
        self.assertTrue(0 <= result.winning_confidence <= 1)

    def test_borda_count_voting(self):
        """Test Borda count voting with alternatives"""
        votes_with_alts = [
            AIVote(
                model_name="Model1",
                recommendation="option_a",
                confidence=0.85,
                reasoning="Primary choice",
                alternative_recommendations=[("option_b", 0.7), ("option_c", 0.6)]
            ),
            AIVote(
                model_name="Model2",
                recommendation="option_b",
                confidence=0.80,
                reasoning="Different view",
                alternative_recommendations=[("option_a", 0.75)]
            )
        ]

        result = self.voting_system.conduct_vote(
            votes_with_alts,
            strategy=VotingStrategy.BORDA_COUNT
        )

        self.assertIsNotNone(result.winner)
        self.assertEqual(result.strategy_used, VotingStrategy.BORDA_COUNT)

    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection"""
        result = self.voting_system.conduct_vote(
            self.votes,
            strategy=VotingStrategy.ADAPTIVE
        )

        self.assertIsNotNone(result.winner)
        # Strategy should be one of the concrete strategies
        self.assertNotEqual(result.strategy_used, VotingStrategy.ADAPTIVE)

    def test_consensus_calculation(self):
        """Test consensus level calculation"""
        # High consensus votes
        high_consensus_votes = [
            AIVote("M1", "option_a", 0.9, "Agree"),
            AIVote("M2", "option_a", 0.85, "Agree"),
            AIVote("M3", "option_a", 0.88, "Agree")
        ]

        result_high = self.voting_system.conduct_vote(high_consensus_votes)

        # Low consensus votes
        low_consensus_votes = [
            AIVote("M1", "option_a", 0.9, "Different"),
            AIVote("M2", "option_b", 0.5, "Different"),
            AIVote("M3", "option_c", 0.6, "Different")
        ]

        result_low = self.voting_system.conduct_vote(low_consensus_votes)

        self.assertGreater(result_high.consensus_level, result_low.consensus_level)

    def test_tie_breaking(self):
        """Test tie-breaking mechanism"""
        tied_votes = [
            AIVote("M1", "option_a", 0.9, "High conf"),
            AIVote("M2", "option_b", 0.7, "Lower conf")
        ]

        result = self.voting_system.conduct_vote(
            tied_votes,
            strategy=VotingStrategy.MAJORITY
        )

        # Should have tie and tie-break method
        self.assertTrue(result.tie_occurred)
        self.assertIsNotNone(result.tie_break_method)

    def test_vote_quality_assessment(self):
        """Test vote quality scoring"""
        result = self.voting_system.conduct_vote(self.votes)

        self.assertTrue(0 <= result.quality_score <= 1)


class TestAIPerformanceAnalytics(unittest.TestCase):
    """Test AI Performance Analytics"""

    def setUp(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = Path(self.temp_dir) / "test_performance.json"
        self.analytics = AIPerformanceAnalytics(data_file=self.data_file)

    def tearDown(self):
        """Cleanup"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_record_prediction(self):
        """Test recording a prediction"""
        self.analytics.record_prediction(
            model_name="TestModel",
            investigation_type="person",
            claimed_confidence=0.8,
            actual_outcome=True,
            response_time_ms=1000,
            tokens_used=100,
            cost_usd=0.003
        )

        self.assertEqual(len(self.analytics.metrics), 1)
        self.assertIn("TestModel", self.analytics.model_stats)

    def test_model_stats_calculation(self):
        """Test model statistics calculation"""
        # Record multiple predictions
        for i in range(10):
            self.analytics.record_prediction(
                model_name="TestModel",
                investigation_type="person",
                claimed_confidence=0.8,
                actual_outcome=i < 7,  # 70% accuracy
                response_time_ms=1000,
                tokens_used=100,
                cost_usd=0.003
            )

        stats = self.analytics.get_model_stats("TestModel")

        self.assertIsNotNone(stats)
        self.assertEqual(stats.total_predictions, 10)
        self.assertEqual(stats.correct_predictions, 7)
        self.assertEqual(stats.overall_accuracy, 0.7)

    def test_model_ranking(self):
        """Test model ranking"""
        # Record predictions for multiple models
        models = ["Model1", "Model2", "Model3"]

        for model in models:
            accuracy = {"Model1": 0.9, "Model2": 0.7, "Model3": 0.8}[model]

            for i in range(10):
                self.analytics.record_prediction(
                    model_name=model,
                    investigation_type="person",
                    claimed_confidence=0.8,
                    actual_outcome=(i / 10) < accuracy,
                    response_time_ms=1000,
                    tokens_used=100,
                    cost_usd=0.003
                )

        ranking = self.analytics.rank_models()

        # Model1 should rank first (highest accuracy)
        self.assertEqual(ranking.rankings["Model1"], 1)
        self.assertGreater(
            ranking.scores["Model1"],
            ranking.scores["Model2"]
        )

    def test_performance_degradation_detection(self):
        """Test performance degradation detection"""
        # Record improving performance
        for i in range(30):
            accuracy = 0.5 + (i / 30) * 0.3  # Improving from 50% to 80%
            self.analytics.record_prediction(
                model_name="ImprovingModel",
                investigation_type="person",
                claimed_confidence=0.8,
                actual_outcome=(i % 10) < (accuracy * 10),
                response_time_ms=1000,
                tokens_used=100,
                cost_usd=0.003
            )

        # Record degrading performance
        for i in range(30):
            accuracy = 0.8 - (i / 30) * 0.3  # Degrading from 80% to 50%
            self.analytics.record_prediction(
                model_name="DegradingModel",
                investigation_type="person",
                claimed_confidence=0.8,
                actual_outcome=(i % 10) < (accuracy * 10),
                response_time_ms=1000,
                tokens_used=100,
                cost_usd=0.003
            )

        # Check degradation
        is_degrading, reason = self.analytics.detect_performance_degradation("DegradingModel")

        self.assertTrue(is_degrading)
        self.assertIsNotNone(reason)

    def test_performance_report_generation(self):
        """Test comprehensive performance report"""
        # Record some predictions
        for i in range(20):
            self.analytics.record_prediction(
                model_name="TestModel",
                investigation_type="person",
                claimed_confidence=0.8,
                actual_outcome=i < 15,  # 75% accuracy
                response_time_ms=1000,
                tokens_used=100,
                cost_usd=0.003
            )

        report = self.analytics.get_performance_report()

        self.assertIn('summary', report)
        self.assertIn('models', report)
        self.assertIn('rankings', report)
        self.assertIn('recommendations', report)

        self.assertEqual(report['summary']['total_predictions'], 20)

    def test_data_persistence(self):
        """Test saving and loading performance data"""
        # Record predictions
        self.analytics.record_prediction(
            model_name="TestModel",
            investigation_type="person",
            claimed_confidence=0.8,
            actual_outcome=True,
            response_time_ms=1000,
            tokens_used=100,
            cost_usd=0.003
        )

        # Force save
        self.analytics._save_data()

        # Create new analytics instance (should load data)
        new_analytics = AIPerformanceAnalytics(data_file=self.data_file)

        self.assertEqual(len(new_analytics.metrics), 1)
        self.assertIn("TestModel", new_analytics.model_stats)


class TestIntegration(unittest.TestCase):
    """Integration tests for AI Enhancement components"""

    def test_confidence_and_voting_integration(self):
        """Test integration of confidence scoring and voting"""
        scorer = EnhancedConfidenceScorer()
        voting_system = AIVotingSystem()

        # Calculate confidence scores
        score1 = scorer.calculate_enhanced_confidence(
            "Model1", 0.85, [0.80, 0.75], {}
        )
        score2 = scorer.calculate_enhanced_confidence(
            "Model2", 0.80, [0.85, 0.75], {}
        )
        score3 = scorer.calculate_enhanced_confidence(
            "Model3", 0.75, [0.85, 0.80], {}
        )

        # Create votes using enhanced confidence scores
        votes = [
            AIVote("Model1", "option_a", score1.overall_score, "Enhanced"),
            AIVote("Model2", "option_a", score2.overall_score, "Enhanced"),
            AIVote("Model3", "option_b", score3.overall_score, "Enhanced")
        ]

        # Conduct vote
        result = voting_system.conduct_vote(votes)

        self.assertIsNotNone(result.winner)
        self.assertTrue(0 <= result.winning_confidence <= 1)

    def test_full_ai_enhancement_workflow(self):
        """Test complete AI enhancement workflow"""
        scorer = EnhancedConfidenceScorer()
        voting_system = AIVotingSystem()
        analytics = AIPerformanceAnalytics()

        # Simulate investigation with multiple models
        models = ["GPT-4", "Gemini", "Claude"]
        recommendations = ["deep_search", "social_media", "deep_search"]
        confidences = [0.85, 0.80, 0.90]

        # 1. Calculate enhanced confidence scores
        enhanced_scores = []
        for i, model in enumerate(models):
            other_confs = [c for j, c in enumerate(confidences) if j != i]
            score = scorer.calculate_enhanced_confidence(
                model, confidences[i], other_confs,
                {'investigation_type': 'person'}
            )
            enhanced_scores.append(score)

        # 2. Create votes
        votes = [
            AIVote(model, rec, score.overall_score, "Test")
            for model, rec, score in zip(models, recommendations, enhanced_scores)
        ]

        # 3. Conduct voting
        voting_result = voting_system.conduct_vote(votes)

        # 4. Record performance
        for model, rec, score in zip(models, recommendations, enhanced_scores):
            outcome = (rec == voting_result.winner)
            analytics.record_prediction(
                model_name=model,
                investigation_type="person",
                claimed_confidence=score.overall_score,
                actual_outcome=outcome,
                response_time_ms=1500,
                tokens_used=200,
                cost_usd=0.006
            )

        # Verify workflow completed
        self.assertIsNotNone(voting_result.winner)
        self.assertEqual(len(analytics.metrics), 3)
        self.assertEqual(len(analytics.model_stats), 3)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedConfidenceScorer))
    suite.addTests(loader.loadTestsFromTestCase(TestAIVotingSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestAIPerformanceAnalytics))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("🧪 AI ENHANCEMENT PHASE 3 TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped)}")
    print("="*80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
