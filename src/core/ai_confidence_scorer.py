#!/usr/bin/env python3
"""
🎯 AI Confidence Scoring System - Advanced Multi-Model Confidence Analysis
OSINT Desktop Suite - Phase 3 AI Enhancement
LakyLuk Enhanced Edition - 4.10.2025

Features:
- Multi-dimensional confidence scoring
- Bayesian confidence aggregation
- Historical performance weighting
- Context-aware confidence adjustment
- Uncertainty quantification
- Ensemble agreement analysis
"""

import math
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path


class ConfidenceMetric(Enum):
    """Types of confidence metrics"""
    MODEL_INTRINSIC = "model_intrinsic"  # Model's own confidence
    HISTORICAL_ACCURACY = "historical_accuracy"  # Past performance
    CONSENSUS_AGREEMENT = "consensus_agreement"  # Agreement with other models
    DATA_QUALITY = "data_quality"  # Quality of input data
    CONTEXT_RELEVANCE = "context_relevance"  # Relevance to investigation
    TEMPORAL_CONSISTENCY = "temporal_consistency"  # Consistency over time
    SOURCE_RELIABILITY = "source_reliability"  # Reliability of data sources


@dataclass
class ConfidenceFactors:
    """Individual confidence factors for analysis"""
    model_confidence: float = 0.5  # 0-1
    historical_accuracy: float = 0.5  # 0-1
    consensus_agreement: float = 0.5  # 0-1
    data_quality: float = 0.5  # 0-1
    context_relevance: float = 0.5  # 0-1
    temporal_consistency: float = 0.5  # 0-1
    source_reliability: float = 0.5  # 0-1

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'model_confidence': self.model_confidence,
            'historical_accuracy': self.historical_accuracy,
            'consensus_agreement': self.consensus_agreement,
            'data_quality': self.data_quality,
            'context_relevance': self.context_relevance,
            'temporal_consistency': self.temporal_consistency,
            'source_reliability': self.source_reliability
        }


@dataclass
class EnhancedConfidenceScore:
    """Enhanced confidence score with detailed breakdown"""
    overall_score: float  # 0-1
    certainty_level: str  # very_low, low, medium, high, very_high
    confidence_factors: ConfidenceFactors
    uncertainty_estimate: float  # 0-1 (higher = more uncertain)
    confidence_interval: Tuple[float, float]  # (lower, upper) bounds
    contributing_factors: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived metrics"""
        self.certainty_level = self._determine_certainty_level()

    def _determine_certainty_level(self) -> str:
        """Determine certainty level from overall score"""
        if self.overall_score >= 0.9:
            return "very_high"
        elif self.overall_score >= 0.75:
            return "high"
        elif self.overall_score >= 0.5:
            return "medium"
        elif self.overall_score >= 0.25:
            return "low"
        else:
            return "very_low"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'overall_score': self.overall_score,
            'certainty_level': self.certainty_level,
            'confidence_factors': self.confidence_factors.to_dict(),
            'uncertainty_estimate': self.uncertainty_estimate,
            'confidence_interval': list(self.confidence_interval),
            'contributing_factors': self.contributing_factors,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


class EnhancedConfidenceScorer:
    """
    Advanced Confidence Scoring System

    Provides multi-dimensional confidence analysis using:
    - Bayesian confidence aggregation
    - Historical performance tracking
    - Ensemble agreement analysis
    - Context-aware adjustments
    - Uncertainty quantification
    """

    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize confidence scorer

        Args:
            history_file: Path to historical performance data
        """
        self.history_file = history_file or Path("ai_performance_history.json")

        # Default weights for confidence factors
        self.factor_weights = {
            ConfidenceMetric.MODEL_INTRINSIC: 0.25,
            ConfidenceMetric.HISTORICAL_ACCURACY: 0.20,
            ConfidenceMetric.CONSENSUS_AGREEMENT: 0.20,
            ConfidenceMetric.DATA_QUALITY: 0.15,
            ConfidenceMetric.CONTEXT_RELEVANCE: 0.10,
            ConfidenceMetric.TEMPORAL_CONSISTENCY: 0.05,
            ConfidenceMetric.SOURCE_RELIABILITY: 0.05
        }

        # Load historical performance data
        self.performance_history = self._load_performance_history()

    def calculate_enhanced_confidence(
        self,
        model_name: str,
        intrinsic_confidence: float,
        other_confidences: List[float],
        context: Optional[Dict[str, Any]] = None
    ) -> EnhancedConfidenceScore:
        """
        Calculate enhanced confidence score

        Args:
            model_name: Name of AI model
            intrinsic_confidence: Model's self-reported confidence (0-1)
            other_confidences: Confidences from other models
            context: Optional context information

        Returns:
            EnhancedConfidenceScore with detailed breakdown
        """
        context = context or {}

        # Calculate individual confidence factors
        factors = ConfidenceFactors()

        # 1. Model intrinsic confidence
        factors.model_confidence = self._calibrate_model_confidence(
            model_name, intrinsic_confidence
        )

        # 2. Historical accuracy
        factors.historical_accuracy = self._calculate_historical_accuracy(
            model_name, context.get('investigation_type')
        )

        # 3. Consensus agreement
        factors.consensus_agreement = self._calculate_consensus_agreement(
            intrinsic_confidence, other_confidences
        )

        # 4. Data quality
        factors.data_quality = self._assess_data_quality(
            context.get('data_sources', []),
            context.get('data_completeness', 0.5)
        )

        # 5. Context relevance
        factors.context_relevance = self._assess_context_relevance(
            model_name, context.get('investigation_type'), context.get('task_type')
        )

        # 6. Temporal consistency
        factors.temporal_consistency = self._assess_temporal_consistency(
            model_name, intrinsic_confidence
        )

        # 7. Source reliability
        factors.source_reliability = self._assess_source_reliability(
            context.get('data_sources', [])
        )

        # Calculate weighted overall score using Bayesian aggregation
        overall_score = self._bayesian_aggregate_confidence(factors)

        # Calculate uncertainty estimate
        uncertainty = self._estimate_uncertainty(factors, other_confidences)

        # Calculate confidence interval (95%)
        confidence_interval = self._calculate_confidence_interval(
            overall_score, uncertainty
        )

        # Determine contributing factors
        contributing_factors = self._rank_contributing_factors(factors)

        # Generate warnings
        warnings = self._generate_confidence_warnings(
            overall_score, factors, uncertainty
        )

        # Create enhanced confidence score
        enhanced_score = EnhancedConfidenceScore(
            overall_score=overall_score,
            certainty_level="",  # Will be set in __post_init__
            confidence_factors=factors,
            uncertainty_estimate=uncertainty,
            confidence_interval=confidence_interval,
            contributing_factors=contributing_factors,
            warnings=warnings,
            metadata={
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'context': context
            }
        )

        return enhanced_score

    def _calibrate_model_confidence(
        self, model_name: str, raw_confidence: float
    ) -> float:
        """
        Calibrate model confidence based on historical performance

        Some models tend to be over/under-confident
        """
        if model_name not in self.performance_history:
            return raw_confidence

        # Get calibration factor from historical data
        history = self.performance_history[model_name]
        avg_claimed = history.get('avg_claimed_confidence', 0.5)
        avg_actual = history.get('avg_actual_accuracy', 0.5)

        # Calculate calibration ratio
        if avg_claimed > 0:
            calibration_ratio = avg_actual / avg_claimed
        else:
            calibration_ratio = 1.0

        # Apply calibration with smoothing
        calibrated = raw_confidence * (0.7 * calibration_ratio + 0.3)

        # Ensure bounds
        return max(0.0, min(1.0, calibrated))

    def _calculate_historical_accuracy(
        self, model_name: str, investigation_type: Optional[str]
    ) -> float:
        """Calculate historical accuracy score"""
        if model_name not in self.performance_history:
            return 0.5  # Default for new models

        history = self.performance_history[model_name]

        # Overall accuracy
        overall_accuracy = history.get('avg_actual_accuracy', 0.5)

        # Type-specific accuracy if available
        if investigation_type and 'type_accuracies' in history:
            type_accuracy = history['type_accuracies'].get(investigation_type, overall_accuracy)
            # Weight type-specific more heavily
            return 0.7 * type_accuracy + 0.3 * overall_accuracy

        return overall_accuracy

    def _calculate_consensus_agreement(
        self, this_confidence: float, other_confidences: List[float]
    ) -> float:
        """
        Calculate consensus agreement score

        Higher when models agree, lower when they disagree
        """
        if not other_confidences:
            return 0.5  # No other models to compare

        all_confidences = [this_confidence] + other_confidences

        # Calculate variance (lower variance = higher agreement)
        if len(all_confidences) < 2:
            return 0.5

        variance = statistics.variance(all_confidences)

        # Convert variance to agreement score (0-1)
        # Max variance is 0.25 (when values are 0 and 1)
        max_variance = 0.25
        agreement = 1.0 - min(variance / max_variance, 1.0)

        return agreement

    def _assess_data_quality(
        self, data_sources: List[str], completeness: float
    ) -> float:
        """Assess quality of input data"""
        # Base quality from completeness
        quality = completeness

        # Adjust for number of sources (more sources = higher quality)
        source_count = len(data_sources)
        if source_count == 0:
            source_bonus = 0.0
        elif source_count <= 2:
            source_bonus = 0.1
        elif source_count <= 5:
            source_bonus = 0.2
        else:
            source_bonus = 0.3

        quality = min(1.0, quality + source_bonus)

        return quality

    def _assess_context_relevance(
        self, model_name: str, investigation_type: Optional[str], task_type: Optional[str]
    ) -> float:
        """
        Assess how relevant the model is for this context

        Different models excel at different tasks
        """
        # Model specializations (these would ideally be learned from data)
        model_strengths = {
            'gpt4': {'technical': 0.9, 'person': 0.8, 'business': 0.85},
            'gemini': {'context': 0.9, 'correlation': 0.85, 'business': 0.8},
            'claude': {'strategic': 0.9, 'person': 0.85, 'analysis': 0.9}
        }

        model_key = model_name.lower().replace('-', '').replace('_', '')

        if model_key not in model_strengths:
            return 0.75  # Default

        strengths = model_strengths[model_key]

        # Check investigation type relevance
        relevance = 0.75  # Default
        if investigation_type:
            type_key = investigation_type.lower()
            relevance = strengths.get(type_key, 0.75)

        # Check task type relevance
        if task_type:
            task_key = task_type.lower()
            task_relevance = strengths.get(task_key, 0.75)
            relevance = 0.6 * relevance + 0.4 * task_relevance

        return relevance

    def _assess_temporal_consistency(
        self, model_name: str, current_confidence: float
    ) -> float:
        """
        Assess temporal consistency of model confidence

        Penalize models that change confidence drastically
        """
        if model_name not in self.performance_history:
            return 0.75  # Default for new models

        history = self.performance_history[model_name]
        recent_confidences = history.get('recent_confidences', [])

        if len(recent_confidences) < 2:
            return 0.75

        # Calculate variance of recent confidences
        variance = statistics.variance(recent_confidences + [current_confidence])

        # Lower variance = higher consistency
        max_variance = 0.1  # Reasonable threshold
        consistency = 1.0 - min(variance / max_variance, 1.0)

        return consistency

    def _assess_source_reliability(self, data_sources: List[str]) -> float:
        """Assess reliability of data sources"""
        if not data_sources:
            return 0.5

        # Source reliability ratings (would ideally be maintained in a database)
        source_ratings = {
            'official_records': 0.95,
            'government_database': 0.9,
            'verified_social_media': 0.75,
            'public_records': 0.8,
            'web_search': 0.6,
            'unverified_source': 0.4
        }

        # Calculate average reliability
        total_reliability = 0.0
        for source in data_sources:
            source_key = source.lower().replace(' ', '_')
            total_reliability += source_ratings.get(source_key, 0.5)

        avg_reliability = total_reliability / len(data_sources)

        return avg_reliability

    def _bayesian_aggregate_confidence(self, factors: ConfidenceFactors) -> float:
        """
        Aggregate confidence factors using Bayesian approach

        Combines multiple confidence signals with appropriate weighting
        """
        # Extract factor values
        factor_values = {
            ConfidenceMetric.MODEL_INTRINSIC: factors.model_confidence,
            ConfidenceMetric.HISTORICAL_ACCURACY: factors.historical_accuracy,
            ConfidenceMetric.CONSENSUS_AGREEMENT: factors.consensus_agreement,
            ConfidenceMetric.DATA_QUALITY: factors.data_quality,
            ConfidenceMetric.CONTEXT_RELEVANCE: factors.context_relevance,
            ConfidenceMetric.TEMPORAL_CONSISTENCY: factors.temporal_consistency,
            ConfidenceMetric.SOURCE_RELIABILITY: factors.source_reliability
        }

        # Calculate weighted sum
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, value in factor_values.items():
            weight = self.factor_weights.get(metric, 0.1)
            weighted_sum += value * weight
            total_weight += weight

        # Normalize
        if total_weight > 0:
            aggregated = weighted_sum / total_weight
        else:
            aggregated = 0.5

        return max(0.0, min(1.0, aggregated))

    def _estimate_uncertainty(
        self, factors: ConfidenceFactors, other_confidences: List[float]
    ) -> float:
        """
        Estimate uncertainty in confidence score

        Higher uncertainty when:
        - Low data quality
        - High disagreement between models
        - Low historical accuracy
        """
        uncertainty_factors = []

        # Data quality uncertainty
        data_uncertainty = 1.0 - factors.data_quality
        uncertainty_factors.append(data_uncertainty * 0.3)

        # Consensus uncertainty
        if other_confidences:
            variance = statistics.variance([factors.model_confidence] + other_confidences)
            consensus_uncertainty = min(variance * 4, 1.0)  # Scale variance
            uncertainty_factors.append(consensus_uncertainty * 0.3)

        # Historical accuracy uncertainty
        history_uncertainty = 1.0 - factors.historical_accuracy
        uncertainty_factors.append(history_uncertainty * 0.2)

        # Context relevance uncertainty
        context_uncertainty = 1.0 - factors.context_relevance
        uncertainty_factors.append(context_uncertainty * 0.2)

        # Aggregate uncertainties
        total_uncertainty = sum(uncertainty_factors)

        return max(0.0, min(1.0, total_uncertainty))

    def _calculate_confidence_interval(
        self, confidence: float, uncertainty: float
    ) -> Tuple[float, float]:
        """Calculate 95% confidence interval"""
        # Use uncertainty to determine interval width
        # Higher uncertainty = wider interval
        margin = uncertainty * 0.5  # Max margin of ±0.5

        lower = max(0.0, confidence - margin)
        upper = min(1.0, confidence + margin)

        return (lower, upper)

    def _rank_contributing_factors(self, factors: ConfidenceFactors) -> Dict[str, float]:
        """Rank confidence factors by contribution to final score"""
        factor_values = factors.to_dict()

        # Weight each factor by its configured weight
        weighted_factors = {}
        for metric in ConfidenceMetric:
            metric_key = metric.value
            if metric_key in factor_values:
                value = factor_values[metric_key]
                weight = self.factor_weights.get(metric, 0.1)
                weighted_factors[metric_key] = value * weight

        # Sort by contribution
        sorted_factors = dict(
            sorted(weighted_factors.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_factors

    def _generate_confidence_warnings(
        self, overall_score: float, factors: ConfidenceFactors, uncertainty: float
    ) -> List[str]:
        """Generate warnings about confidence issues"""
        warnings = []

        if overall_score < 0.3:
            warnings.append("Very low confidence - results should be carefully validated")

        if uncertainty > 0.7:
            warnings.append("High uncertainty in confidence estimate")

        if factors.data_quality < 0.4:
            warnings.append("Low data quality detected")

        if factors.consensus_agreement < 0.4:
            warnings.append("Low agreement between AI models")

        if factors.historical_accuracy < 0.5:
            warnings.append("Model has below-average historical accuracy")

        if factors.source_reliability < 0.5:
            warnings.append("Data sources have questionable reliability")

        return warnings

    def _load_performance_history(self) -> Dict:
        """Load historical performance data"""
        if not self.history_file.exists():
            return {}

        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_performance_history(self):
        """Save performance history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            print(f"Error saving performance history: {e}")

    def update_performance_history(
        self, model_name: str, claimed_confidence: float,
        actual_outcome: bool, investigation_type: Optional[str] = None
    ):
        """
        Update performance history with new result

        Args:
            model_name: Name of model
            claimed_confidence: Confidence model claimed
            actual_outcome: Whether prediction was correct
            investigation_type: Type of investigation
        """
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {
                'avg_claimed_confidence': 0.0,
                'avg_actual_accuracy': 0.0,
                'total_predictions': 0,
                'recent_confidences': [],
                'type_accuracies': {}
            }

        history = self.performance_history[model_name]

        # Update overall stats
        n = history['total_predictions']
        history['avg_claimed_confidence'] = (
            (history['avg_claimed_confidence'] * n + claimed_confidence) / (n + 1)
        )

        actual_accuracy = 1.0 if actual_outcome else 0.0
        history['avg_actual_accuracy'] = (
            (history['avg_actual_accuracy'] * n + actual_accuracy) / (n + 1)
        )

        history['total_predictions'] += 1

        # Update recent confidences (keep last 10)
        history['recent_confidences'].append(claimed_confidence)
        if len(history['recent_confidences']) > 10:
            history['recent_confidences'] = history['recent_confidences'][-10:]

        # Update type-specific accuracy
        if investigation_type:
            if investigation_type not in history['type_accuracies']:
                history['type_accuracies'][investigation_type] = actual_accuracy
            else:
                old_accuracy = history['type_accuracies'][investigation_type]
                # Simple exponential moving average
                history['type_accuracies'][investigation_type] = (
                    0.9 * old_accuracy + 0.1 * actual_accuracy
                )

        # Save to disk
        self.save_performance_history()


# Example usage and testing
if __name__ == "__main__":
    # Create confidence scorer
    scorer = EnhancedConfidenceScorer()

    # Example: Calculate confidence for GPT-4
    enhanced_score = scorer.calculate_enhanced_confidence(
        model_name="gpt4",
        intrinsic_confidence=0.85,
        other_confidences=[0.75, 0.80],  # From Gemini and Claude
        context={
            'investigation_type': 'person',
            'task_type': 'technical',
            'data_sources': ['official_records', 'verified_social_media'],
            'data_completeness': 0.8
        }
    )

    print("="*80)
    print("🎯 ENHANCED CONFIDENCE SCORE EXAMPLE")
    print("="*80)
    print(f"\nOverall Score: {enhanced_score.overall_score:.3f}")
    print(f"Certainty Level: {enhanced_score.certainty_level}")
    print(f"Uncertainty: {enhanced_score.uncertainty_estimate:.3f}")
    print(f"Confidence Interval: {enhanced_score.confidence_interval[0]:.3f} - {enhanced_score.confidence_interval[1]:.3f}")

    print("\n📊 Confidence Factors:")
    for factor, value in enhanced_score.confidence_factors.to_dict().items():
        print(f"  {factor}: {value:.3f}")

    print("\n🎖️  Contributing Factors (ranked):")
    for factor, contribution in enhanced_score.contributing_factors.items():
        print(f"  {factor}: {contribution:.3f}")

    if enhanced_score.warnings:
        print("\n⚠️  Warnings:")
        for warning in enhanced_score.warnings:
            print(f"  - {warning}")

    print("\n" + "="*80)

    # Simulate updating performance history
    print("\n📈 Updating performance history...")
    scorer.update_performance_history(
        model_name="gpt4",
        claimed_confidence=0.85,
        actual_outcome=True,
        investigation_type="person"
    )
    print("✅ Performance history updated")
