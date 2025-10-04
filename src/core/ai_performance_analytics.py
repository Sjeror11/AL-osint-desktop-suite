#!/usr/bin/env python3
"""
📊 AI Performance Analytics - Model Performance Tracking & Optimization
OSINT Desktop Suite - Phase 3 AI Enhancement
LakyLuk Enhanced Edition - 4.10.2025

Features:
- Real-time performance tracking for each AI model
- Accuracy metrics by investigation type
- Response time analysis
- Cost tracking (API tokens/calls)
- Model comparison and ranking
- Performance degradation detection
- Adaptive model selection recommendations
"""

import json
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import math


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    timestamp: datetime
    model_name: str
    investigation_type: str
    claimed_confidence: float
    actual_outcome: bool  # True = correct, False = incorrect
    response_time_ms: float
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceStats:
    """Aggregated performance statistics for a model"""
    model_name: str
    total_predictions: int
    correct_predictions: int
    overall_accuracy: float
    avg_confidence: float
    avg_response_time_ms: float
    total_tokens_used: int
    total_cost_usd: float
    accuracy_by_type: Dict[str, float] = field(default_factory=dict)
    confidence_calibration_ratio: float = 1.0
    performance_trend: str = "stable"  # improving, stable, degrading
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'model_name': self.model_name,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'overall_accuracy': self.overall_accuracy,
            'avg_confidence': self.avg_confidence,
            'avg_response_time_ms': self.avg_response_time_ms,
            'total_tokens_used': self.total_tokens_used,
            'total_cost_usd': self.total_cost_usd,
            'accuracy_by_type': self.accuracy_by_type,
            'confidence_calibration_ratio': self.confidence_calibration_ratio,
            'performance_trend': self.performance_trend,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class ModelRanking:
    """Model ranking based on multiple criteria"""
    rankings: Dict[str, int]  # model_name -> rank
    scores: Dict[str, float]  # model_name -> composite score
    criteria_scores: Dict[str, Dict[str, float]]  # model -> criterion -> score
    best_for_type: Dict[str, str]  # investigation_type -> best_model
    recommendations: List[str] = field(default_factory=list)


class AIPerformanceAnalytics:
    """
    AI Performance Analytics System

    Tracks and analyzes AI model performance over time:
    - Accuracy tracking by investigation type
    - Response time monitoring
    - Cost analysis
    - Performance degradation detection
    - Model ranking and recommendations
    """

    def __init__(self, data_file: Optional[Path] = None, history_window: int = 100):
        """
        Initialize performance analytics

        Args:
            data_file: Path to performance data file
            history_window: Number of recent predictions to track
        """
        self.data_file = data_file or Path("ai_performance_data.json")
        self.history_window = history_window

        # Performance metrics storage
        self.metrics: List[PerformanceMetric] = []
        self.model_stats: Dict[str, ModelPerformanceStats] = {}

        # Recent performance for trend analysis
        self.recent_performance: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_window)
        )

        # Load historical data
        self._load_data()

    def record_prediction(
        self,
        model_name: str,
        investigation_type: str,
        claimed_confidence: float,
        actual_outcome: bool,
        response_time_ms: float,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        """
        Record a prediction and its outcome

        Args:
            model_name: Name of AI model
            investigation_type: Type of investigation
            claimed_confidence: Confidence model reported
            actual_outcome: Whether prediction was correct
            response_time_ms: Response time in milliseconds
            tokens_used: Number of API tokens used
            cost_usd: Cost in USD
            metadata: Additional metadata
        """
        # Create metric
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            model_name=model_name,
            investigation_type=investigation_type,
            claimed_confidence=claimed_confidence,
            actual_outcome=actual_outcome,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            metadata=metadata or {}
        )

        # Store metric
        self.metrics.append(metric)
        self.recent_performance[model_name].append(metric)

        # Update model statistics
        self._update_model_stats(model_name)

        # Save to disk periodically (every 10 predictions)
        if len(self.metrics) % 10 == 0:
            self._save_data()

    def get_model_stats(self, model_name: str) -> Optional[ModelPerformanceStats]:
        """Get performance statistics for a model"""
        return self.model_stats.get(model_name)

    def get_all_stats(self) -> Dict[str, ModelPerformanceStats]:
        """Get statistics for all models"""
        return self.model_stats.copy()

    def rank_models(
        self,
        criteria: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> ModelRanking:
        """
        Rank models based on multiple criteria

        Args:
            criteria: List of criteria to rank by
            weights: Weights for each criterion

        Returns:
            ModelRanking with rankings and recommendations
        """
        if not criteria:
            criteria = ['accuracy', 'response_time', 'cost_efficiency']

        if not weights:
            weights = {
                'accuracy': 0.5,
                'response_time': 0.3,
                'cost_efficiency': 0.2
            }

        # Calculate scores for each criterion
        criterion_scores = {}

        for criterion in criteria:
            if criterion == 'accuracy':
                criterion_scores['accuracy'] = self._score_by_accuracy()
            elif criterion == 'response_time':
                criterion_scores['response_time'] = self._score_by_response_time()
            elif criterion == 'cost_efficiency':
                criterion_scores['cost_efficiency'] = self._score_by_cost_efficiency()
            elif criterion == 'confidence_calibration':
                criterion_scores['confidence_calibration'] = self._score_by_calibration()

        # Calculate composite scores
        composite_scores = {}
        criteria_scores_dict = {}

        for model_name in self.model_stats.keys():
            criteria_scores_dict[model_name] = {}
            composite_score = 0.0

            for criterion, scores in criterion_scores.items():
                if model_name in scores:
                    score = scores[model_name]
                    criteria_scores_dict[model_name][criterion] = score
                    weight = weights.get(criterion, 0.0)
                    composite_score += score * weight

            composite_scores[model_name] = composite_score

        # Rank models
        sorted_models = sorted(
            composite_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        rankings = {
            model: rank + 1
            for rank, (model, _) in enumerate(sorted_models)
        }

        # Find best model for each investigation type
        best_for_type = {}
        for inv_type in self._get_investigation_types():
            best_model = self._find_best_for_type(inv_type)
            if best_model:
                best_for_type[inv_type] = best_model

        # Generate recommendations
        recommendations = self._generate_recommendations(rankings, composite_scores)

        return ModelRanking(
            rankings=rankings,
            scores=composite_scores,
            criteria_scores=criteria_scores_dict,
            best_for_type=best_for_type,
            recommendations=recommendations
        )

    def detect_performance_degradation(
        self,
        model_name: str,
        threshold: float = 0.1
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if model performance is degrading

        Args:
            model_name: Model to check
            threshold: Degradation threshold (0-1)

        Returns:
            (is_degrading, reason)
        """
        if model_name not in self.recent_performance:
            return False, None

        recent = list(self.recent_performance[model_name])

        if len(recent) < 20:  # Need sufficient data
            return False, "Insufficient data"

        # Split into two halves
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]

        # Compare accuracy
        first_accuracy = sum(1 for m in first_half if m.actual_outcome) / len(first_half)
        second_accuracy = sum(1 for m in second_half if m.actual_outcome) / len(second_half)

        accuracy_drop = first_accuracy - second_accuracy

        if accuracy_drop > threshold:
            return True, f"Accuracy dropped by {accuracy_drop:.1%}"

        # Compare response time
        first_avg_time = statistics.mean(m.response_time_ms for m in first_half)
        second_avg_time = statistics.mean(m.response_time_ms for m in second_half)

        time_increase = (second_avg_time - first_avg_time) / first_avg_time

        if time_increase > 0.5:  # 50% slower
            return True, f"Response time increased by {time_increase:.1%}"

        return False, None

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        # Overall statistics
        total_predictions = len(self.metrics)
        if total_predictions == 0:
            return {'error': 'No data available'}

        total_correct = sum(1 for m in self.metrics if m.actual_outcome)
        overall_accuracy = total_correct / total_predictions

        # Per-model statistics
        model_summaries = {}
        for model_name, stats in self.model_stats.items():
            # Check for degradation
            is_degrading, reason = self.detect_performance_degradation(model_name)

            model_summaries[model_name] = {
                **stats.to_dict(),
                'is_degrading': is_degrading,
                'degradation_reason': reason
            }

        # Rankings
        ranking = self.rank_models()

        # Cost analysis
        total_cost = sum(m.cost_usd for m in self.metrics)
        total_tokens = sum(m.tokens_used for m in self.metrics)

        # Time analysis
        avg_response_time = statistics.mean(m.response_time_ms for m in self.metrics)

        return {
            'summary': {
                'total_predictions': total_predictions,
                'overall_accuracy': overall_accuracy,
                'total_cost_usd': total_cost,
                'total_tokens': total_tokens,
                'avg_response_time_ms': avg_response_time
            },
            'models': model_summaries,
            'rankings': {
                'overall': ranking.rankings,
                'scores': ranking.scores,
                'best_for_type': ranking.best_for_type
            },
            'recommendations': ranking.recommendations
        }

    def _update_model_stats(self, model_name: str):
        """Update statistics for a model"""
        # Filter metrics for this model
        model_metrics = [m for m in self.metrics if m.model_name == model_name]

        if not model_metrics:
            return

        # Calculate statistics
        total = len(model_metrics)
        correct = sum(1 for m in model_metrics if m.actual_outcome)
        accuracy = correct / total

        avg_confidence = statistics.mean(m.claimed_confidence for m in model_metrics)
        avg_response_time = statistics.mean(m.response_time_ms for m in model_metrics)
        total_tokens = sum(m.tokens_used for m in model_metrics)
        total_cost = sum(m.cost_usd for m in model_metrics)

        # Accuracy by investigation type
        accuracy_by_type = {}
        types = set(m.investigation_type for m in model_metrics)

        for inv_type in types:
            type_metrics = [m for m in model_metrics if m.investigation_type == inv_type]
            type_correct = sum(1 for m in type_metrics if m.actual_outcome)
            accuracy_by_type[inv_type] = type_correct / len(type_metrics)

        # Confidence calibration
        if accuracy > 0:
            calibration_ratio = accuracy / avg_confidence
        else:
            calibration_ratio = 0.0

        # Performance trend
        trend = self._calculate_performance_trend(model_name)

        # Create/update stats
        self.model_stats[model_name] = ModelPerformanceStats(
            model_name=model_name,
            total_predictions=total,
            correct_predictions=correct,
            overall_accuracy=accuracy,
            avg_confidence=avg_confidence,
            avg_response_time_ms=avg_response_time,
            total_tokens_used=total_tokens,
            total_cost_usd=total_cost,
            accuracy_by_type=accuracy_by_type,
            confidence_calibration_ratio=calibration_ratio,
            performance_trend=trend
        )

    def _calculate_performance_trend(self, model_name: str) -> str:
        """Calculate performance trend (improving/stable/degrading)"""
        if model_name not in self.recent_performance:
            return "stable"

        recent = list(self.recent_performance[model_name])

        if len(recent) < 10:
            return "stable"

        # Split into thirds
        third = len(recent) // 3
        first_third = recent[:third]
        last_third = recent[-third:]

        # Compare accuracy
        first_acc = sum(1 for m in first_third if m.actual_outcome) / len(first_third)
        last_acc = sum(1 for m in last_third if m.actual_outcome) / len(last_third)

        diff = last_acc - first_acc

        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        else:
            return "stable"

    def _score_by_accuracy(self) -> Dict[str, float]:
        """Score models by accuracy"""
        return {
            name: stats.overall_accuracy
            for name, stats in self.model_stats.items()
        }

    def _score_by_response_time(self) -> Dict[str, float]:
        """Score models by response time (lower is better)"""
        times = {
            name: stats.avg_response_time_ms
            for name, stats in self.model_stats.items()
        }

        if not times:
            return {}

        # Invert and normalize (lower time = higher score)
        max_time = max(times.values())
        return {
            name: 1.0 - (time / max_time)
            for name, time in times.items()
        }

    def _score_by_cost_efficiency(self) -> Dict[str, float]:
        """Score models by cost efficiency"""
        efficiency = {}

        for name, stats in self.model_stats.items():
            if stats.total_cost_usd > 0 and stats.correct_predictions > 0:
                # Cost per correct prediction
                cost_per_correct = stats.total_cost_usd / stats.correct_predictions
                efficiency[name] = cost_per_correct
            else:
                efficiency[name] = 0.0

        if not efficiency or max(efficiency.values()) == 0:
            return {name: 0.5 for name in self.model_stats.keys()}

        # Invert and normalize
        max_cost = max(efficiency.values())
        return {
            name: 1.0 - (cost / max_cost)
            for name, cost in efficiency.items()
        }

    def _score_by_calibration(self) -> Dict[str, float]:
        """Score models by confidence calibration (1.0 = perfect)"""
        scores = {}

        for name, stats in self.model_stats.items():
            # Perfect calibration = 1.0
            # Score based on how close to 1.0
            calibration = stats.confidence_calibration_ratio
            score = 1.0 - abs(1.0 - calibration)
            scores[name] = max(0.0, score)

        return scores

    def _get_investigation_types(self) -> List[str]:
        """Get all investigation types"""
        types = set()
        for stats in self.model_stats.values():
            types.update(stats.accuracy_by_type.keys())
        return list(types)

    def _find_best_for_type(self, inv_type: str) -> Optional[str]:
        """Find best model for investigation type"""
        best_model = None
        best_accuracy = 0.0

        for name, stats in self.model_stats.items():
            if inv_type in stats.accuracy_by_type:
                accuracy = stats.accuracy_by_type[inv_type]
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = name

        return best_model

    def _generate_recommendations(
        self, rankings: Dict[str, int], scores: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on performance"""
        recommendations = []

        # Overall best model
        best_model = min(rankings.items(), key=lambda x: x[1])[0]
        recommendations.append(f"Best overall model: {best_model}")

        # Degrading models
        for name, stats in self.model_stats.items():
            if stats.performance_trend == "degrading":
                recommendations.append(
                    f"⚠️ {name} performance is degrading - consider alternatives"
                )

        # Improving models
        for name, stats in self.model_stats.items():
            if stats.performance_trend == "improving":
                recommendations.append(
                    f"✅ {name} performance is improving"
                )

        # Cost warnings
        for name, stats in self.model_stats.items():
            if stats.total_predictions > 0:
                cost_per_prediction = stats.total_cost_usd / stats.total_predictions
                if cost_per_prediction > 0.10:  # $0.10 per prediction
                    recommendations.append(
                        f"💰 {name} has high cost per prediction (${cost_per_prediction:.3f})"
                    )

        return recommendations

    def _load_data(self):
        """Load performance data from file"""
        if not self.data_file.exists():
            return

        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            # Load metrics (convert timestamps)
            for metric_data in data.get('metrics', []):
                metric_data['timestamp'] = datetime.fromisoformat(metric_data['timestamp'])
                metric = PerformanceMetric(**metric_data)
                self.metrics.append(metric)
                self.recent_performance[metric.model_name].append(metric)

            # Rebuild stats
            for model_name in set(m.model_name for m in self.metrics):
                self._update_model_stats(model_name)

        except Exception as e:
            print(f"Error loading performance data: {e}")

    def _save_data(self):
        """Save performance data to file"""
        try:
            # Convert metrics to serializable format
            metrics_data = []
            for metric in self.metrics:
                metric_dict = asdict(metric)
                metric_dict['timestamp'] = metric.timestamp.isoformat()
                metrics_data.append(metric_dict)

            data = {'metrics': metrics_data}

            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error saving performance data: {e}")


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("📊 AI PERFORMANCE ANALYTICS - DEMONSTRATION")
    print("="*80)

    # Create analytics system
    analytics = AIPerformanceAnalytics()

    # Simulate some predictions
    models = ["GPT-4", "Gemini", "Claude"]
    inv_types = ["person", "business", "location"]

    import random

    print("\n📝 Recording sample predictions...")
    for i in range(50):
        model = random.choice(models)
        inv_type = random.choice(inv_types)
        confidence = random.uniform(0.6, 0.95)
        outcome = random.random() < confidence  # Higher confidence = more likely correct
        response_time = random.uniform(500, 2000)
        tokens = random.randint(100, 500)
        cost = tokens * 0.00003  # ~$0.03 per 1k tokens

        analytics.record_prediction(
            model_name=model,
            investigation_type=inv_type,
            claimed_confidence=confidence,
            actual_outcome=outcome,
            response_time_ms=response_time,
            tokens_used=tokens,
            cost_usd=cost
        )

    print("✅ Recorded 50 predictions\n")

    # Get performance report
    report = analytics.get_performance_report()

    print("="*80)
    print("📊 PERFORMANCE REPORT")
    print("="*80)

    print(f"\n📈 Overall Summary:")
    for key, value in report['summary'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print(f"\n🏆 Model Rankings:")
    for model, rank in sorted(report['rankings']['overall'].items(), key=lambda x: x[1]):
        score = report['rankings']['scores'][model]
        print(f"  #{rank}. {model}: {score:.3f}")

    print(f"\n🎯 Best for Each Type:")
    for inv_type, model in report['rankings']['best_for_type'].items():
        print(f"  {inv_type}: {model}")

    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

    print("\n" + "="*80)
