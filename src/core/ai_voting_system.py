#!/usr/bin/env python3
"""
🗳️ AI Voting System - Advanced Multi-Model Decision Aggregation
OSINT Desktop Suite - Phase 3 AI Enhancement
LakyLuk Enhanced Edition - 4.10.2025

Features:
- Multiple voting strategies (majority, weighted, Borda count, approval)
- Confidence-weighted voting
- Adaptive strategy selection
- Consensus detection and tie-breaking
- Vote quality assessment
- Strategic voting prevention
"""

import math
import statistics
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import Counter, defaultdict


class VotingStrategy(Enum):
    """Available voting strategies"""
    MAJORITY = "majority"  # Simple majority vote
    WEIGHTED = "weighted"  # Confidence-weighted voting
    BORDA_COUNT = "borda_count"  # Ranked preference voting
    APPROVAL = "approval"  # Approval voting (threshold-based)
    CONDORCET = "condorcet"  # Pairwise comparison
    ADAPTIVE = "adaptive"  # Auto-select best strategy


@dataclass
class AIVote:
    """Individual AI model vote"""
    model_name: str
    recommendation: str
    confidence: float
    reasoning: str
    alternative_recommendations: List[Tuple[str, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate vote"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass
class VotingResult:
    """Result of voting process"""
    winner: str
    winning_confidence: float
    strategy_used: VotingStrategy
    vote_distribution: Dict[str, float]
    consensus_level: float  # 0-1 (higher = more agreement)
    tie_occurred: bool
    tie_break_method: Optional[str]
    individual_votes: List[AIVote]
    quality_score: float  # 0-1 (quality of voting process)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'winner': self.winner,
            'winning_confidence': self.winning_confidence,
            'strategy_used': self.strategy_used.value,
            'vote_distribution': self.vote_distribution,
            'consensus_level': self.consensus_level,
            'tie_occurred': self.tie_occurred,
            'tie_break_method': self.tie_break_method,
            'quality_score': self.quality_score,
            'warnings': self.warnings,
            'metadata': self.metadata,
            'vote_count': len(self.individual_votes)
        }


class AIVotingSystem:
    """
    Advanced AI Voting System

    Implements multiple voting strategies for aggregating
    AI model decisions with proper handling of:
    - Confidence weighting
    - Tie-breaking
    - Consensus detection
    - Vote quality assessment
    """

    def __init__(self, default_strategy: VotingStrategy = VotingStrategy.ADAPTIVE):
        """
        Initialize voting system

        Args:
            default_strategy: Default voting strategy to use
        """
        self.default_strategy = default_strategy

        # Strategy selection criteria
        self.strategy_thresholds = {
            'high_consensus': 0.8,  # Use simple majority if consensus is high
            'low_confidence': 0.5,  # Use weighted if confidence is mixed
            'many_options': 4  # Use Borda count if many alternatives
        }

    def conduct_vote(
        self,
        votes: List[AIVote],
        strategy: Optional[VotingStrategy] = None,
        confidence_threshold: float = 0.5
    ) -> VotingResult:
        """
        Conduct vote using specified strategy

        Args:
            votes: List of AI votes
            strategy: Voting strategy to use (None = use default)
            confidence_threshold: Minimum confidence for approval voting

        Returns:
            VotingResult with winner and details
        """
        if not votes:
            raise ValueError("No votes provided")

        # Select strategy
        if strategy is None:
            strategy = self.default_strategy

        if strategy == VotingStrategy.ADAPTIVE:
            strategy = self._select_adaptive_strategy(votes)

        # Execute voting strategy
        if strategy == VotingStrategy.MAJORITY:
            result = self._majority_vote(votes)
        elif strategy == VotingStrategy.WEIGHTED:
            result = self._weighted_vote(votes)
        elif strategy == VotingStrategy.BORDA_COUNT:
            result = self._borda_count_vote(votes)
        elif strategy == VotingStrategy.APPROVAL:
            result = self._approval_vote(votes, confidence_threshold)
        elif strategy == VotingStrategy.CONDORCET:
            result = self._condorcet_vote(votes)
        else:
            # Fallback to weighted
            result = self._weighted_vote(votes)

        # Calculate consensus level
        consensus = self._calculate_consensus_level(votes, result.winner)

        # Assess vote quality
        quality = self._assess_vote_quality(votes, result)

        # Generate warnings
        warnings = self._generate_warnings(votes, result, consensus, quality)

        # Update result
        result.consensus_level = consensus
        result.quality_score = quality
        result.warnings = warnings
        result.strategy_used = strategy

        return result

    def _majority_vote(self, votes: List[AIVote]) -> VotingResult:
        """Simple majority voting"""
        # Count votes for each recommendation
        vote_counts = Counter([v.recommendation for v in votes])

        # Find winner
        winner, winner_count = vote_counts.most_common(1)[0]

        # Check for ties
        all_counts = vote_counts.most_common()
        tie_occurred = len([c for _, c in all_counts if c == winner_count]) > 1

        tie_break_method = None
        if tie_occurred:
            # Break tie using average confidence
            tied_options = [opt for opt, c in all_counts if c == winner_count]
            winner, tie_break_method = self._break_tie_by_confidence(
                votes, tied_options
            )

        # Calculate winning confidence (average of votes for winner)
        winner_votes = [v for v in votes if v.recommendation == winner]
        winning_confidence = sum(v.confidence for v in winner_votes) / len(winner_votes)

        # Vote distribution
        total_votes = len(votes)
        vote_distribution = {
            option: count / total_votes
            for option, count in vote_counts.items()
        }

        return VotingResult(
            winner=winner,
            winning_confidence=winning_confidence,
            strategy_used=VotingStrategy.MAJORITY,
            vote_distribution=vote_distribution,
            consensus_level=0.0,  # Will be calculated later
            tie_occurred=tie_occurred,
            tie_break_method=tie_break_method,
            individual_votes=votes,
            quality_score=0.0  # Will be calculated later
        )

    def _weighted_vote(self, votes: List[AIVote]) -> VotingResult:
        """Confidence-weighted voting"""
        # Calculate weighted scores for each recommendation
        weighted_scores = defaultdict(float)
        total_weight = 0.0

        for vote in votes:
            weighted_scores[vote.recommendation] += vote.confidence
            total_weight += vote.confidence

        # Find winner
        winner = max(weighted_scores.items(), key=lambda x: x[1])[0]
        winner_score = weighted_scores[winner]

        # Check for ties (within 5% of winner score)
        tie_threshold = winner_score * 0.95
        tied_options = [
            opt for opt, score in weighted_scores.items()
            if score >= tie_threshold
        ]

        tie_occurred = len(tied_options) > 1
        tie_break_method = None

        if tie_occurred:
            winner, tie_break_method = self._break_tie_by_vote_count(
                votes, tied_options
            )

        # Winning confidence (weighted average)
        winner_votes = [v for v in votes if v.recommendation == winner]
        winning_confidence = sum(v.confidence for v in winner_votes) / len(winner_votes)

        # Vote distribution (normalized)
        vote_distribution = {
            option: score / total_weight
            for option, score in weighted_scores.items()
        }

        return VotingResult(
            winner=winner,
            winning_confidence=winning_confidence,
            strategy_used=VotingStrategy.WEIGHTED,
            vote_distribution=vote_distribution,
            consensus_level=0.0,
            tie_occurred=tie_occurred,
            tie_break_method=tie_break_method,
            individual_votes=votes,
            quality_score=0.0
        )

    def _borda_count_vote(self, votes: List[AIVote]) -> VotingResult:
        """Borda count voting (ranked preferences)"""
        # Calculate Borda count scores
        borda_scores = defaultdict(float)

        for vote in votes:
            # Primary recommendation gets full points
            all_recommendations = [(vote.recommendation, vote.confidence)]
            all_recommendations.extend(vote.alternative_recommendations)

            # Sort by confidence
            all_recommendations.sort(key=lambda x: x[1], reverse=True)

            # Assign Borda points
            n = len(all_recommendations)
            for rank, (rec, conf) in enumerate(all_recommendations):
                points = (n - rank) * conf  # Weight by confidence
                borda_scores[rec] += points

        # Find winner
        winner = max(borda_scores.items(), key=lambda x: x[1])[0]

        # Check for ties
        winner_score = borda_scores[winner]
        tie_threshold = winner_score * 0.95
        tied_options = [
            opt for opt, score in borda_scores.items()
            if score >= tie_threshold
        ]

        tie_occurred = len(tied_options) > 1
        tie_break_method = None

        if tie_occurred:
            winner, tie_break_method = self._break_tie_by_confidence(
                votes, tied_options
            )

        # Winning confidence
        winner_votes = [v for v in votes if v.recommendation == winner]
        winning_confidence = sum(v.confidence for v in winner_votes) / len(winner_votes)

        # Vote distribution
        total_score = sum(borda_scores.values())
        vote_distribution = {
            option: score / total_score
            for option, score in borda_scores.items()
        }

        return VotingResult(
            winner=winner,
            winning_confidence=winning_confidence,
            strategy_used=VotingStrategy.BORDA_COUNT,
            vote_distribution=vote_distribution,
            consensus_level=0.0,
            tie_occurred=tie_occurred,
            tie_break_method=tie_break_method,
            individual_votes=votes,
            quality_score=0.0
        )

    def _approval_vote(self, votes: List[AIVote], threshold: float) -> VotingResult:
        """Approval voting (approve all above threshold)"""
        # Count approvals for each recommendation
        approvals = defaultdict(int)
        approval_confidence = defaultdict(list)

        for vote in votes:
            # Approve primary if above threshold
            if vote.confidence >= threshold:
                approvals[vote.recommendation] += 1
                approval_confidence[vote.recommendation].append(vote.confidence)

            # Approve alternatives if above threshold
            for rec, conf in vote.alternative_recommendations:
                if conf >= threshold:
                    approvals[rec] += 1
                    approval_confidence[rec].append(conf)

        if not approvals:
            # No recommendations above threshold - use highest confidence
            all_votes_sorted = sorted(votes, key=lambda v: v.confidence, reverse=True)
            winner = all_votes_sorted[0].recommendation
            winning_confidence = all_votes_sorted[0].confidence
            tie_occurred = False
            tie_break_method = "highest_confidence"
        else:
            # Find winner
            winner = max(approvals.items(), key=lambda x: x[1])[0]
            winner_approvals = approvals[winner]

            # Check for ties
            tied_options = [
                opt for opt, count in approvals.items()
                if count == winner_approvals
            ]

            tie_occurred = len(tied_options) > 1
            tie_break_method = None

            if tie_occurred:
                winner, tie_break_method = self._break_tie_by_confidence(
                    votes, tied_options
                )

            # Winning confidence
            winning_confidence = statistics.mean(approval_confidence[winner])

        # Vote distribution
        total_approvals = sum(approvals.values()) if approvals else 1
        vote_distribution = {
            option: count / total_approvals
            for option, count in approvals.items()
        }

        return VotingResult(
            winner=winner,
            winning_confidence=winning_confidence,
            strategy_used=VotingStrategy.APPROVAL,
            vote_distribution=vote_distribution,
            consensus_level=0.0,
            tie_occurred=tie_occurred,
            tie_break_method=tie_break_method,
            individual_votes=votes,
            quality_score=0.0
        )

    def _condorcet_vote(self, votes: List[AIVote]) -> VotingResult:
        """Condorcet method (pairwise comparison)"""
        # Get all unique recommendations
        all_recs = set(v.recommendation for v in votes)
        for vote in votes:
            all_recs.update(r for r, _ in vote.alternative_recommendations)

        # Perform pairwise comparisons
        pairwise_wins = defaultdict(int)

        for rec1 in all_recs:
            for rec2 in all_recs:
                if rec1 == rec2:
                    continue

                # Count votes where rec1 preferred over rec2
                rec1_preferred = 0
                for vote in votes:
                    # Get rankings
                    rec1_rank = self._get_recommendation_rank(vote, rec1)
                    rec2_rank = self._get_recommendation_rank(vote, rec2)

                    if rec1_rank < rec2_rank:  # Lower rank = higher preference
                        rec1_preferred += 1

                if rec1_preferred > len(votes) / 2:
                    pairwise_wins[rec1] += 1

        # Find Condorcet winner (beats all others)
        max_wins = max(pairwise_wins.values()) if pairwise_wins else 0
        condorcet_winners = [
            rec for rec, wins in pairwise_wins.items()
            if wins == max_wins
        ]

        tie_occurred = len(condorcet_winners) > 1
        tie_break_method = None

        if condorcet_winners:
            winner = condorcet_winners[0]
            if tie_occurred:
                winner, tie_break_method = self._break_tie_by_confidence(
                    votes, condorcet_winners
                )
        else:
            # No Condorcet winner - fall back to weighted vote
            return self._weighted_vote(votes)

        # Winning confidence
        winner_votes = [v for v in votes if v.recommendation == winner]
        winning_confidence = sum(v.confidence for v in winner_votes) / len(winner_votes)

        # Vote distribution
        total_wins = sum(pairwise_wins.values())
        vote_distribution = {
            option: wins / total_wins if total_wins > 0 else 0.0
            for option, wins in pairwise_wins.items()
        }

        return VotingResult(
            winner=winner,
            winning_confidence=winning_confidence,
            strategy_used=VotingStrategy.CONDORCET,
            vote_distribution=vote_distribution,
            consensus_level=0.0,
            tie_occurred=tie_occurred,
            tie_break_method=tie_break_method,
            individual_votes=votes,
            quality_score=0.0
        )

    def _select_adaptive_strategy(self, votes: List[AIVote]) -> VotingStrategy:
        """Adaptively select best voting strategy based on vote characteristics"""
        # Calculate vote characteristics
        confidences = [v.confidence for v in votes]
        avg_confidence = statistics.mean(confidences)
        confidence_variance = statistics.variance(confidences) if len(confidences) > 1 else 0

        # Count unique recommendations
        recommendations = [v.recommendation for v in votes]
        unique_recommendations = len(set(recommendations))

        # Decision logic
        if confidence_variance < 0.05 and avg_confidence > 0.7:
            # High consensus, high confidence → simple majority
            return VotingStrategy.MAJORITY

        elif confidence_variance > 0.15:
            # High variance in confidence → weighted voting
            return VotingStrategy.WEIGHTED

        elif unique_recommendations >= self.strategy_thresholds['many_options']:
            # Many alternatives → Borda count
            return VotingStrategy.BORDA_COUNT

        elif avg_confidence < self.strategy_thresholds['low_confidence']:
            # Low confidence overall → approval voting
            return VotingStrategy.APPROVAL

        else:
            # Default to weighted
            return VotingStrategy.WEIGHTED

    def _break_tie_by_confidence(
        self, votes: List[AIVote], tied_options: List[str]
    ) -> Tuple[str, str]:
        """Break tie using average confidence"""
        avg_confidences = {}

        for option in tied_options:
            option_votes = [v for v in votes if v.recommendation == option]
            if option_votes:
                avg_confidences[option] = sum(v.confidence for v in option_votes) / len(option_votes)
            else:
                avg_confidences[option] = 0.0

        winner = max(avg_confidences.items(), key=lambda x: x[1])[0]
        return winner, "average_confidence"

    def _break_tie_by_vote_count(
        self, votes: List[AIVote], tied_options: List[str]
    ) -> Tuple[str, str]:
        """Break tie using vote count"""
        vote_counts = {
            option: len([v for v in votes if v.recommendation == option])
            for option in tied_options
        }

        winner = max(vote_counts.items(), key=lambda x: x[1])[0]
        return winner, "vote_count"

    def _get_recommendation_rank(self, vote: AIVote, recommendation: str) -> int:
        """Get rank of recommendation in vote (0 = highest)"""
        if vote.recommendation == recommendation:
            return 0

        for idx, (rec, _) in enumerate(vote.alternative_recommendations):
            if rec == recommendation:
                return idx + 1

        return 999  # Not in rankings

    def _calculate_consensus_level(self, votes: List[AIVote], winner: str) -> float:
        """Calculate consensus level (0-1)"""
        winner_votes = len([v for v in votes if v.recommendation == winner])
        total_votes = len(votes)

        # Raw agreement rate
        agreement_rate = winner_votes / total_votes

        # Weight by confidence variance
        confidences = [v.confidence for v in votes]
        confidence_variance = statistics.variance(confidences) if len(confidences) > 1 else 0

        # Lower variance = higher consensus
        variance_factor = 1.0 - min(confidence_variance / 0.25, 1.0)

        # Combined consensus
        consensus = 0.7 * agreement_rate + 0.3 * variance_factor

        return consensus

    def _assess_vote_quality(self, votes: List[AIVote], result: VotingResult) -> float:
        """Assess quality of voting process (0-1)"""
        quality_factors = []

        # 1. Number of votes (more is better)
        vote_count_quality = min(len(votes) / 5, 1.0)  # Optimal at 5+ votes
        quality_factors.append(vote_count_quality * 0.2)

        # 2. Average confidence
        avg_confidence = statistics.mean([v.confidence for v in votes])
        quality_factors.append(avg_confidence * 0.3)

        # 3. Consensus level
        quality_factors.append(result.consensus_level * 0.25)

        # 4. No tie (ties reduce quality)
        no_tie_quality = 0.0 if result.tie_occurred else 1.0
        quality_factors.append(no_tie_quality * 0.15)

        # 5. Clear winner (margin of victory)
        if result.vote_distribution:
            sorted_votes = sorted(result.vote_distribution.values(), reverse=True)
            if len(sorted_votes) > 1:
                margin = sorted_votes[0] - sorted_votes[1]
                margin_quality = min(margin * 2, 1.0)  # Higher margin = better
            else:
                margin_quality = 1.0
            quality_factors.append(margin_quality * 0.1)

        return sum(quality_factors)

    def _generate_warnings(
        self, votes: List[AIVote], result: VotingResult,
        consensus: float, quality: float
    ) -> List[str]:
        """Generate warnings about voting issues"""
        warnings = []

        if len(votes) < 3:
            warnings.append("Low number of votes - consider more models")

        if consensus < 0.4:
            warnings.append("Low consensus - models significantly disagree")

        if quality < 0.5:
            warnings.append("Low vote quality - results may be unreliable")

        if result.tie_occurred:
            warnings.append(f"Tie occurred - broken by {result.tie_break_method}")

        avg_confidence = statistics.mean([v.confidence for v in votes])
        if avg_confidence < 0.5:
            warnings.append("Low average confidence in recommendations")

        return warnings


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("🗳️ AI VOTING SYSTEM - DEMONSTRATION")
    print("="*80)

    # Create voting system
    voting_system = AIVotingSystem()

    # Create sample votes
    votes = [
        AIVote(
            model_name="GPT-4",
            recommendation="deep_web_search",
            confidence=0.85,
            reasoning="Strong technical indicators suggest deep web search",
            alternative_recommendations=[("social_media", 0.70), ("government_records", 0.60)]
        ),
        AIVote(
            model_name="Gemini",
            recommendation="social_media",
            confidence=0.80,
            reasoning="High probability of social media presence",
            alternative_recommendations=[("deep_web_search", 0.75), ("public_records", 0.65)]
        ),
        AIVote(
            model_name="Claude",
            recommendation="deep_web_search",
            confidence=0.90,
            reasoning="Strategic analysis points to deep web investigation",
            alternative_recommendations=[("government_records", 0.72), ("social_media", 0.68)]
        )
    ]

    # Test different voting strategies
    strategies = [
        VotingStrategy.MAJORITY,
        VotingStrategy.WEIGHTED,
        VotingStrategy.BORDA_COUNT,
        VotingStrategy.ADAPTIVE
    ]

    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"📊 Strategy: {strategy.value.upper()}")
        print("="*80)

        result = voting_system.conduct_vote(votes, strategy=strategy)

        print(f"\n🏆 Winner: {result.winner}")
        print(f"💯 Confidence: {result.winning_confidence:.3f}")
        print(f"🤝 Consensus: {result.consensus_level:.3f}")
        print(f"⭐ Quality: {result.quality_score:.3f}")
        print(f"🎲 Tie Occurred: {result.tie_occurred}")

        print(f"\n📊 Vote Distribution:")
        for option, score in sorted(result.vote_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {option}: {score:.3f}")

        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

    print("\n" + "="*80)
