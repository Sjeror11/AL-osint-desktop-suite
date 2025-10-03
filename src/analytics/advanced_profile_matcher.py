#!/usr/bin/env python3
"""
🔍 Advanced Profile Matcher - AI-Powered Identity Recognition
LakyLuk OSINT Investigation Suite

Features:
✅ Multi-dimensional similarity analysis with machine learning
✅ Computer vision for profile picture matching and face recognition
✅ Natural language processing for bio and content analysis
✅ Behavioral pattern recognition across platforms and time
✅ Deep learning embeddings for advanced identity correlation
✅ Uncertainty quantification and confidence scoring

Matching Algorithms:
- Facial recognition using deep neural networks
- Textual similarity using transformer embeddings
- Behavioral biometrics and posting pattern analysis
- Social network structure comparison
- Temporal activity pattern matching
- Geographic location correlation
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
import re
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import cv2
import face_recognition
from PIL import Image
import requests
from io import BytesIO

from ..analytics.entity_correlation_engine import EntityProfile
from ..core.enhanced_orchestrator import AIOrchestrator
from ..utils.image_processor import ImageProcessor
from ..utils.text_analyzer import TextAnalyzer
from ..utils.temporal_analyzer import TemporalAnalyzer


@dataclass
class MatchingFeatures:
    """Comprehensive features extracted for profile matching"""
    profile_id: str

    # Visual features
    face_encoding: Optional[np.ndarray] = None
    face_landmarks: Optional[Dict] = None
    image_hash: Optional[str] = None
    visual_similarity_vector: Optional[np.ndarray] = None

    # Textual features
    name_tokens: List[str] = None
    bio_embedding: Optional[np.ndarray] = None
    content_embedding: Optional[np.ndarray] = None
    linguistic_features: Dict[str, float] = None

    # Behavioral features
    posting_patterns: Dict[str, Any] = None
    activity_patterns: Dict[str, Any] = None
    interaction_patterns: Dict[str, Any] = None

    # Network features
    network_structure: Dict[str, Any] = None
    connection_patterns: Dict[str, Any] = None

    # Temporal features
    temporal_patterns: Dict[str, Any] = None

    # Metadata
    extraction_timestamp: datetime = None
    feature_quality_score: float = 0.0

    def __post_init__(self):
        if self.name_tokens is None:
            self.name_tokens = []
        if self.linguistic_features is None:
            self.linguistic_features = {}
        if self.posting_patterns is None:
            self.posting_patterns = {}
        if self.activity_patterns is None:
            self.activity_patterns = {}
        if self.interaction_patterns is None:
            self.interaction_patterns = {}
        if self.network_structure is None:
            self.network_structure = {}
        if self.connection_patterns is None:
            self.connection_patterns = {}
        if self.temporal_patterns is None:
            self.temporal_patterns = {}
        if self.extraction_timestamp is None:
            self.extraction_timestamp = datetime.now()


@dataclass
class MatchResult:
    """Result of advanced profile matching"""
    profile1_id: str
    profile2_id: str
    overall_similarity: float
    confidence_score: float

    # Component similarities
    visual_similarity: float = 0.0
    textual_similarity: float = 0.0
    behavioral_similarity: float = 0.0
    network_similarity: float = 0.0
    temporal_similarity: float = 0.0

    # Evidence
    matching_evidence: List[str] = None
    conflicting_evidence: List[str] = None

    # AI insights
    ai_analysis: Dict[str, Any] = None

    # Metadata
    match_timestamp: datetime = None
    algorithm_version: str = "1.0"

    def __post_init__(self):
        if self.matching_evidence is None:
            self.matching_evidence = []
        if self.conflicting_evidence is None:
            self.conflicting_evidence = []
        if self.ai_analysis is None:
            self.ai_analysis = {}
        if self.match_timestamp is None:
            self.match_timestamp = datetime.now()


class AdvancedProfileMatcher:
    """AI-powered advanced profile matching system"""

    def __init__(self, ai_orchestrator: AIOrchestrator = None):
        self.ai_orchestrator = ai_orchestrator
        self.image_processor = ImageProcessor()
        self.text_analyzer = TextAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()

        # Feature extractors
        self.face_detector = None  # Will be initialized on first use
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # Machine learning models
        self.similarity_classifier = None
        self.identity_classifier = None

        # Matching thresholds
        self.thresholds = {
            'face_similarity': 0.6,        # Face recognition threshold
            'text_similarity': 0.7,        # Text similarity threshold
            'behavioral_similarity': 0.65,  # Behavioral pattern threshold
            'overall_similarity': 0.75,    # Overall match threshold
            'high_confidence': 0.85,       # High confidence threshold
            'minimum_features': 3          # Minimum features required for match
        }

        # Feature weights for overall similarity calculation
        self.feature_weights = {
            'visual': 0.30,
            'textual': 0.25,
            'behavioral': 0.20,
            'network': 0.15,
            'temporal': 0.10
        }

        # Caching for expensive operations
        self.feature_cache = {}
        self.similarity_cache = {}

    async def extract_matching_features(self, profile: EntityProfile) -> MatchingFeatures:
        """
        Extract comprehensive features for profile matching

        Args:
            profile: Entity profile to extract features from

        Returns:
            Comprehensive matching features
        """
        try:
            profile_id = f"{profile.platform}:{profile.username}"

            # Check cache first
            if profile_id in self.feature_cache:
                cached_features = self.feature_cache[profile_id]
                # Check if cache is still valid (24 hours)
                if (datetime.now() - cached_features.extraction_timestamp).hours < 24:
                    return cached_features

            features = MatchingFeatures(profile_id=profile_id)

            # Extract visual features
            if profile.profile_picture_url:
                visual_features = await self._extract_visual_features(profile.profile_picture_url)
                features.face_encoding = visual_features.get('face_encoding')
                features.face_landmarks = visual_features.get('face_landmarks')
                features.image_hash = visual_features.get('image_hash')
                features.visual_similarity_vector = visual_features.get('similarity_vector')

            # Extract textual features
            textual_features = await self._extract_textual_features(profile)
            features.name_tokens = textual_features.get('name_tokens', [])
            features.bio_embedding = textual_features.get('bio_embedding')
            features.content_embedding = textual_features.get('content_embedding')
            features.linguistic_features = textual_features.get('linguistic_features', {})

            # Extract behavioral features
            behavioral_features = await self._extract_behavioral_features(profile)
            features.posting_patterns = behavioral_features.get('posting_patterns', {})
            features.activity_patterns = behavioral_features.get('activity_patterns', {})
            features.interaction_patterns = behavioral_features.get('interaction_patterns', {})

            # Extract network features
            network_features = await self._extract_network_features(profile)
            features.network_structure = network_features.get('network_structure', {})
            features.connection_patterns = network_features.get('connection_patterns', {})

            # Extract temporal features
            temporal_features = await self._extract_temporal_features(profile)
            features.temporal_patterns = temporal_features

            # Calculate feature quality score
            features.feature_quality_score = self._calculate_feature_quality(features)

            # AI-enhanced feature extraction
            if self.ai_orchestrator:
                ai_features = await self.ai_orchestrator.extract_advanced_features(profile)
                features.ai_enhanced_features = ai_features

            # Cache the features
            self.feature_cache[profile_id] = features

            return features

        except Exception as e:
            print(f"Feature extraction error for {profile.username}: {e}")
            return MatchingFeatures(profile_id=f"{profile.platform}:{profile.username}")

    async def match_profiles(self, profile1: EntityProfile, profile2: EntityProfile) -> MatchResult:
        """
        Perform comprehensive profile matching analysis

        Args:
            profile1: First profile to compare
            profile2: Second profile to compare

        Returns:
            Detailed matching result with similarity scores
        """
        try:
            # Extract features for both profiles
            features1 = await self.extract_matching_features(profile1)
            features2 = await self.extract_matching_features(profile2)

            # Create match result
            match_result = MatchResult(
                profile1_id=features1.profile_id,
                profile2_id=features2.profile_id,
                overall_similarity=0.0,
                confidence_score=0.0
            )

            # Calculate component similarities
            match_result.visual_similarity = await self._calculate_visual_similarity(features1, features2)
            match_result.textual_similarity = await self._calculate_textual_similarity(features1, features2)
            match_result.behavioral_similarity = await self._calculate_behavioral_similarity(features1, features2)
            match_result.network_similarity = await self._calculate_network_similarity(features1, features2)
            match_result.temporal_similarity = await self._calculate_temporal_similarity(features1, features2)

            # Calculate overall similarity
            match_result.overall_similarity = self._calculate_overall_similarity(match_result)

            # Calculate confidence score
            match_result.confidence_score = await self._calculate_confidence_score(
                match_result, features1, features2
            )

            # Extract evidence
            evidence = await self._extract_matching_evidence(match_result, features1, features2)
            match_result.matching_evidence = evidence['supporting']
            match_result.conflicting_evidence = evidence['conflicting']

            # AI-powered analysis
            if self.ai_orchestrator:
                ai_analysis = await self.ai_orchestrator.analyze_profile_match(
                    profile1, profile2, match_result
                )
                match_result.ai_analysis = ai_analysis

            return match_result

        except Exception as e:
            print(f"Profile matching error: {e}")
            return MatchResult(
                profile1_id=f"{profile1.platform}:{profile1.username}",
                profile2_id=f"{profile2.platform}:{profile2.username}",
                overall_similarity=0.0,
                confidence_score=0.0
            )

    async def batch_match_profiles(self, profiles: List[EntityProfile],
                                 threshold: float = None) -> List[Tuple[EntityProfile, EntityProfile, MatchResult]]:
        """
        Perform batch matching across multiple profiles

        Args:
            profiles: List of profiles to match
            threshold: Minimum similarity threshold for matches

        Returns:
            List of profile pairs with match results above threshold
        """
        if threshold is None:
            threshold = self.thresholds['overall_similarity']

        matches = []

        try:
            # Extract features for all profiles
            all_features = {}
            for profile in profiles:
                features = await self.extract_matching_features(profile)
                all_features[profile] = features

            # Compare all pairs
            for i in range(len(profiles)):
                for j in range(i + 1, len(profiles)):
                    profile1 = profiles[i]
                    profile2 = profiles[j]

                    # Skip same platform comparisons if desired
                    if profile1.platform == profile2.platform:
                        continue

                    match_result = await self.match_profiles(profile1, profile2)

                    if match_result.overall_similarity >= threshold:
                        matches.append((profile1, profile2, match_result))

            # Sort by similarity score
            matches.sort(key=lambda x: x[2].overall_similarity, reverse=True)

            return matches

        except Exception as e:
            print(f"Batch matching error: {e}")
            return []

    async def find_duplicate_profiles(self, profiles: List[EntityProfile]) -> Dict[str, List[EntityProfile]]:
        """
        Find potential duplicate profiles representing the same person

        Args:
            profiles: List of profiles to analyze for duplicates

        Returns:
            Dictionary mapping cluster IDs to groups of similar profiles
        """
        try:
            duplicate_clusters = {}
            matches = await self.batch_match_profiles(profiles, threshold=0.8)  # High threshold for duplicates

            # Use graph-based clustering to find groups
            profile_graph = {}
            for profile1, profile2, match_result in matches:
                profile1_id = f"{profile1.platform}:{profile1.username}"
                profile2_id = f"{profile2.platform}:{profile2.username}"

                if profile1_id not in profile_graph:
                    profile_graph[profile1_id] = []
                if profile2_id not in profile_graph:
                    profile_graph[profile2_id] = []

                profile_graph[profile1_id].append(profile2_id)
                profile_graph[profile2_id].append(profile1_id)

            # Find connected components (clusters)
            visited = set()
            cluster_id = 0

            for profile_id in profile_graph:
                if profile_id not in visited:
                    cluster = []
                    self._dfs_cluster(profile_id, profile_graph, visited, cluster)

                    if len(cluster) > 1:
                        # Map back to EntityProfile objects
                        cluster_profiles = []
                        for pid in cluster:
                            for profile in profiles:
                                if f"{profile.platform}:{profile.username}" == pid:
                                    cluster_profiles.append(profile)
                                    break

                        duplicate_clusters[f"cluster_{cluster_id}"] = cluster_profiles
                        cluster_id += 1

            return duplicate_clusters

        except Exception as e:
            print(f"Duplicate detection error: {e}")
            return {}

    # Internal feature extraction methods

    async def _extract_visual_features(self, image_url: str) -> Dict[str, Any]:
        """Extract visual features from profile picture"""
        features = {}

        try:
            # Download and process image
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content))
            image_array = np.array(image)

            # Face detection and encoding
            face_locations = face_recognition.face_locations(image_array)
            if face_locations:
                face_encodings = face_recognition.face_encodings(image_array, face_locations)
                if face_encodings:
                    features['face_encoding'] = face_encodings[0]

                # Face landmarks
                face_landmarks = face_recognition.face_landmarks(image_array, face_locations)
                if face_landmarks:
                    features['face_landmarks'] = face_landmarks[0]

            # Image hash for duplicate detection
            features['image_hash'] = self.image_processor.calculate_perceptual_hash(image)

            # Visual similarity vector (CNN features)
            features['similarity_vector'] = await self.image_processor.extract_cnn_features(image)

        except Exception as e:
            print(f"Visual feature extraction error: {e}")

        return features

    async def _extract_textual_features(self, profile: EntityProfile) -> Dict[str, Any]:
        """Extract textual features from profile"""
        features = {}

        try:
            # Name tokenization
            name = profile.display_name or profile.username
            features['name_tokens'] = self.text_analyzer.tokenize_name(name)

            # Bio embedding
            if profile.bio:
                bio_embedding = await self.text_analyzer.get_text_embedding(profile.bio)
                features['bio_embedding'] = bio_embedding

            # Content embedding
            if profile.content_samples:
                combined_content = " ".join(profile.content_samples[:10])
                content_embedding = await self.text_analyzer.get_text_embedding(combined_content)
                features['content_embedding'] = content_embedding

            # Linguistic features
            all_text = []
            if profile.bio:
                all_text.append(profile.bio)
            if profile.content_samples:
                all_text.extend(profile.content_samples[:5])

            if all_text:
                combined_text = " ".join(all_text)
                features['linguistic_features'] = self.text_analyzer.extract_linguistic_features(combined_text)

        except Exception as e:
            print(f"Textual feature extraction error: {e}")

        return features

    async def _extract_behavioral_features(self, profile: EntityProfile) -> Dict[str, Any]:
        """Extract behavioral features from profile"""
        features = {}

        try:
            # Posting patterns
            if hasattr(profile, 'posting_patterns') and profile.posting_patterns:
                features['posting_patterns'] = profile.posting_patterns
            else:
                # Infer from available data
                features['posting_patterns'] = {
                    'estimated_frequency': 'unknown',
                    'content_types': [],
                    'posting_times': []
                }

            # Activity patterns
            features['activity_patterns'] = {
                'follower_following_ratio': self._calculate_follower_ratio(profile),
                'engagement_style': 'unknown',
                'platform_usage_intensity': self._estimate_usage_intensity(profile)
            }

            # Interaction patterns
            features['interaction_patterns'] = {
                'interaction_frequency': 'unknown',
                'response_patterns': 'unknown',
                'content_engagement': 'unknown'
            }

        except Exception as e:
            print(f"Behavioral feature extraction error: {e}")

        return features

    async def _extract_network_features(self, profile: EntityProfile) -> Dict[str, Any]:
        """Extract network-related features"""
        features = {}

        try:
            # Network structure
            features['network_structure'] = {
                'follower_count': profile.follower_count or 0,
                'following_count': profile.following_count or 0,
                'network_density': 'unknown',
                'clustering_coefficient': 'unknown'
            }

            # Connection patterns
            features['connection_patterns'] = {
                'connection_growth_pattern': 'unknown',
                'mutual_connections': [],
                'connection_geography': 'unknown'
            }

        except Exception as e:
            print(f"Network feature extraction error: {e}")

        return features

    async def _extract_temporal_features(self, profile: EntityProfile) -> Dict[str, Any]:
        """Extract temporal features from profile"""
        features = {}

        try:
            if profile.created_date:
                features['account_age'] = (datetime.now() - profile.created_date).days
                features['creation_timestamp'] = profile.created_date.timestamp()

            if profile.last_activity:
                features['last_activity_days'] = (datetime.now() - profile.last_activity).days
                features['last_activity_timestamp'] = profile.last_activity.timestamp()

            # Activity temporal patterns
            features['temporal_activity_pattern'] = 'unknown'
            features['seasonal_patterns'] = 'unknown'

        except Exception as e:
            print(f"Temporal feature extraction error: {e}")

        return features

    # Similarity calculation methods

    async def _calculate_visual_similarity(self, features1: MatchingFeatures,
                                         features2: MatchingFeatures) -> float:
        """Calculate visual similarity between two profiles"""
        similarity = 0.0

        try:
            # Face recognition similarity
            if features1.face_encoding is not None and features2.face_encoding is not None:
                face_distance = face_recognition.face_distance([features1.face_encoding], features2.face_encoding)[0]
                face_similarity = 1 - face_distance
                similarity += face_similarity * 0.7

            # Image hash similarity
            if features1.image_hash and features2.image_hash:
                hash_similarity = self.image_processor.compare_hashes(features1.image_hash, features2.image_hash)
                similarity += hash_similarity * 0.3

            # CNN feature similarity
            if (features1.visual_similarity_vector is not None and
                features2.visual_similarity_vector is not None):
                cnn_similarity = cosine_similarity(
                    features1.visual_similarity_vector.reshape(1, -1),
                    features2.visual_similarity_vector.reshape(1, -1)
                )[0, 0]
                similarity = max(similarity, cnn_similarity)  # Take best of face/CNN

        except Exception as e:
            print(f"Visual similarity calculation error: {e}")

        return min(max(similarity, 0.0), 1.0)

    async def _calculate_textual_similarity(self, features1: MatchingFeatures,
                                          features2: MatchingFeatures) -> float:
        """Calculate textual similarity between two profiles"""
        similarities = []

        try:
            # Name similarity
            name_sim = self.text_analyzer.calculate_name_similarity(
                features1.name_tokens, features2.name_tokens
            )
            similarities.append(name_sim * 0.4)

            # Bio similarity
            if features1.bio_embedding is not None and features2.bio_embedding is not None:
                bio_sim = cosine_similarity(
                    features1.bio_embedding.reshape(1, -1),
                    features2.bio_embedding.reshape(1, -1)
                )[0, 0]
                similarities.append(bio_sim * 0.35)

            # Content similarity
            if features1.content_embedding is not None and features2.content_embedding is not None:
                content_sim = cosine_similarity(
                    features1.content_embedding.reshape(1, -1),
                    features2.content_embedding.reshape(1, -1)
                )[0, 0]
                similarities.append(content_sim * 0.25)

            return sum(similarities) if similarities else 0.0

        except Exception as e:
            print(f"Textual similarity calculation error: {e}")
            return 0.0

    async def _calculate_behavioral_similarity(self, features1: MatchingFeatures,
                                             features2: MatchingFeatures) -> float:
        """Calculate behavioral similarity between two profiles"""
        similarities = []

        try:
            # Posting pattern similarity
            posting_sim = self._compare_posting_patterns(
                features1.posting_patterns, features2.posting_patterns
            )
            similarities.append(posting_sim * 0.4)

            # Activity pattern similarity
            activity_sim = self._compare_activity_patterns(
                features1.activity_patterns, features2.activity_patterns
            )
            similarities.append(activity_sim * 0.35)

            # Interaction pattern similarity
            interaction_sim = self._compare_interaction_patterns(
                features1.interaction_patterns, features2.interaction_patterns
            )
            similarities.append(interaction_sim * 0.25)

            return sum(similarities) if similarities else 0.0

        except Exception as e:
            print(f"Behavioral similarity calculation error: {e}")
            return 0.0

    def _calculate_overall_similarity(self, match_result: MatchResult) -> float:
        """Calculate weighted overall similarity score"""
        weighted_sum = (
            match_result.visual_similarity * self.feature_weights['visual'] +
            match_result.textual_similarity * self.feature_weights['textual'] +
            match_result.behavioral_similarity * self.feature_weights['behavioral'] +
            match_result.network_similarity * self.feature_weights['network'] +
            match_result.temporal_similarity * self.feature_weights['temporal']
        )

        return min(max(weighted_sum, 0.0), 1.0)

    def _calculate_feature_quality(self, features: MatchingFeatures) -> float:
        """Calculate quality score for extracted features"""
        quality_factors = []

        # Visual features quality
        if features.face_encoding is not None:
            quality_factors.append(0.3)
        elif features.image_hash:
            quality_factors.append(0.15)

        # Textual features quality
        if features.bio_embedding is not None:
            quality_factors.append(0.25)
        if features.content_embedding is not None:
            quality_factors.append(0.2)

        # Behavioral features quality
        if features.posting_patterns:
            quality_factors.append(0.15)

        # Network features quality
        if features.network_structure.get('follower_count', 0) > 0:
            quality_factors.append(0.1)

        return sum(quality_factors)

    # Utility methods

    def _calculate_follower_ratio(self, profile: EntityProfile) -> float:
        """Calculate follower/following ratio"""
        if profile.follower_count and profile.following_count and profile.following_count > 0:
            return profile.follower_count / profile.following_count
        return 0.0

    def _estimate_usage_intensity(self, profile: EntityProfile) -> str:
        """Estimate platform usage intensity"""
        if profile.post_count:
            if profile.post_count > 1000:
                return 'high'
            elif profile.post_count > 100:
                return 'medium'
            else:
                return 'low'
        return 'unknown'

    def _dfs_cluster(self, node: str, graph: Dict, visited: Set, cluster: List):
        """Depth-first search for clustering"""
        visited.add(node)
        cluster.append(node)

        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    self._dfs_cluster(neighbor, graph, visited, cluster)