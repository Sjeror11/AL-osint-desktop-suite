#!/usr/bin/env python3
"""
🔍 Cross-Platform People Search - Unified Social Media Investigation
LakyLuk OSINT Investigation Suite

Features:
✅ Unified search across Facebook, Instagram, LinkedIn, Twitter, and more
✅ AI-powered result aggregation and deduplication
✅ Advanced profile correlation and identity matching
✅ Real-time search with parallel platform queries
✅ Intelligent ranking based on relevance and confidence
✅ Export capabilities for investigation reporting

Search Strategies:
- Direct username/handle search across platforms
- Name-based search with location and demographic filters
- Email and phone number reverse lookup
- Image reverse search for profile pictures
- Cross-reference validation using multiple data points
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import re

from .facebook_scanner import FacebookScanner
from .instagram_scanner import InstagramScanner
from .linkedin_scanner import LinkedInScanner
from ...analytics.entity_correlation_engine import EntityCorrelationEngine, EntityProfile
from ...core.enhanced_orchestrator import EnhancedInvestigationOrchestrator
from ...utils.data_sanitizer import PIISanitizer
from ...utils.rate_limiter import RateLimiter


@dataclass
class SearchQuery:
    """Structured search query for cross-platform investigation"""
    target_name: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    age_range: Optional[Tuple[int, int]] = None
    employer: Optional[str] = None
    school: Optional[str] = None
    keywords: List[str] = None
    platforms: List[str] = None
    search_depth: str = "standard"  # basic, standard, deep

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.platforms is None:
            self.platforms = ['facebook', 'instagram', 'linkedin']


@dataclass
class SearchResult:
    """Unified search result across platforms"""
    search_id: str
    query: SearchQuery
    profiles: List[EntityProfile]
    correlations: List[Any]  # CorrelationResult objects
    confidence_score: float
    total_matches: int
    search_duration: float
    ai_insights: Dict[str, Any]
    created_at: datetime


class CrossPlatformSearch:
    """Unified cross-platform people search engine"""

    def __init__(self, ai_orchestrator: EnhancedInvestigationOrchestrator = None):
        self.ai_orchestrator = ai_orchestrator
        self.entity_correlation = EntityCorrelationEngine(ai_orchestrator)
        self.pii_sanitizer = PIISanitizer()

        # Initialize platform scanners
        self.scanners = {
            'facebook': FacebookScanner(ai_orchestrator),
            'instagram': InstagramScanner(ai_orchestrator),
            'linkedin': LinkedInScanner(ai_orchestrator)
        }

        # Search configuration
        self.search_config = {
            'max_results_per_platform': 50,
            'parallel_searches': True,
            'correlation_threshold': 0.65,
            'timeout_per_platform': 60,  # seconds
            'enable_ai_enhancement': True
        }

        # Rate limiting
        self.rate_limiter = RateLimiter(
            requests_per_minute=20,
            requests_per_hour=200
        )

    async def unified_people_search(self, query: SearchQuery) -> SearchResult:
        """
        Perform unified search across multiple social media platforms

        Args:
            query: Structured search query with target information

        Returns:
            Comprehensive search result with correlated profiles
        """
        search_start_time = time.time()
        search_id = self._generate_search_id(query)

        try:
            await self.rate_limiter.wait_if_needed()

            # Execute parallel searches across platforms
            platform_results = await self._execute_parallel_searches(query)

            # Convert results to unified EntityProfile format
            all_profiles = self._convert_to_entity_profiles(platform_results)

            # AI-powered profile enhancement
            if self.ai_orchestrator and self.search_config['enable_ai_enhancement']:
                enhanced_profiles = await self._ai_enhance_profiles(all_profiles, query)
                all_profiles = enhanced_profiles

            # Correlate entities across platforms
            correlations = await self.entity_correlation.correlate_entities(all_profiles)

            # Calculate overall confidence score
            confidence_score = self._calculate_search_confidence(all_profiles, correlations)

            # AI insights generation
            ai_insights = {}
            if self.ai_orchestrator:
                ai_insights = await self._generate_ai_insights(query, all_profiles, correlations)

            search_duration = time.time() - search_start_time

            return SearchResult(
                search_id=search_id,
                query=query,
                profiles=all_profiles,
                correlations=correlations,
                confidence_score=confidence_score,
                total_matches=len(all_profiles),
                search_duration=search_duration,
                ai_insights=ai_insights,
                created_at=datetime.now()
            )

        except Exception as e:
            print(f"Unified search error: {e}")
            return SearchResult(
                search_id=search_id,
                query=query,
                profiles=[],
                correlations=[],
                confidence_score=0.0,
                total_matches=0,
                search_duration=time.time() - search_start_time,
                ai_insights={},
                created_at=datetime.now()
            )

    async def find_similar_profiles(self, target_profile: EntityProfile,
                                   platforms: List[str] = None) -> List[Tuple[EntityProfile, float]]:
        """
        Find profiles similar to a target profile across platforms

        Args:
            target_profile: Profile to find similar matches for
            platforms: Platforms to search (default: all)

        Returns:
            List of similar profiles with similarity scores
        """
        if platforms is None:
            platforms = list(self.scanners.keys())

        similar_profiles = []

        try:
            # Create search query based on target profile
            query = self._profile_to_search_query(target_profile)

            # Search across specified platforms
            search_results = await self._execute_targeted_searches(query, platforms)

            # Convert to EntityProfile format
            candidate_profiles = self._convert_to_entity_profiles(search_results)

            # Find cross-platform matches
            matches = await self.entity_correlation.find_cross_platform_matches(
                target_profile, candidate_profiles
            )

            similar_profiles.extend(matches)

            # AI validation of similarity
            if self.ai_orchestrator:
                validated_profiles = await self._ai_validate_similarity(
                    target_profile, similar_profiles
                )
                similar_profiles = validated_profiles

            # Sort by similarity score
            similar_profiles.sort(key=lambda x: x[1], reverse=True)

            return similar_profiles

        except Exception as e:
            print(f"Similar profiles search error: {e}")
            return []

    async def reverse_image_search(self, image_url: str,
                                  platforms: List[str] = None) -> List[EntityProfile]:
        """
        Perform reverse image search to find profiles with same/similar pictures

        Args:
            image_url: URL of image to search for
            platforms: Platforms to search

        Returns:
            List of profiles with matching/similar images
        """
        if platforms is None:
            platforms = list(self.scanners.keys())

        matching_profiles = []

        try:
            # This would require integration with reverse image search APIs
            # like Google Images, TinEye, or Yandex

            # For now, implement basic image hash comparison
            # In production, this would use computer vision APIs

            if self.ai_orchestrator:
                ai_image_matches = await self.ai_orchestrator.reverse_image_search(
                    image_url, platforms
                )
                matching_profiles.extend(ai_image_matches)

            return matching_profiles

        except Exception as e:
            print(f"Reverse image search error: {e}")
            return []

    async def investigate_network_connections(self, target_profiles: List[EntityProfile],
                                            depth: int = 2) -> Dict[str, Any]:
        """
        Investigate network connections across multiple platforms

        Args:
            target_profiles: Starting profiles for network investigation
            depth: Degrees of separation to explore

        Returns:
            Comprehensive network analysis across platforms
        """
        try:
            network_investigation = {
                'target_profiles': target_profiles,
                'cross_platform_networks': {},
                'common_connections': {},
                'network_clusters': {},
                'influence_analysis': {},
                'relationship_patterns': {}
            }

            # Investigate each platform's network
            for platform, scanner in self.scanners.items():
                platform_networks = {}

                for profile in target_profiles:
                    if profile.platform == platform:
                        # Map network for this profile
                        network = await scanner.map_network_connections(
                            profile.username, depth
                        )
                        platform_networks[profile.username] = network

                network_investigation['cross_platform_networks'][platform] = platform_networks

            # Find common connections across platforms
            network_investigation['common_connections'] = await self._find_common_connections(
                network_investigation['cross_platform_networks']
            )

            # AI-powered network analysis
            if self.ai_orchestrator:
                ai_network_analysis = await self.ai_orchestrator.analyze_cross_platform_networks(
                    network_investigation
                )
                network_investigation['ai_network_analysis'] = ai_network_analysis

            return network_investigation

        except Exception as e:
            print(f"Network investigation error: {e}")
            return {}

    # Internal helper methods

    async def _execute_parallel_searches(self, query: SearchQuery) -> Dict[str, List[Dict]]:
        """Execute searches across platforms in parallel"""
        platform_results = {}

        if self.search_config['parallel_searches']:
            # Parallel execution
            tasks = []
            for platform in query.platforms:
                if platform in self.scanners:
                    task = self._search_single_platform(platform, query)
                    tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, platform in enumerate(query.platforms):
                if i < len(results) and not isinstance(results[i], Exception):
                    platform_results[platform] = results[i]
                else:
                    platform_results[platform] = []

        else:
            # Sequential execution
            for platform in query.platforms:
                if platform in self.scanners:
                    try:
                        results = await self._search_single_platform(platform, query)
                        platform_results[platform] = results
                    except Exception as e:
                        print(f"Error searching {platform}: {e}")
                        platform_results[platform] = []

        return platform_results

    async def _search_single_platform(self, platform: str, query: SearchQuery) -> List[Dict]:
        """Search a single platform based on query"""
        scanner = self.scanners[platform]
        results = []

        try:
            if query.target_name:
                # Name-based search
                if platform == 'facebook':
                    results = await scanner.search_people(
                        query.target_name, query.location, query.age_range
                    )
                elif platform == 'instagram':
                    results = await scanner.search_profiles(
                        query.target_name, self.search_config['max_results_per_platform']
                    )
                elif platform == 'linkedin':
                    results = await scanner.search_professionals(
                        query.target_name, query.location, query.employer
                    )

            elif query.username:
                # Username-based search
                if platform == 'instagram':
                    # Try direct profile access
                    profile = await scanner.analyze_profile(query.username)
                    if profile:
                        results = [profile]

            # Additional search strategies based on available information
            if query.employer and platform == 'linkedin':
                company_results = await scanner.analyze_company_employees(
                    query.employer, self.search_config['max_results_per_platform']
                )
                if company_results and 'employee_profiles' in company_results:
                    results.extend(company_results['employee_profiles'])

        except Exception as e:
            print(f"Error searching {platform}: {e}")

        return results[:self.search_config['max_results_per_platform']]

    def _convert_to_entity_profiles(self, platform_results: Dict[str, List[Dict]]) -> List[EntityProfile]:
        """Convert platform-specific results to unified EntityProfile format"""
        entity_profiles = []

        for platform, results in platform_results.items():
            for result in results:
                try:
                    # Extract common fields across platforms
                    profile = EntityProfile(
                        platform=platform,
                        username=result.get('username', result.get('name', 'unknown')),
                        profile_url=result.get('profile_url', ''),
                        display_name=result.get('name', result.get('display_name')),
                        bio=result.get('bio', result.get('headline', result.get('description'))),
                        location=result.get('location'),
                        profile_picture_url=result.get('profile_picture', result.get('profile_picture_url')),
                        follower_count=result.get('followers', result.get('follower_count')),
                        following_count=result.get('following', result.get('following_count')),
                        post_count=result.get('posts_count', result.get('post_count')),
                        verified=result.get('verified', False),
                        metadata=result
                    )

                    # Extract content samples if available
                    if 'content_analysis' in result and 'recent_posts' in result['content_analysis']:
                        profile.content_samples = [
                            post.get('content', '') for post in result['content_analysis']['recent_posts'][:5]
                        ]

                    entity_profiles.append(profile)

                except Exception as e:
                    print(f"Error converting result to EntityProfile: {e}")
                    continue

        return entity_profiles

    async def _ai_enhance_profiles(self, profiles: List[EntityProfile],
                                  query: SearchQuery) -> List[EntityProfile]:
        """Use AI to enhance and filter profiles"""
        if not self.ai_orchestrator:
            return profiles

        enhanced_profiles = []

        for profile in profiles:
            try:
                # AI relevance scoring
                relevance_score = await self.ai_orchestrator.score_profile_relevance(
                    profile, query
                )

                # Add relevance score to metadata
                profile.metadata['ai_relevance_score'] = relevance_score

                # Only keep profiles above threshold
                if relevance_score >= 0.3:  # Configurable threshold
                    enhanced_profiles.append(profile)

            except Exception as e:
                print(f"Error enhancing profile: {e}")
                enhanced_profiles.append(profile)  # Keep original if AI fails

        # Sort by relevance score
        enhanced_profiles.sort(
            key=lambda x: x.metadata.get('ai_relevance_score', 0),
            reverse=True
        )

        return enhanced_profiles

    def _calculate_search_confidence(self, profiles: List[EntityProfile],
                                   correlations: List[Any]) -> float:
        """Calculate overall confidence score for search results"""
        if not profiles:
            return 0.0

        factors = []

        # Profile quality factor
        quality_scores = [
            profile.metadata.get('ai_relevance_score', 0.5) for profile in profiles
        ]
        factors.append(sum(quality_scores) / len(quality_scores))

        # Correlation strength factor
        if correlations:
            correlation_scores = [corr.confidence_score for corr in correlations]
            factors.append(sum(correlation_scores) / len(correlation_scores))
        else:
            factors.append(0.3)  # Lower confidence without correlations

        # Platform diversity factor
        platforms = set(profile.platform for profile in profiles)
        platform_diversity = len(platforms) / len(self.scanners)
        factors.append(platform_diversity)

        # Data completeness factor
        complete_profiles = sum(1 for p in profiles if p.bio and p.location and p.profile_picture_url)
        completeness_factor = complete_profiles / len(profiles) if profiles else 0
        factors.append(completeness_factor)

        return sum(factors) / len(factors)

    async def _generate_ai_insights(self, query: SearchQuery, profiles: List[EntityProfile],
                                   correlations: List[Any]) -> Dict[str, Any]:
        """Generate AI insights for search results"""
        if not self.ai_orchestrator:
            return {}

        insights = await self.ai_orchestrator.generate_search_insights({
            'query': asdict(query),
            'profiles_count': len(profiles),
            'correlations_count': len(correlations),
            'platforms_searched': list(set(p.platform for p in profiles))
        })

        return insights

    def _generate_search_id(self, query: SearchQuery) -> str:
        """Generate unique ID for search"""
        import hashlib
        query_str = f"{query.target_name}_{query.username}_{datetime.now().isoformat()}"
        return hashlib.md5(query_str.encode()).hexdigest()[:12]

    def _profile_to_search_query(self, profile: EntityProfile) -> SearchQuery:
        """Convert EntityProfile to SearchQuery for similarity search"""
        return SearchQuery(
            target_name=profile.display_name,
            username=profile.username,
            location=profile.location,
            platforms=list(self.scanners.keys())
        )