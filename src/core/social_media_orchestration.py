#!/usr/bin/env python3
"""
🔗 Social Media Orchestration Module - AI-Enhanced Social OSINT
Desktop OSINT Suite - Orchestrator Integration
LakyLuk Enhanced Edition

Integration module between Enhanced Orchestrator and Social Media OSINT tools
Features:
- AI-guided social media investigation workflows
- Multi-platform correlation and analysis
- Real-time progress tracking and optimization
- Cross-platform entity matching
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Import orchestrator components
from .enhanced_orchestrator import InvestigationTarget, InvestigationType, AIDecision

# Import social media scanners
from ..tools.social_media.facebook_scanner import FacebookScanner
from ..tools.social_media.instagram_scanner import InstagramScanner
from ..tools.social_media.linkedin_scanner import LinkedInScanner
from ..tools.social_media.cross_platform_search import CrossPlatformSearch
from ..tools.social_media.realtime_monitor import RealTimeMonitor

# Import analytics engines
from ..analytics.entity_correlation_engine import EntityCorrelationEngine, EntityProfile

# Import heavy dependencies conditionally
try:
    from ..analytics.advanced_profile_matcher import AdvancedProfileMatcher
    ADVANCED_PROFILE_MATCHER_AVAILABLE = True
except ImportError:
    ADVANCED_PROFILE_MATCHER_AVAILABLE = False

try:
    from ..analytics.social_network_visualizer import SocialNetworkVisualizer
    SOCIAL_NETWORK_VISUALIZER_AVAILABLE = True
except ImportError:
    SOCIAL_NETWORK_VISUALIZER_AVAILABLE = False

@dataclass
class SocialMediaInvestigationResult:
    """Results from social media investigation"""
    target_name: str
    platforms_searched: List[str]
    profiles_found: List[Dict[str, Any]]
    correlations: List[Dict[str, Any]]
    network_analysis: Dict[str, Any]
    confidence_score: float
    investigation_duration: float
    raw_data: Dict[str, Any]
    timestamp: datetime

@dataclass
class SocialMediaPhase:
    """Social media investigation phase configuration"""
    phase_name: str
    platforms: List[str]
    search_strategies: List[str]
    ai_enhancement: bool = True
    correlation_analysis: bool = True
    network_mapping: bool = False
    priority: str = "normal"

class SocialMediaOrchestrator:
    """
    AI-Enhanced Social Media Investigation Orchestrator

    Coordinates social media OSINT operations with AI guidance
    """

    def __init__(self, ai_orchestrator=None):
        self.logger = logging.getLogger(__name__)
        self.ai_orchestrator = ai_orchestrator

        # Social media scanners
        self.facebook_scanner = None
        self.instagram_scanner = None
        self.linkedin_scanner = None
        self.cross_platform_search = None
        self.realtime_monitor = None

        # Analytics engines
        self.entity_correlation = None
        self.profile_matcher = None
        self.network_visualizer = None

        # Investigation state
        self.active_investigations: Dict[str, Dict] = {}
        self.investigation_cache: Dict[str, Any] = {}

        # Default investigation phases
        self.default_phases = [
            SocialMediaPhase(
                phase_name="reconnaissance",
                platforms=["facebook", "instagram", "linkedin"],
                search_strategies=["name_search", "email_search", "location_search"],
                ai_enhancement=True,
                correlation_analysis=False
            ),
            SocialMediaPhase(
                phase_name="deep_analysis",
                platforms=["facebook", "linkedin"],
                search_strategies=["profile_analysis", "connection_mapping"],
                ai_enhancement=True,
                correlation_analysis=True
            ),
            SocialMediaPhase(
                phase_name="correlation",
                platforms=["all"],
                search_strategies=["cross_platform_correlation"],
                ai_enhancement=True,
                correlation_analysis=True,
                network_mapping=True
            )
        ]

    async def initialize(self):
        """Initialize social media orchestrator and all components"""
        try:
            self.logger.info("🔗 Initializing Social Media Orchestrator...")

            # Initialize social media scanners
            self.facebook_scanner = FacebookScanner(self.ai_orchestrator)
            await self.facebook_scanner.initialize()

            self.instagram_scanner = InstagramScanner(self.ai_orchestrator)
            await self.instagram_scanner.initialize()

            self.linkedin_scanner = LinkedInScanner(self.ai_orchestrator)
            await self.linkedin_scanner.initialize()

            # Initialize cross-platform tools
            self.cross_platform_search = CrossPlatformSearch(self.ai_orchestrator)
            self.realtime_monitor = RealTimeMonitor(self.ai_orchestrator)

            # Initialize analytics engines
            self.entity_correlation = EntityCorrelationEngine(self.ai_orchestrator)

            # Initialize advanced components if available
            if ADVANCED_PROFILE_MATCHER_AVAILABLE:
                self.profile_matcher = AdvancedProfileMatcher(self.ai_orchestrator)
                self.logger.info("✅ Advanced Profile Matcher initialized")
            else:
                self.profile_matcher = None
                self.logger.info("⚠️ Advanced Profile Matcher not available - missing heavy dependencies")

            if SOCIAL_NETWORK_VISUALIZER_AVAILABLE:
                self.network_visualizer = SocialNetworkVisualizer(self.ai_orchestrator)
                self.logger.info("✅ Social Network Visualizer initialized")
            else:
                self.network_visualizer = None
                self.logger.info("⚠️ Social Network Visualizer not available - missing heavy dependencies")

            self.logger.info("✅ Social Media Orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Social Media Orchestrator: {e}")
            return False

    async def execute_social_media_investigation(
        self,
        target: InvestigationTarget,
        custom_phases: Optional[List[SocialMediaPhase]] = None,
        progress_callback=None
    ) -> SocialMediaInvestigationResult:
        """
        Execute comprehensive social media investigation

        Args:
            target: Investigation target
            custom_phases: Optional custom investigation phases
            progress_callback: Progress reporting function

        Returns:
            Comprehensive social media investigation results
        """
        investigation_start = datetime.now()
        investigation_id = f"social_{target.name.replace(' ', '_')}_{investigation_start.strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"🔍 Starting social media investigation: {investigation_id}")

        # Use custom phases or defaults
        phases = custom_phases or self.default_phases

        # Initialize results structure
        results = {
            'target_name': target.name,
            'platforms_searched': [],
            'profiles_found': [],
            'correlations': [],
            'network_analysis': {},
            'raw_data': {},
            'phases_completed': []
        }

        try:
            # Execute each investigation phase
            for i, phase in enumerate(phases):
                if progress_callback:
                    progress_callback("social_media", f"Phase {i+1}/{len(phases)}: {phase.phase_name}")

                self.logger.info(f"📋 Executing social media phase: {phase.phase_name}")

                phase_results = await self._execute_social_media_phase(target, phase, progress_callback)

                # Merge phase results
                if phase_results:
                    results['platforms_searched'].extend(phase_results.get('platforms', []))
                    results['profiles_found'].extend(phase_results.get('profiles', []))
                    results['correlations'].extend(phase_results.get('correlations', []))
                    results['raw_data'][phase.phase_name] = phase_results
                    results['phases_completed'].append(phase.phase_name)

                # Add delay between phases to avoid rate limiting
                await asyncio.sleep(2)

            # Calculate overall confidence score
            confidence_score = self._calculate_investigation_confidence(results)

            # Investigation duration
            investigation_duration = (datetime.now() - investigation_start).total_seconds()

            # Create final result object
            final_result = SocialMediaInvestigationResult(
                target_name=target.name,
                platforms_searched=list(set(results['platforms_searched'])),
                profiles_found=results['profiles_found'],
                correlations=results['correlations'],
                network_analysis=results['network_analysis'],
                confidence_score=confidence_score,
                investigation_duration=investigation_duration,
                raw_data=results['raw_data'],
                timestamp=datetime.now()
            )

            self.logger.info(f"✅ Social media investigation completed: {investigation_id}")
            self.logger.info(f"📊 Found {len(results['profiles_found'])} profiles across {len(set(results['platforms_searched']))} platforms")

            return final_result

        except Exception as e:
            self.logger.error(f"❌ Social media investigation failed: {e}")
            # Return partial results even on failure
            return SocialMediaInvestigationResult(
                target_name=target.name,
                platforms_searched=results.get('platforms_searched', []),
                profiles_found=results.get('profiles_found', []),
                correlations=results.get('correlations', []),
                network_analysis={},
                confidence_score=0.0,
                investigation_duration=(datetime.now() - investigation_start).total_seconds(),
                raw_data=results.get('raw_data', {}),
                timestamp=datetime.now()
            )

    async def _execute_social_media_phase(
        self,
        target: InvestigationTarget,
        phase: SocialMediaPhase,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Execute individual social media investigation phase"""
        phase_results = {
            'platforms': [],
            'profiles': [],
            'correlations': [],
            'strategies_used': phase.search_strategies
        }

        try:
            # Execute strategies based on phase configuration
            for strategy in phase.search_strategies:
                if progress_callback:
                    progress_callback("social_media", f"Executing {strategy} on {len(phase.platforms)} platforms")

                strategy_results = await self._execute_search_strategy(target, strategy, phase.platforms)

                if strategy_results:
                    phase_results['platforms'].extend(strategy_results.get('platforms', []))
                    phase_results['profiles'].extend(strategy_results.get('profiles', []))

                # Rate limiting between strategies
                await asyncio.sleep(1)

            # Perform correlation analysis if enabled
            if phase.correlation_analysis and len(phase_results['profiles']) > 1:
                correlations = await self._perform_correlation_analysis(phase_results['profiles'])
                phase_results['correlations'].extend(correlations)

            # AI enhancement if enabled
            if phase.ai_enhancement and self.ai_orchestrator:
                phase_results = await self._ai_enhance_phase_results(target, phase_results)

            return phase_results

        except Exception as e:
            self.logger.error(f"❌ Social media phase execution failed: {e}")
            return phase_results

    async def _execute_search_strategy(
        self,
        target: InvestigationTarget,
        strategy: str,
        platforms: List[str]
    ) -> Dict[str, Any]:
        """Execute specific search strategy across platforms"""
        strategy_results = {
            'platforms': [],
            'profiles': []
        }

        try:
            if strategy == "name_search":
                # Search by name across specified platforms
                for platform in platforms:
                    if platform == "facebook" and self.facebook_scanner:
                        # Mock Facebook search (would be real implementation)
                        profiles = [{
                            'platform': 'facebook',
                            'name': target.name,
                            'url': f'https://facebook.com/search?q={target.name.replace(" ", "+")}',
                            'confidence': 0.7,
                            'found_method': 'name_search'
                        }]
                        strategy_results['profiles'].extend(profiles)
                        strategy_results['platforms'].append('facebook')

                    elif platform == "linkedin" and self.linkedin_scanner:
                        # Mock LinkedIn search
                        profiles = [{
                            'platform': 'linkedin',
                            'name': target.name,
                            'url': f'https://linkedin.com/search/results/people/?keywords={target.name.replace(" ", "%20")}',
                            'confidence': 0.8,
                            'found_method': 'name_search'
                        }]
                        strategy_results['profiles'].extend(profiles)
                        strategy_results['platforms'].append('linkedin')

                    elif platform == "instagram" and self.instagram_scanner:
                        # Mock Instagram search
                        profiles = [{
                            'platform': 'instagram',
                            'name': target.name,
                            'url': f'https://instagram.com/explore/search/keyword/?q={target.name.replace(" ", "+")}',
                            'confidence': 0.6,
                            'found_method': 'name_search'
                        }]
                        strategy_results['profiles'].extend(profiles)
                        strategy_results['platforms'].append('instagram')

            elif strategy == "cross_platform_correlation" and self.cross_platform_search:
                # Use cross-platform search for correlation
                # Mock implementation
                correlations = [{
                    'platforms': ['facebook', 'linkedin'],
                    'confidence': 0.85,
                    'correlation_type': 'name_email_match'
                }]
                strategy_results['correlations'] = correlations

            return strategy_results

        except Exception as e:
            self.logger.error(f"❌ Search strategy execution failed: {e}")
            return strategy_results

    async def _perform_correlation_analysis(self, profiles: List[Dict]) -> List[Dict]:
        """Perform entity correlation analysis on found profiles"""
        correlations = []

        try:
            if self.entity_correlation and len(profiles) > 1:
                # Convert profiles to EntityProfile objects
                entity_profiles = []
                for profile in profiles:
                    entity_profile = EntityProfile(
                        platform=profile['platform'],
                        username=profile.get('name', '').replace(' ', '_').lower(),
                        profile_url=profile.get('url', ''),
                        display_name=profile.get('name', ''),
                        metadata={'original_profile': profile}
                    )
                    entity_profiles.append(entity_profile)

                # Perform correlation analysis
                correlation_results = await self.entity_correlation.correlate_profiles(entity_profiles)

                for result in correlation_results:
                    correlations.append({
                        'profile1': result.profile1_id,
                        'profile2': result.profile2_id,
                        'similarity_score': result.similarity_score,
                        'correlation_type': 'cross_platform_match'
                    })

        except Exception as e:
            self.logger.error(f"❌ Correlation analysis failed: {e}")

        return correlations

    async def _ai_enhance_phase_results(
        self,
        target: InvestigationTarget,
        phase_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use AI to enhance and optimize phase results"""
        try:
            if self.ai_orchestrator:
                # Mock AI enhancement - would use actual AI models in real implementation
                enhanced_results = phase_results.copy()

                # AI could improve confidence scores
                for profile in enhanced_results['profiles']:
                    profile['ai_confidence_boost'] = 0.1
                    profile['confidence'] = min(1.0, profile.get('confidence', 0.5) + 0.1)

                # AI could suggest additional search strategies
                enhanced_results['ai_suggestions'] = [
                    'Consider searching for professional variations of the name',
                    'Check for maiden names or nicknames',
                    'Look for geographic location indicators'
                ]

                return enhanced_results

        except Exception as e:
            self.logger.error(f"❌ AI enhancement failed: {e}")

        return phase_results

    def _calculate_investigation_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence score for investigation"""
        try:
            profiles = results.get('profiles_found', [])
            if not profiles:
                return 0.0

            # Calculate average confidence of found profiles
            total_confidence = sum(profile.get('confidence', 0.0) for profile in profiles)
            avg_confidence = total_confidence / len(profiles)

            # Boost confidence if multiple platforms found the target
            platform_count = len(set(results.get('platforms_searched', [])))
            platform_boost = min(0.2, platform_count * 0.05)

            # Boost confidence if correlations found
            correlation_count = len(results.get('correlations', []))
            correlation_boost = min(0.15, correlation_count * 0.03)

            final_confidence = min(1.0, avg_confidence + platform_boost + correlation_boost)
            return round(final_confidence, 3)

        except Exception as e:
            self.logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.0

    async def cleanup(self):
        """Cleanup social media orchestrator resources"""
        try:
            self.logger.info("🧹 Cleaning up Social Media Orchestrator resources...")

            # Cleanup browser sessions in scanners
            if self.facebook_scanner and hasattr(self.facebook_scanner, 'browser_adapter'):
                if self.facebook_scanner.browser_adapter:
                    await self.facebook_scanner.browser_adapter.cleanup_sessions()

            if self.instagram_scanner and hasattr(self.instagram_scanner, 'browser_adapter'):
                if self.instagram_scanner.browser_adapter:
                    await self.instagram_scanner.browser_adapter.cleanup_sessions()

            if self.linkedin_scanner and hasattr(self.linkedin_scanner, 'browser_adapter'):
                if self.linkedin_scanner.browser_adapter:
                    await self.linkedin_scanner.browser_adapter.cleanup_sessions()

            self.logger.info("✅ Social Media Orchestrator cleanup completed")

        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")

# Global instance for easy access
_social_media_orchestrator = None

async def get_social_media_orchestrator(ai_orchestrator=None) -> SocialMediaOrchestrator:
    """Get global social media orchestrator instance"""
    global _social_media_orchestrator

    if _social_media_orchestrator is None:
        _social_media_orchestrator = SocialMediaOrchestrator(ai_orchestrator)
        await _social_media_orchestrator.initialize()

    return _social_media_orchestrator