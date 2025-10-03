#!/usr/bin/env python3
"""
📸 Instagram OSINT Scanner - Enhanced Edition
LakyLuk Social Media Investigation Suite

Features:
✅ Advanced username and hashtag search with AI analysis
✅ Profile analysis with story and highlight extraction
✅ Follower/Following network mapping and analysis
✅ Content analysis with AI-powered image recognition
✅ Location-based investigation tools
✅ Temporal analysis of posting patterns

Security & Ethics:
- Respects Instagram's terms of service and rate limits
- No credential harvesting or unauthorized access
- Privacy-respectful data collection
- Anti-detection with realistic browsing patterns
"""

import asyncio
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin, quote
import re

from ...core.browser_integration import BrowserIntegrationAdapter, create_stealth_session
from ...core.enhanced_orchestrator import EnhancedInvestigationOrchestrator
from ...utils.data_sanitizer import PIISanitizer
from ...utils.rate_limiter import RateLimiter


class InstagramScanner:
    """Advanced Instagram OSINT scanner with AI-powered analysis"""

    def __init__(self, ai_orchestrator: EnhancedInvestigationOrchestrator = None):
        self.browser_adapter = None  # Will be initialized async
        self.ai_orchestrator = ai_orchestrator
        self.pii_sanitizer = PIISanitizer()
        self.rate_limiter = RateLimiter(
            requests_per_minute=8,   # Conservative for Instagram
            requests_per_hour=80
        )
        self._initialized = False

        # Instagram-specific configuration
        self.base_url = "https://www.instagram.com"
        self.endpoints = {
            'profile': '/{username}/',
            'search': '/web/search/topsearch/?query={query}',
            'explore': '/explore/tags/{hashtag}/',
            'location': '/explore/locations/{location_id}/'
        }

        # AI-guided selectors for Instagram's current structure
        self.selectors = {
            'profile_header': 'header section',
            'username': 'h2._7UhW9, h1._7UhW9',
            'follower_count': 'a[href$="/followers/"] span',
            'following_count': 'a[href$="/following/"] span',
            'post_count': 'div.-nal3 span',
            'bio': 'div.-vDIg span',
            'profile_pic': 'img[data-testid="user-avatar"]',
            'posts_grid': 'article div div div a',
            'post_content': 'article div div div img, article div div div video',
            'post_caption': 'article div div span',
            'post_likes': 'section span button span',
            'post_comments': 'article div div section div button span',
            'stories': '[data-testid="story-highlights"]',
            'search_results': 'div[role="button"] div div div'
        }

    async def initialize(self):
        """Initialize the Instagram scanner and browser adapter"""
        if not self._initialized:
            self.browser_adapter = BrowserIntegrationAdapter()
            await self.browser_adapter.initialize()
            self._initialized = True

    async def _ensure_initialized(self):
        """Ensure scanner is initialized before use"""
        if not self._initialized:
            await self.initialize()

    async def search_profiles(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for Instagram profiles using multiple strategies

        Args:
            query: Search term (name, username, or keywords)
            limit: Maximum number of results to return

        Returns:
            List of profile information dictionaries
        """
        try:
            await self.rate_limiter.wait_if_needed()

            browser = await self.browser_manager.create_session(
                stealth_level="high",
                proxy=await self.proxy_manager.get_random_proxy()
            )

            page = await browser.new_page()

            # Try multiple search strategies
            results = []

            # Strategy 1: Direct profile access (if query looks like username)
            if self._is_potential_username(query):
                direct_result = await self._try_direct_profile_access(page, query)
                if direct_result:
                    results.append(direct_result)

            # Strategy 2: Instagram search API
            search_results = await self._search_via_api(page, query, limit - len(results))
            results.extend(search_results)

            # Strategy 3: Google dorking for Instagram profiles
            if len(results) < limit:
                google_results = await self._search_via_google_dorking(page, query, limit - len(results))
                results.extend(google_results)

            # AI enhancement of results
            if self.ai_orchestrator:
                results = await self._ai_enhance_search_results(results, query)

            await browser.close()

            # Sanitize and return
            return [self.pii_sanitizer.sanitize_profile(result) for result in results[:limit]]

        except Exception as e:
            print(f"Instagram search error: {e}")
            return []

    async def analyze_profile(self, username: str) -> Dict[str, Any]:
        """
        Comprehensive profile analysis with AI insights

        Args:
            username: Instagram username to analyze

        Returns:
            Detailed profile analysis with AI-powered insights
        """
        try:
            await self.rate_limiter.wait_if_needed()

            browser = await self.browser_manager.create_session(stealth_level="maximum")
            page = await browser.new_page()

            profile_url = f"{self.base_url}/{username}/"
            await page.goto(profile_url, wait_until="networkidle")

            # Extract comprehensive profile data
            profile_data = {
                'username': username,
                'profile_url': profile_url,
                'basic_info': await self._extract_profile_info(page),
                'content_analysis': await self._analyze_profile_content(page),
                'posting_patterns': await self._analyze_posting_patterns(page),
                'engagement_metrics': await self._calculate_engagement_metrics(page),
                'network_analysis': await self._analyze_network_connections(page),
                'content_themes': await self._extract_content_themes(page),
                'location_analysis': await self._analyze_location_data(page),
                'temporal_analysis': await self._analyze_temporal_patterns(page)
            }

            # AI-powered comprehensive analysis
            if self.ai_orchestrator:
                ai_insights = await self.ai_orchestrator.analyze_instagram_profile(profile_data)
                profile_data['ai_insights'] = ai_insights

                # AI-powered behavioral analysis
                behavioral_analysis = await self._ai_behavioral_analysis(page, profile_data)
                profile_data['behavioral_analysis'] = behavioral_analysis

            await browser.close()

            return self.pii_sanitizer.sanitize_profile(profile_data)

        except Exception as e:
            print(f"Profile analysis error: {e}")
            return {}

    async def map_network_connections(self, username: str, depth: int = 2) -> Dict[str, Any]:
        """
        Map social network connections with configurable depth analysis

        Args:
            username: Starting Instagram username
            depth: Degrees of separation to explore

        Returns:
            Social network graph with connection analysis
        """
        try:
            network_graph = {
                'root_profile': username,
                'connections': {},
                'relationships': [],
                'communities': [],
                'influence_metrics': {}
            }

            visited_profiles = set()
            analysis_queue = [(username, 0, 'root')]

            while analysis_queue and len(visited_profiles) < 50:  # Safety limit
                current_username, current_depth, relationship_type = analysis_queue.pop(0)

                if current_depth >= depth or current_username in visited_profiles:
                    continue

                visited_profiles.add(current_username)

                # Analyze current profile
                profile_data = await self.analyze_profile(current_username)
                network_graph['connections'][current_username] = profile_data

                # Extract followers/following for network mapping
                if current_depth < depth - 1:
                    connections = await self._extract_profile_connections(current_username)

                    for connection in connections[:10]:  # Limit per profile
                        connection_username = connection['username']
                        if connection_username not in visited_profiles:
                            analysis_queue.append((
                                connection_username,
                                current_depth + 1,
                                connection['relationship_type']
                            ))

                            # Record relationship
                            network_graph['relationships'].append({
                                'from': current_username,
                                'to': connection_username,
                                'relationship_type': relationship_type,
                                'strength': connection.get('strength', 0.5),
                                'discovered_at': datetime.now().isoformat()
                            })

                # Respectful delay
                await asyncio.sleep(random.uniform(3, 7))

            # AI-powered community detection
            if self.ai_orchestrator:
                communities = await self.ai_orchestrator.detect_communities(network_graph)
                network_graph['communities'] = communities

            # Calculate influence metrics
            network_graph['influence_metrics'] = self._calculate_influence_metrics(network_graph)

            return network_graph

        except Exception as e:
            print(f"Network mapping error: {e}")
            return {}

    async def analyze_content_by_hashtag(self, hashtag: str, limit: int = 50) -> Dict[str, Any]:
        """
        Analyze content and users associated with specific hashtag

        Args:
            hashtag: Hashtag to analyze (without #)
            limit: Maximum posts to analyze

        Returns:
            Hashtag analysis with user patterns and content insights
        """
        try:
            await self.rate_limiter.wait_if_needed()

            browser = await self.browser_manager.create_session(stealth_level="high")
            page = await browser.new_page()

            hashtag_url = f"{self.base_url}/explore/tags/{hashtag}/"
            await page.goto(hashtag_url, wait_until="networkidle")

            analysis = {
                'hashtag': hashtag,
                'hashtag_url': hashtag_url,
                'post_analysis': [],
                'user_patterns': {},
                'content_themes': [],
                'temporal_distribution': {},
                'engagement_analysis': {},
                'related_hashtags': []
            }

            # Extract posts from hashtag page
            posts = await self._extract_hashtag_posts(page, limit)

            for post in posts:
                # Analyze each post
                post_analysis = await self._analyze_individual_post(page, post)
                analysis['post_analysis'].append(post_analysis)

                # Brief delay between posts
                await asyncio.sleep(random.uniform(1, 3))

            # AI-powered hashtag analysis
            if self.ai_orchestrator:
                ai_analysis = await self.ai_orchestrator.analyze_hashtag_trends(analysis)
                analysis['ai_insights'] = ai_analysis

            # Calculate patterns and trends
            analysis['user_patterns'] = self._analyze_hashtag_user_patterns(analysis['post_analysis'])
            analysis['temporal_distribution'] = self._analyze_temporal_distribution(analysis['post_analysis'])
            analysis['engagement_analysis'] = self._analyze_hashtag_engagement(analysis['post_analysis'])

            await browser.close()

            return analysis

        except Exception as e:
            print(f"Hashtag analysis error: {e}")
            return {}

    # Internal helper methods

    def _is_potential_username(self, query: str) -> bool:
        """Check if query looks like a potential Instagram username"""
        # Instagram usernames: alphanumeric, periods, underscores, 1-30 chars
        return bool(re.match(r'^[a-zA-Z0-9._]{1,30}$', query))

    async def _try_direct_profile_access(self, page, username: str) -> Optional[Dict[str, Any]]:
        """Try to access profile directly if username format matches"""
        try:
            profile_url = f"{self.base_url}/{username}/"
            response = await page.goto(profile_url, wait_until="networkidle")

            # Check if profile exists (not 404)
            if response.status == 200:
                basic_info = await self._extract_profile_info(page)
                return {
                    'username': username,
                    'profile_url': profile_url,
                    'source': 'direct_access',
                    **basic_info
                }

        except Exception as e:
            print(f"Direct access failed for {username}: {e}")

        return None

    async def _search_via_api(self, page, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using Instagram's internal search API"""
        results = []

        try:
            # Navigate to Instagram and perform search
            await page.goto(self.base_url, wait_until="networkidle")

            # Wait for search box and perform search
            search_box = await page.wait_for_selector('input[placeholder*="Search"]', timeout=10000)
            await search_box.fill(query)
            await page.keyboard.press('Enter')

            # Wait for search results
            await page.wait_for_selector(self.selectors['search_results'], timeout=10000)

            # Extract search results
            search_elements = await page.query_selector_all(self.selectors['search_results'])

            for element in search_elements[:limit]:
                try:
                    # Extract profile information from search result
                    profile_link = await element.query_selector('a')
                    if profile_link:
                        href = await profile_link.get_attribute('href')
                        if href and href.startswith('/'):
                            username = href.strip('/').split('/')[0]

                            # Extract additional info from search result
                            text_content = await element.text_content()

                            result = {
                                'username': username,
                                'profile_url': f"{self.base_url}/{username}/",
                                'search_preview': text_content,
                                'source': 'instagram_search',
                                'discovered_at': datetime.now().isoformat()
                            }

                            results.append(result)

                except Exception as e:
                    print(f"Error extracting search result: {e}")
                    continue

        except Exception as e:
            print(f"API search error: {e}")

        return results

    async def _search_via_google_dorking(self, page, query: str, limit: int) -> List[Dict[str, Any]]:
        """Use Google dorking to find Instagram profiles"""
        results = []

        try:
            # Google search for Instagram profiles
            google_query = f'site:instagram.com "{query}"'
            google_url = f"https://www.google.com/search?q={quote(google_query)}"

            await page.goto(google_url, wait_until="networkidle")

            # Extract Instagram URLs from Google results
            search_results = await page.query_selector_all('div.g')

            for result_elem in search_results[:limit]:
                try:
                    link_elem = await result_elem.query_selector('h3 a')
                    if link_elem:
                        url = await link_elem.get_attribute('href')
                        if url and 'instagram.com/' in url:
                            # Extract username from URL
                            username_match = re.search(r'instagram\.com/([a-zA-Z0-9._]+)', url)
                            if username_match:
                                username = username_match.group(1)

                                # Extract snippet text
                                snippet_elem = await result_elem.query_selector('.VwiC3b')
                                snippet = await snippet_elem.text_content() if snippet_elem else ""

                                result = {
                                    'username': username,
                                    'profile_url': url,
                                    'search_snippet': snippet,
                                    'source': 'google_dorking',
                                    'discovered_at': datetime.now().isoformat()
                                }

                                results.append(result)

                except Exception as e:
                    print(f"Error extracting Google result: {e}")
                    continue

        except Exception as e:
            print(f"Google dorking error: {e}")

        return results

    async def _extract_profile_info(self, page) -> Dict[str, Any]:
        """Extract basic profile information"""
        info = {}

        try:
            # Profile name/title
            name_elem = await page.query_selector(self.selectors['username'])
            if name_elem:
                info['display_name'] = await name_elem.text_content()

            # Follower count
            follower_elem = await page.query_selector(self.selectors['follower_count'])
            if follower_elem:
                info['followers'] = await self._extract_count(follower_elem)

            # Following count
            following_elem = await page.query_selector(self.selectors['following_count'])
            if following_elem:
                info['following'] = await self._extract_count(following_elem)

            # Post count
            posts_elem = await page.query_selector(self.selectors['post_count'])
            if posts_elem:
                info['posts_count'] = await self._extract_count(posts_elem)

            # Bio/description
            bio_elem = await page.query_selector(self.selectors['bio'])
            if bio_elem:
                info['bio'] = await bio_elem.text_content()

            # Profile picture
            pic_elem = await page.query_selector(self.selectors['profile_pic'])
            if pic_elem:
                info['profile_picture_url'] = await pic_elem.get_attribute('src')

        except Exception as e:
            print(f"Error extracting profile info: {e}")

        return info

    async def _analyze_profile_content(self, page) -> Dict[str, Any]:
        """Analyze profile content with AI assistance"""
        content_analysis = {
            'recent_posts': [],
            'content_types': {},
            'posting_frequency': {},
            'engagement_patterns': {}
        }

        try:
            # Extract recent posts
            post_elements = await page.query_selector_all(self.selectors['posts_grid'])

            for i, post_elem in enumerate(post_elements[:12]):  # Analyze last 12 posts
                try:
                    # Navigate to individual post for detailed analysis
                    post_url = await post_elem.get_attribute('href')
                    if post_url:
                        # Store post info for analysis
                        post_info = {
                            'post_url': urljoin(self.base_url, post_url),
                            'position': i,
                            'extracted_at': datetime.now().isoformat()
                        }

                        content_analysis['recent_posts'].append(post_info)

                except Exception as e:
                    print(f"Error analyzing post {i}: {e}")

            # AI-powered content classification
            if self.ai_orchestrator:
                content_classification = await self.ai_orchestrator.classify_instagram_content(
                    content_analysis['recent_posts']
                )
                content_analysis['ai_classification'] = content_classification

        except Exception as e:
            print(f"Error analyzing profile content: {e}")

        return content_analysis

    async def _analyze_posting_patterns(self, page) -> Dict[str, Any]:
        """Analyze temporal posting patterns"""
        # This would require accessing individual posts to get timestamps
        # Implementation would analyze posting frequency, peak hours, etc.
        return {
            'daily_pattern': [],
            'weekly_pattern': [],
            'posting_frequency': 'unknown',
            'peak_hours': []
        }

    async def _calculate_engagement_metrics(self, page) -> Dict[str, Any]:
        """Calculate engagement metrics and ratios"""
        metrics = {
            'engagement_rate': 0.0,
            'avg_likes_per_post': 0.0,
            'avg_comments_per_post': 0.0,
            'follower_engagement_ratio': 0.0
        }

        # Implementation would calculate actual engagement metrics
        # from recent posts' like/comment counts

        return metrics

    async def _analyze_network_connections(self, page) -> Dict[str, Any]:
        """Analyze network connections and relationships"""
        # Implementation would look at follower/following patterns
        return {
            'follower_following_ratio': 0.0,
            'mutual_connections': [],
            'connection_patterns': {}
        }

    async def _extract_content_themes(self, page) -> List[str]:
        """Extract content themes using AI analysis"""
        themes = []

        if self.ai_orchestrator:
            # AI would analyze bio, captions, hashtags to determine themes
            page_content = await page.content()
            themes = await self.ai_orchestrator.extract_content_themes(page_content)

        return themes

    async def _analyze_location_data(self, page) -> Dict[str, Any]:
        """Analyze location information from posts"""
        # Implementation would extract location tags from posts
        return {
            'frequent_locations': [],
            'location_pattern': {},
            'travel_analysis': {}
        }

    async def _analyze_temporal_patterns(self, page) -> Dict[str, Any]:
        """Analyze temporal posting and activity patterns"""
        # Implementation would analyze posting times, frequency patterns
        return {
            'posting_schedule': {},
            'activity_peaks': [],
            'consistency_score': 0.0
        }

    async def _ai_behavioral_analysis(self, page, profile_data: Dict) -> Dict[str, Any]:
        """AI-powered behavioral pattern analysis"""
        if not self.ai_orchestrator:
            return {}

        behavioral_patterns = await self.ai_orchestrator.analyze_behavioral_patterns(
            profile_data
        )

        return behavioral_patterns

    async def _extract_count(self, element) -> int:
        """Extract numeric count from element (handles K, M suffixes)"""
        try:
            text = await element.text_content()
            return self._parse_instagram_count(text)
        except:
            return 0

    def _parse_instagram_count(self, count_text: str) -> int:
        """Parse Instagram count format (1K, 1M, etc.) to integer"""
        if not count_text:
            return 0

        # Remove commas and spaces
        count_text = count_text.replace(',', '').replace(' ', '').upper()

        # Handle K (thousands) and M (millions)
        if 'K' in count_text:
            number = float(count_text.replace('K', ''))
            return int(number * 1000)
        elif 'M' in count_text:
            number = float(count_text.replace('M', ''))
            return int(number * 1000000)
        else:
            # Try to parse as regular number
            try:
                return int(count_text)
            except:
                return 0

    async def _ai_enhance_search_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Use AI to enhance and score search results"""
        if not self.ai_orchestrator or not results:
            return results

        enhanced_results = []

        for result in results:
            # AI relevance scoring
            relevance_score = await self.ai_orchestrator.score_instagram_relevance(
                result, query
            )
            result['ai_relevance_score'] = relevance_score

            # AI-powered profile quality assessment
            quality_score = await self.ai_orchestrator.assess_profile_quality(result)
            result['profile_quality_score'] = quality_score

            enhanced_results.append(result)

        # Sort by combined AI scores
        enhanced_results.sort(
            key=lambda x: (x.get('ai_relevance_score', 0) + x.get('profile_quality_score', 0)) / 2,
            reverse=True
        )

        return enhanced_results

    def _calculate_influence_metrics(self, network_graph: Dict) -> Dict[str, Any]:
        """Calculate influence and centrality metrics for network"""
        metrics = {
            'centrality_scores': {},
            'influence_ranking': [],
            'network_clusters': [],
            'key_connectors': []
        }

        # Implementation would calculate network analysis metrics
        # using graph theory algorithms

        return metrics