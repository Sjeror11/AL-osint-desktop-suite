#!/usr/bin/env python3
"""
🔍 Bing Search API Integration
OSINT Desktop Suite - Web Search Tools

Features:
- Bing Search API v7 integration
- Web, news, images, videos search
- Advanced search filters
- Results correlation with Google
- Rate limiting and caching
"""

import os
import asyncio
import aiohttp
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus
import time

class BingSearchTool:
    """Bing Search API wrapper for comprehensive OSINT investigations"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Bing Search Tool

        Args:
            api_key: Bing Search API key
        """
        self.api_key = api_key or os.getenv('BING_SEARCH_API_KEY')
        self.base_url = "https://api.bing.microsoft.com/v7.0"

        # Different search endpoints
        self.endpoints = {
            'web': f"{self.base_url}/search",
            'news': f"{self.base_url}/news/search",
            'images': f"{self.base_url}/images/search",
            'videos': f"{self.base_url}/videos/search"
        }

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Cache
        self.cache = {}
        self.cache_duration = timedelta(hours=1)

        # Logger
        self.logger = logging.getLogger(__name__)

        if not self.api_key:
            self.logger.warning("⚠️ Bing Search API key not found - some features will be limited")

    def _get_cache_key(self, search_type: str, query: str, **params) -> str:
        """Generate cache key for query"""
        cache_data = f"{search_type}_{query}_{str(sorted(params.items()))}"
        return f"bing_{hash(cache_data)}"

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cache entry is still valid"""
        return datetime.now() - timestamp < self.cache_duration

    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        search_type: str = "web"
    ) -> Dict[str, Any]:
        """Make HTTP request to Bing API"""

        if not self.api_key:
            return {"error": "API key not configured", "results": []}

        # Check cache
        cache_key = self._get_cache_key(search_type, params.get('q', ''), **params)

        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.logger.info(f"📦 Using cached {search_type} results")
                return cached_data

        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            await self._rate_limit()

            self.logger.info(f"🔍 Bing {search_type} search: {params.get('q', '')}")

            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_response(data, search_type)

                        # Cache results
                        self.cache[cache_key] = (results, datetime.now())

                        self.logger.info(f"✅ Found {len(results.get('results', []))} {search_type} results")
                        return results

                    elif response.status == 429:
                        self.logger.warning("⚠️ Bing Search API rate limit exceeded")
                        return {"error": "Rate limit exceeded", "results": []}

                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ Bing Search API error {response.status}: {error_text}")
                        return {"error": f"API error {response.status}", "results": []}

        except asyncio.TimeoutError:
            self.logger.error("❌ Bing Search API timeout")
            return {"error": "Timeout", "results": []}

        except Exception as e:
            self.logger.error(f"❌ Bing Search error: {e}")
            return {"error": str(e), "results": []}

    def _parse_response(self, data: Dict[str, Any], search_type: str) -> Dict[str, Any]:
        """Parse Bing API response based on search type"""

        results = {
            "search_type": search_type,
            "query": data.get('queryContext', {}).get('originalQuery', ''),
            "timestamp": datetime.now().isoformat(),
            "results": [],
            "metadata": {}
        }

        if search_type == "web":
            web_pages = data.get('webPages', {})
            results["total_estimated"] = web_pages.get('totalEstimatedMatches', 0)

            for page in web_pages.get('value', []):
                result = {
                    "title": page.get('name', ''),
                    "url": page.get('url', ''),
                    "snippet": page.get('snippet', ''),
                    "display_url": page.get('displayUrl', ''),
                    "date_last_crawled": page.get('dateLastCrawled', ''),
                    "language": page.get('language', ''),
                    "is_family_friendly": page.get('isFamilyFriendly', True),
                    "deep_links": page.get('deepLinks', [])
                }
                results["results"].append(result)

        elif search_type == "news":
            news_articles = data.get('value', [])

            for article in news_articles:
                result = {
                    "title": article.get('name', ''),
                    "url": article.get('url', ''),
                    "description": article.get('description', ''),
                    "provider": article.get('provider', [{}])[0].get('name', '') if article.get('provider') else '',
                    "date_published": article.get('datePublished', ''),
                    "category": article.get('category', ''),
                    "image": article.get('image', {}),
                    "mentions": article.get('mentions', [])
                }
                results["results"].append(result)

        elif search_type == "images":
            images = data.get('value', [])

            for image in images:
                result = {
                    "title": image.get('name', ''),
                    "content_url": image.get('contentUrl', ''),
                    "host_page_url": image.get('hostPageUrl', ''),
                    "thumbnail_url": image.get('thumbnailUrl', ''),
                    "width": image.get('width', 0),
                    "height": image.get('height', 0),
                    "file_size": image.get('contentSize', ''),
                    "encoding_format": image.get('encodingFormat', ''),
                    "host_page_display_url": image.get('hostPageDisplayUrl', ''),
                    "date_published": image.get('datePublished', '')
                }
                results["results"].append(result)

        elif search_type == "videos":
            videos = data.get('value', [])

            for video in videos:
                result = {
                    "title": video.get('name', ''),
                    "description": video.get('description', ''),
                    "web_search_url": video.get('webSearchUrl', ''),
                    "thumbnail_url": video.get('thumbnailUrl', ''),
                    "date_published": video.get('datePublished', ''),
                    "publisher": video.get('publisher', [{}])[0].get('name', '') if video.get('publisher') else '',
                    "duration": video.get('duration', ''),
                    "view_count": video.get('viewCount', 0),
                    "embed_html": video.get('embedHtml', '')
                }
                results["results"].append(result)

        return results

    async def web_search(
        self,
        query: str,
        count: int = 10,
        offset: int = 0,
        market: str = 'cs-CZ',
        safe_search: str = 'Off',
        text_decorations: bool = True,
        text_format: str = 'HTML'
    ) -> Dict[str, Any]:
        """
        Perform web search

        Args:
            query: Search query
            count: Number of results (1-50)
            offset: Starting offset for pagination
            market: Market code (cs-CZ for Czech Republic)
            safe_search: Safe search setting (Off, Moderate, Strict)
            text_decorations: Whether to include text decorations
            text_format: Response format (HTML, Raw)

        Returns:
            Web search results
        """

        params = {
            'q': query,
            'count': min(count, 50),
            'offset': offset,
            'mkt': market,
            'safeSearch': safe_search,
            'textDecorations': text_decorations,
            'textFormat': text_format,
            'responseFilter': 'WebPages'
        }

        return await self._make_request(self.endpoints['web'], params, 'web')

    async def news_search(
        self,
        query: str,
        count: int = 10,
        offset: int = 0,
        market: str = 'cs-CZ',
        category: Optional[str] = None,
        sort_by: str = 'Date',
        freshness: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for news articles

        Args:
            query: Search query
            count: Number of results (1-100)
            offset: Starting offset
            market: Market code
            category: News category (Business, Entertainment, etc.)
            sort_by: Sort order (Date, Relevance)
            freshness: Time filter (Day, Week, Month)

        Returns:
            News search results
        """

        params = {
            'q': query,
            'count': min(count, 100),
            'offset': offset,
            'mkt': market,
            'sortBy': sort_by
        }

        if category:
            params['category'] = category

        if freshness:
            params['freshness'] = freshness

        return await self._make_request(self.endpoints['news'], params, 'news')

    async def image_search(
        self,
        query: str,
        count: int = 10,
        offset: int = 0,
        market: str = 'cs-CZ',
        image_type: Optional[str] = None,
        size: Optional[str] = None,
        color: Optional[str] = None,
        license: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for images

        Args:
            query: Search query
            count: Number of results (1-150)
            offset: Starting offset
            market: Market code
            image_type: Type (Photo, Clipart, Line, Shopping)
            size: Size filter (Small, Medium, Large, etc.)
            color: Color filter (ColorOnly, Monochrome, etc.)
            license: License filter (Public, Share, etc.)

        Returns:
            Image search results
        """

        params = {
            'q': query,
            'count': min(count, 150),
            'offset': offset,
            'mkt': market
        }

        if image_type:
            params['imageType'] = image_type
        if size:
            params['size'] = size
        if color:
            params['color'] = color
        if license:
            params['license'] = license

        return await self._make_request(self.endpoints['images'], params, 'images')

    async def comprehensive_search(self, target_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive multi-type search for OSINT

        Args:
            target_name: Target name or entity to investigate

        Returns:
            Combined results from all search types
        """

        self.logger.info(f"🎯 Starting comprehensive Bing search for: {target_name}")

        # Search strategies for different types
        search_queries = {
            "basic": f'"{target_name}"',
            "profile": f'{target_name} profil kontakt',
            "social": f'{target_name} facebook linkedin instagram',
            "business": f'{target_name} firma společnost',
            "czech_sites": f'{target_name} site:justice.cz OR site:ares.gov.cz OR site:firmy.cz'
        }

        results = {
            "target": target_name,
            "timestamp": datetime.now().isoformat(),
            "searches": {}
        }

        # Perform web searches
        for search_name, query in search_queries.items():
            self.logger.info(f"🔍 Web search - {search_name}: {query}")

            web_result = await self.web_search(query, count=10)
            results["searches"][f"web_{search_name}"] = web_result

            await asyncio.sleep(0.2)  # Small delay

        # News search
        self.logger.info(f"📰 News search for: {target_name}")
        news_result = await self.news_search(f'"{target_name}"', count=20, freshness='Month')
        results["searches"]["news"] = news_result

        await asyncio.sleep(0.2)

        # Image search
        self.logger.info(f"🖼️ Image search for: {target_name}")
        image_result = await self.image_search(f'"{target_name}"', count=30)
        results["searches"]["images"] = image_result

        # Generate summary
        total_web_results = sum(
            len(search.get('results', []))
            for key, search in results["searches"].items()
            if key.startswith('web_')
        )

        results["summary"] = {
            "total_web_results": total_web_results,
            "news_articles": len(results["searches"]["news"].get('results', [])),
            "images_found": len(results["searches"]["images"].get('results', [])),
            "unique_domains": list(set(
                result.get('display_url', '').split('/')[0]
                for search in results["searches"].values()
                if search.get('results')
                for result in search['results']
                if result.get('display_url')
            ))
        }

        self.logger.info(f"✅ Comprehensive search completed: {total_web_results} web results, "
                        f"{results['summary']['news_articles']} news, "
                        f"{results['summary']['images_found']} images")

        return results

    async def czech_osint_search(self, target_name: str) -> Dict[str, Any]:
        """
        Specialized search for Czech OSINT sources

        Args:
            target_name: Target name to investigate

        Returns:
            Results from Czech-specific sources
        """

        czech_sources = {
            "justice": f'{target_name} site:justice.cz',
            "ares": f'{target_name} site:ares.gov.cz',
            "firmy": f'{target_name} site:firmy.cz',
            "zivnostenske": f'{target_name} site:zivnostenskyrejstrik.cz',
            "ceska_posta": f'{target_name} site:psc.ceskaposta.cz',
            "cesky_rozhlas": f'{target_name} site:radiozurnal.rozhlas.cz',
            "ceska_televize": f'{target_name} site:ct24.ceskatelevize.cz',
            "idnes": f'{target_name} site:idnes.cz',
            "novinky": f'{target_name} site:novinky.cz'
        }

        results = {
            "target": target_name,
            "timestamp": datetime.now().isoformat(),
            "czech_sources": {}
        }

        for source_name, query in czech_sources.items():
            self.logger.info(f"🇨🇿 Czech search - {source_name}: {query}")

            result = await self.web_search(query, count=5)
            if result.get('results'):
                results["czech_sources"][source_name] = result

            await asyncio.sleep(0.3)  # Longer delay for respectful crawling

        return results

    def get_search_filters_help(self) -> Dict[str, Dict[str, str]]:
        """Get help for Bing search filters and operators"""

        return {
            "Basic Operators": {
                '"exact phrase"': 'Search for exact phrase',
                'word1 OR word2': 'Search for either word',
                'word1 AND word2': 'Search for both words',
                'NOT unwanted': 'Exclude word from results',
                '(word1 OR word2) AND word3': 'Group terms with parentheses'
            },
            "Site Operators": {
                'site:example.com': 'Search only on specific site',
                'domain:example.com': 'Search specific domain and subdomains',
                'url:keyword': 'Search for keyword in URL',
                'inurl:keyword': 'Keyword appears in URL',
                'intitle:keyword': 'Keyword appears in title',
                'inbody:keyword': 'Keyword appears in body text'
            },
            "Content Filters": {
                'filetype:pdf': 'Search for PDF files',
                'filetype:doc': 'Search for Word documents',
                'filetype:xls': 'Search for Excel files',
                'filetype:ppt': 'Search for PowerPoint files',
                'contains:downloads': 'Pages with download links',
                'feed:url': 'Find RSS feeds'
            },
            "Location & Language": {
                'loc:CZ': 'Results from Czech Republic',
                'language:cs': 'Results in Czech language',
                'prefer:english': 'Prefer English results'
            },
            "Advanced OSINT": {
                'ip:192.168.1.1': 'Search for specific IP address',
                'linkfromdomain:example.com': 'Links from specific domain',
                'related:example.com': 'Sites related to domain',
                'cache:url': 'Show cached version (if available)'
            }
        }

# Example usage and testing
async def test_bing_search():
    """Test Bing Search functionality"""

    # Initialize with API key from environment
    bing = BingSearchTool()

    if not bing.api_key:
        print("⚠️ No Bing Search API key found")
        print("💡 Set BING_SEARCH_API_KEY environment variable")
        return

    # Test web search
    print("🔍 Testing web search...")
    web_result = await bing.web_search("OSINT nástroje Česká republika", count=5)
    print(f"📊 Web results: {len(web_result.get('results', []))}")

    # Test news search
    print("\n📰 Testing news search...")
    news_result = await bing.news_search("kybernetická bezpečnost", count=5, freshness='Week')
    print(f"📊 News results: {len(news_result.get('results', []))}")

    # Test comprehensive search
    print("\n🎯 Testing comprehensive search...")
    comp_result = await bing.comprehensive_search("Jan Novák")
    print(f"📊 Total web results: {comp_result['summary']['total_web_results']}")
    print(f"📊 News articles: {comp_result['summary']['news_articles']}")
    print(f"📊 Images found: {comp_result['summary']['images_found']}")

    # Test Czech OSINT search
    print("\n🇨🇿 Testing Czech OSINT search...")
    czech_result = await bing.czech_osint_search("Test Firma")
    print(f"📊 Czech sources with results: {len(czech_result['czech_sources'])}")

    # Print search filters help
    print("\n📖 Bing Search Filters:")
    filters = bing.get_search_filters_help()

    for category, ops in filters.items():
        print(f"\n{category}:")
        for operator, description in ops.items():
            print(f"  {operator:<25} - {description}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_bing_search())