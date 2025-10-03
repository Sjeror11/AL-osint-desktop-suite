#!/usr/bin/env python3
"""
🔍 Google Search API Integration
OSINT Desktop Suite - Web Search Tools

Features:
- Google Custom Search API integration
- Advanced search operators support
- Results filtering and parsing
- Rate limiting and error handling
- Cache support for performance
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

class GoogleSearchTool:
    """Google Custom Search API wrapper for OSINT investigations"""

    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        """
        Initialize Google Search Tool

        Args:
            api_key: Google Custom Search API key
            search_engine_id: Custom Search Engine ID
        """
        self.api_key = api_key or os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id = search_engine_id or os.getenv('GOOGLE_SEARCH_ENGINE_ID', '017576662512468239146:omuauf_lfve')
        self.base_url = "https://www.googleapis.com/customsearch/v1"

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Cache
        self.cache = {}
        self.cache_duration = timedelta(hours=1)

        # Logger
        self.logger = logging.getLogger(__name__)

        if not self.api_key:
            self.logger.warning("⚠️ Google Search API key not found - some features will be limited")

    def _get_cache_key(self, query: str, **params) -> str:
        """Generate cache key for query"""
        cache_data = f"{query}_{str(sorted(params.items()))}"
        return f"google_{hash(cache_data)}"

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

    async def search(
        self,
        query: str,
        num_results: int = 10,
        start_index: int = 1,
        site_search: Optional[str] = None,
        file_type: Optional[str] = None,
        date_restrict: Optional[str] = None,
        country: str = 'cz',
        language: str = 'cs',
        safe_search: str = 'off'
    ) -> Dict[str, Any]:
        """
        Perform Google search with advanced parameters

        Args:
            query: Search query
            num_results: Number of results (1-10)
            start_index: Starting index for pagination
            site_search: Restrict search to specific site
            file_type: Filter by file type (pdf, doc, etc.)
            date_restrict: Date restriction (d1, w1, m1, y1)
            country: Country code for localized results
            language: Language code
            safe_search: Safe search setting (off, medium, high)

        Returns:
            Dictionary with search results and metadata
        """

        if not self.api_key:
            self.logger.error("❌ Google Search API key not configured")
            return {"error": "API key not configured", "results": []}

        # Check cache first
        cache_key = self._get_cache_key(
            query, num_results=num_results, start_index=start_index,
            site_search=site_search, file_type=file_type, date_restrict=date_restrict
        )

        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.logger.info(f"📦 Using cached results for: {query}")
                return cached_data

        # Build search parameters
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10),  # Max 10 per request
            'start': start_index,
            'gl': country,
            'hl': language,
            'safe': safe_search
        }

        # Add optional parameters
        if site_search:
            params['siteSearch'] = site_search

        if file_type:
            params['fileType'] = file_type

        if date_restrict:
            params['dateRestrict'] = date_restrict

        try:
            # Apply rate limiting
            await self._rate_limit()

            self.logger.info(f"🔍 Searching Google for: {query}")

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_results(data)

                        # Cache results
                        self.cache[cache_key] = (results, datetime.now())

                        self.logger.info(f"✅ Found {len(results.get('results', []))} results")
                        return results

                    elif response.status == 429:
                        self.logger.warning("⚠️ Google Search API rate limit exceeded")
                        return {"error": "Rate limit exceeded", "results": []}

                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ Google Search API error {response.status}: {error_text}")
                        return {"error": f"API error {response.status}", "results": []}

        except asyncio.TimeoutError:
            self.logger.error("❌ Google Search API timeout")
            return {"error": "Timeout", "results": []}

        except Exception as e:
            self.logger.error(f"❌ Google Search error: {e}")
            return {"error": str(e), "results": []}

    def _parse_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Google Search API response"""

        results = {
            "query": data.get('queries', {}).get('request', [{}])[0].get('searchTerms', ''),
            "total_results": int(data.get('searchInformation', {}).get('totalResults', 0)),
            "search_time": float(data.get('searchInformation', {}).get('searchTime', 0)),
            "results": [],
            "metadata": {
                "spelling": data.get('spelling', {}),
                "context": data.get('context', {}),
                "timestamp": datetime.now().isoformat()
            }
        }

        # Parse individual results
        for item in data.get('items', []):
            result = {
                "title": item.get('title', ''),
                "link": item.get('link', ''),
                "snippet": item.get('snippet', ''),
                "display_link": item.get('displayLink', ''),
                "formatted_url": item.get('formattedUrl', ''),
                "file_format": item.get('fileFormat', ''),
                "mime": item.get('mime', ''),
                "page_map": item.get('pagemap', {}),
                "cache_id": item.get('cacheId', ''),
                "image": item.get('image', {}) if 'image' in item else None
            }

            results["results"].append(result)

        return results

    async def advanced_search(self, target_name: str) -> Dict[str, Any]:
        """
        Perform advanced OSINT search for a target

        Args:
            target_name: Name or entity to investigate

        Returns:
            Comprehensive search results across multiple queries
        """

        self.logger.info(f"🎯 Starting advanced search for: {target_name}")

        # Define search strategies
        search_queries = [
            # Basic searches
            f'"{target_name}"',
            f'{target_name} profil',
            f'{target_name} kontakt',

            # Social media
            f'{target_name} site:facebook.com',
            f'{target_name} site:linkedin.com',
            f'{target_name} site:instagram.com',
            f'{target_name} site:twitter.com',

            # Czech specific
            f'{target_name} site:justice.cz',
            f'{target_name} site:ares.gov.cz',
            f'{target_name} site:firmy.cz',
            f'{target_name} site:zivnostenskyrejstrik.cz',

            # Professional
            f'{target_name} CV životopis',
            f'{target_name} firma společnost',
            f'{target_name} zaměstnanec employee',

            # Documents
            f'{target_name} filetype:pdf',
            f'{target_name} filetype:doc',
            f'{target_name} filetype:docx',
        ]

        all_results = {
            "target": target_name,
            "timestamp": datetime.now().isoformat(),
            "queries": [],
            "summary": {
                "total_queries": len(search_queries),
                "total_results": 0,
                "unique_domains": set(),
                "file_types": set()
            }
        }

        # Execute searches
        for i, query in enumerate(search_queries):
            self.logger.info(f"🔍 Query {i+1}/{len(search_queries)}: {query}")

            result = await self.search(query, num_results=10)

            query_data = {
                "query": query,
                "results_count": len(result.get('results', [])),
                "total_found": result.get('total_results', 0),
                "search_time": result.get('search_time', 0),
                "results": result.get('results', []),
                "error": result.get('error')
            }

            all_results["queries"].append(query_data)

            # Update summary
            all_results["summary"]["total_results"] += len(result.get('results', []))

            for res in result.get('results', []):
                if res.get('display_link'):
                    all_results["summary"]["unique_domains"].add(res['display_link'])
                if res.get('file_format'):
                    all_results["summary"]["file_types"].add(res['file_format'])

            # Small delay between queries
            await asyncio.sleep(0.2)

        # Convert sets to lists for JSON serialization
        all_results["summary"]["unique_domains"] = list(all_results["summary"]["unique_domains"])
        all_results["summary"]["file_types"] = list(all_results["summary"]["file_types"])

        self.logger.info(f"✅ Advanced search completed: {all_results['summary']['total_results']} total results")

        return all_results

    async def search_news(self, query: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Search for recent news articles

        Args:
            query: Search query
            days_back: How many days back to search

        Returns:
            News search results
        """

        # Date restriction for recent news
        date_restrict = f'd{days_back}' if days_back <= 365 else 'y1'

        # Search news sites
        news_sites = [
            'site:idnes.cz OR site:novinky.cz OR site:aktualne.cz',
            'site:denik.cz OR site:blesk.cz OR site:tn.cz',
            'site:ct24.ceskatelevize.cz OR site:radiozurnal.rozhlas.cz'
        ]

        all_news = {
            "query": query,
            "date_range": f"Last {days_back} days",
            "timestamp": datetime.now().isoformat(),
            "sources": []
        }

        for site_restriction in news_sites:
            search_query = f'{query} {site_restriction}'

            result = await self.search(
                search_query,
                num_results=10,
                date_restrict=date_restrict
            )

            if result.get('results'):
                all_news["sources"].append({
                    "site_restriction": site_restriction,
                    "results_count": len(result['results']),
                    "results": result['results']
                })

        return all_news

    def get_search_operators_help(self) -> Dict[str, str]:
        """Get help for Google search operators"""

        return {
            "Basic": {
                '"exact phrase"': 'Search for exact phrase',
                'word1 OR word2': 'Search for either word',
                'word1 AND word2': 'Search for both words',
                '-unwanted': 'Exclude word from results',
                '*': 'Wildcard for unknown words',
                '()': 'Group terms together'
            },
            "Site specific": {
                'site:example.com': 'Search only on specific site',
                'site:*.example.com': 'Search all subdomains',
                'inurl:keyword': 'Keyword appears in URL',
                'intitle:keyword': 'Keyword appears in title',
                'intext:keyword': 'Keyword appears in text'
            },
            "File types": {
                'filetype:pdf': 'Search for PDF files',
                'filetype:doc': 'Search for Word documents',
                'filetype:xls': 'Search for Excel files',
                'filetype:ppt': 'Search for PowerPoint files'
            },
            "Date/Location": {
                'after:2020': 'Results after specific year',
                'before:2023': 'Results before specific year',
                'location:"Prague"': 'Results from specific location'
            },
            "OSINT specific": {
                'cache:url': 'Show cached version of page',
                'related:url': 'Find related pages',
                'info:url': 'Get info about specific URL',
                'define:term': 'Get definition of term'
            }
        }

# Example usage and testing
async def test_google_search():
    """Test Google Search functionality"""

    # Initialize with API key from environment
    google = GoogleSearchTool()

    if not google.api_key:
        print("⚠️ No Google Search API key found")
        print("💡 Set GOOGLE_SEARCH_API_KEY environment variable")
        return

    # Test basic search
    print("🔍 Testing basic search...")
    result = await google.search("OSINT tools Czech Republic", num_results=5)

    print(f"📊 Results: {len(result.get('results', []))}")
    print(f"⏱️ Search time: {result.get('search_time', 0)} seconds")

    # Test advanced search
    print("\n🎯 Testing advanced search...")
    advanced_result = await google.advanced_search("Jan Novák")

    print(f"📊 Total queries: {advanced_result['summary']['total_queries']}")
    print(f"📊 Total results: {advanced_result['summary']['total_results']}")
    print(f"🌐 Unique domains: {len(advanced_result['summary']['unique_domains'])}")

    # Print search operators help
    print("\n📖 Google Search Operators:")
    operators = google.get_search_operators_help()

    for category, ops in operators.items():
        print(f"\n{category}:")
        for operator, description in ops.items():
            print(f"  {operator:<20} - {description}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_google_search())