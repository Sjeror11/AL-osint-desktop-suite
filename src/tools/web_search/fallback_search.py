#!/usr/bin/env python3
"""
🔍 Fallback Web Search Tool
Basic web scraping when API keys are not available

Features:
- Direct Google/Bing scraping (respectful)
- No API keys required
- Basic result extraction
- Rate limiting for ethical use
"""

import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus
import time
import re
from bs4 import BeautifulSoup

class FallbackSearchTool:
    """Fallback web search using direct scraping when APIs unavailable"""

    def __init__(self):
        """Initialize fallback search tool"""

        self.base_urls = {
            'google': 'https://www.google.com/search',
            'bing': 'https://www.bing.com/search'
        }

        # Rate limiting for respectful scraping
        self.last_request_time = 0
        self.min_request_interval = 3.0  # 3 seconds between requests

        # Request headers to appear like regular browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'cs,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        self.logger = logging.getLogger(__name__)

    async def _rate_limit(self):
        """Implement respectful rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    async def _make_request(self, url: str, params: Dict[str, str]) -> Optional[str]:
        """Make HTTP request with rate limiting"""

        await self._rate_limit()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self.headers, timeout=10) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        self.logger.warning(f"⚠️ HTTP {response.status} for fallback search")
                        return None

        except asyncio.TimeoutError:
            self.logger.error("❌ Timeout for fallback search")
            return None
        except Exception as e:
            self.logger.error(f"❌ Fallback search error: {e}")
            return None

    def _parse_google_results(self, html: str) -> List[Dict[str, str]]:
        """Parse Google search results from HTML"""

        results = []

        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Find result containers (Google structure can change)
            result_containers = soup.find_all('div', {'class': re.compile(r'g\b')})

            for container in result_containers[:10]:  # Limit to 10 results
                try:
                    # Try to find title and link
                    title_element = container.find('h3')
                    link_element = container.find('a')
                    snippet_element = container.find('div', {'class': re.compile(r'VwiC3b|IsZvec')})

                    if title_element and link_element:
                        result = {
                            'title': title_element.get_text(strip=True),
                            'url': link_element.get('href', ''),
                            'snippet': snippet_element.get_text(strip=True) if snippet_element else '',
                            'source': 'google_fallback'
                        }

                        # Filter out non-HTTP URLs
                        if result['url'].startswith('http'):
                            results.append(result)

                except Exception as e:
                    continue

        except Exception as e:
            self.logger.error(f"❌ Google results parsing error: {e}")

        return results

    def _parse_bing_results(self, html: str) -> List[Dict[str, str]]:
        """Parse Bing search results from HTML"""

        results = []

        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Find result containers (Bing structure)
            result_containers = soup.find_all('li', {'class': 'b_algo'})

            for container in result_containers[:10]:  # Limit to 10 results
                try:
                    # Try to find title and link
                    title_element = container.find('h2')
                    link_element = title_element.find('a') if title_element else None
                    snippet_element = container.find('p')

                    if title_element and link_element:
                        result = {
                            'title': title_element.get_text(strip=True),
                            'url': link_element.get('href', ''),
                            'snippet': snippet_element.get_text(strip=True) if snippet_element else '',
                            'source': 'bing_fallback'
                        }

                        # Filter out non-HTTP URLs
                        if result['url'].startswith('http'):
                            results.append(result)

                except Exception as e:
                    continue

        except Exception as e:
            self.logger.error(f"❌ Bing results parsing error: {e}")

        return results

    async def google_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Perform Google search using fallback scraping"""

        self.logger.info(f"🔍 Google fallback search: {query}")

        params = {
            'q': query,
            'num': min(num_results, 10),
            'hl': 'cs',
            'gl': 'cz'
        }

        html = await self._make_request(self.base_urls['google'], params)

        if not html:
            return {
                'error': 'Failed to fetch Google results',
                'results': [],
                'query': query
            }

        results = self._parse_google_results(html)

        return {
            'query': query,
            'results': results,
            'total_found': len(results),
            'source': 'google_fallback',
            'timestamp': datetime.now().isoformat(),
            'note': 'Fallback scraping - limited results'
        }

    async def bing_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Perform Bing search using fallback scraping"""

        self.logger.info(f"🔍 Bing fallback search: {query}")

        params = {
            'q': query,
            'count': min(num_results, 10),
            'cc': 'CZ',
            'setLang': 'cs'
        }

        html = await self._make_request(self.base_urls['bing'], params)

        if not html:
            return {
                'error': 'Failed to fetch Bing results',
                'results': [],
                'query': query
            }

        results = self._parse_bing_results(html)

        return {
            'query': query,
            'results': results,
            'total_found': len(results),
            'source': 'bing_fallback',
            'timestamp': datetime.now().isoformat(),
            'note': 'Fallback scraping - limited results'
        }

    async def multi_search(self, query: str, engines: List[str] = ['google', 'bing']) -> Dict[str, Any]:
        """Perform search across multiple engines"""

        self.logger.info(f"🎯 Multi-engine fallback search: {query}")

        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'engines': {},
            'consolidated_results': []
        }

        search_tasks = []

        if 'google' in engines:
            search_tasks.append(('google', self.google_search(query)))

        if 'bing' in engines:
            search_tasks.append(('bing', self.bing_search(query)))

        # Execute searches with delays
        for engine_name, search_task in search_tasks:
            try:
                result = await search_task
                results['engines'][engine_name] = result

                # Add to consolidated results
                for res in result.get('results', []):
                    res['found_by'] = engine_name
                    results['consolidated_results'].append(res)

            except Exception as e:
                results['engines'][engine_name] = {
                    'error': str(e),
                    'query': query
                }

        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []

        for result in results['consolidated_results']:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        results['consolidated_results'] = unique_results
        results['total_unique_results'] = len(unique_results)

        return results

    def get_search_disclaimer(self) -> str:
        """Get disclaimer for fallback search usage"""

        return """
🔍 FALLBACK SEARCH DISCLAIMER

This fallback search tool is used when API keys are not available.

IMPORTANT NOTES:
• Direct web scraping with respectful rate limiting (3+ second delays)
• Limited to 10 results per engine to minimize server load
• Results may be blocked by anti-bot measures
• Search engine structures can change, affecting parsing
• For production use, proper API keys are recommended

ETHICAL USAGE:
• Respectful rate limiting implemented
• No aggressive scraping or automation
• Educational and research purposes only
• Complies with robots.txt when possible

For better results and reliability, configure proper API keys:
• Google Custom Search API
• Bing Search API
        """

# Example usage and testing
async def test_fallback_search():
    """Test fallback search functionality"""

    fallback = FallbackSearchTool()

    print("🔍 Testing Fallback Search Tool")
    print("=" * 50)

    # Show disclaimer
    print(fallback.get_search_disclaimer())

    # Test multi-engine search
    print("\n🎯 Testing multi-engine search...")
    result = await fallback.multi_search("Lukáš Janovský Litoměřice")

    print(f"📊 Search completed:")
    print(f"  • Total unique results: {result['total_unique_results']}")

    for engine, engine_result in result['engines'].items():
        if 'error' in engine_result:
            print(f"  • {engine}: ❌ {engine_result['error']}")
        else:
            print(f"  • {engine}: ✅ {len(engine_result.get('results', []))} results")

    # Show first few results
    if result['consolidated_results']:
        print(f"\n🏆 Top {min(3, len(result['consolidated_results']))} results:")
        for i, res in enumerate(result['consolidated_results'][:3]):
            print(f"  {i+1}. {res['title'][:60]}...")
            print(f"     {res['url'][:80]}...")
            print(f"     Found by: {res['found_by']}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_fallback_search())