#!/usr/bin/env python3
"""
🎯 Search Orchestrator - Multi-Engine Search Coordination
OSINT Desktop Suite - Web Search Tools

Features:
- Coordinates Google and Bing search engines
- Cross-engine result correlation and deduplication
- Confidence scoring and result ranking
- Comprehensive search strategy execution
- AI-enhanced result analysis
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from urllib.parse import urlparse
import json
import os

# Import search engines
from .google_search import GoogleSearchTool
from .bing_search import BingSearchTool
from .fallback_search import FallbackSearchTool

# Import Czech government databases
try:
    from ..government.ares_cz import AresCzTool
    from ..government.justice_cz import JusticeCzTool
    CZECH_TOOLS_AVAILABLE = True
except ImportError:
    CZECH_TOOLS_AVAILABLE = False

class SearchOrchestrator:
    """Orchestrates multiple search engines for comprehensive OSINT investigations"""

    def __init__(self):
        """Initialize search orchestrator with available engines"""

        self.logger = logging.getLogger(__name__)

        # Initialize search engines
        self.google = GoogleSearchTool()
        self.bing = BingSearchTool()
        self.fallback = FallbackSearchTool()

        # Initialize Czech government databases
        self.ares = None
        self.justice = None

        if CZECH_TOOLS_AVAILABLE:
            self.ares = AresCzTool()
            self.justice = JusticeCzTool()

        # Track which engines are available
        self.available_engines = []
        self.available_databases = []
        self.fallback_available = True

        if self.google.api_key:
            self.available_engines.append('google')
            self.logger.info("✅ Google Search API available")
        else:
            self.logger.warning("⚠️ Google Search API key not found")

        if self.bing.api_key:
            self.available_engines.append('bing')
            self.logger.info("✅ Bing Search API available")
        else:
            self.logger.warning("⚠️ Bing Search API key not found")

        # Always add fallback if no API engines available
        if not self.available_engines and self.fallback_available:
            self.available_engines.append('fallback')
            self.logger.info("✅ Fallback web search available (no API keys required)")

        if CZECH_TOOLS_AVAILABLE:
            self.available_databases.extend(['ares', 'justice'])
            self.logger.info("✅ Czech government databases available")

        if not self.available_engines and not self.available_databases:
            self.logger.error("❌ No search engines or databases available")

        # Result correlation settings
        self.similarity_threshold = 0.8
        self.max_results_per_engine = 20

    async def comprehensive_investigation(self, target_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive multi-engine OSINT investigation

        Args:
            target_name: Target name or entity to investigate

        Returns:
            Comprehensive investigation results with cross-engine correlation
        """

        self.logger.info(f"🎯 Starting comprehensive investigation for: {target_name}")

        if not self.available_engines and not self.available_databases:
            return {
                "error": "No search engines or databases available",
                "target": target_name,
                "timestamp": datetime.now().isoformat()
            }

        investigation_results = {
            "target": target_name,
            "timestamp": datetime.now().isoformat(),
            "engines_used": self.available_engines.copy(),
            "raw_results": {},
            "correlated_results": {},
            "summary": {},
            "confidence_scores": {}
        }

        # Execute searches on all available engines and databases
        search_tasks = []

        if 'google' in self.available_engines:
            search_tasks.append(self._google_comprehensive_search(target_name))

        if 'bing' in self.available_engines:
            search_tasks.append(self._bing_comprehensive_search(target_name))

        if 'fallback' in self.available_engines:
            search_tasks.append(self._fallback_comprehensive_search(target_name))

        if 'ares' in self.available_databases:
            search_tasks.append(self._ares_comprehensive_search(target_name))

        if 'justice' in self.available_databases:
            search_tasks.append(self._justice_comprehensive_search(target_name))

        # Wait for all searches to complete
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results from each engine and database
        all_sources = self.available_engines + self.available_databases

        for i, result in enumerate(search_results):
            source_name = all_sources[i]

            if isinstance(result, Exception):
                self.logger.error(f"❌ {source_name} search failed: {result}")
                investigation_results["raw_results"][source_name] = {"error": str(result)}
            else:
                investigation_results["raw_results"][source_name] = result
                self.logger.info(f"✅ {source_name} search completed")

        # Correlate and deduplicate results
        investigation_results["correlated_results"] = self._correlate_results(
            investigation_results["raw_results"]
        )

        # Generate summary and confidence scores
        investigation_results["summary"] = self._generate_summary(
            investigation_results["correlated_results"]
        )

        investigation_results["confidence_scores"] = self._calculate_confidence_scores(
            investigation_results["correlated_results"]
        )

        self.logger.info(f"✅ Comprehensive investigation completed for: {target_name}")

        return investigation_results

    async def _google_comprehensive_search(self, target_name: str) -> Dict[str, Any]:
        """Execute comprehensive Google search"""

        try:
            # Use Google's advanced search functionality
            return await self.google.advanced_search(target_name)
        except Exception as e:
            self.logger.error(f"❌ Google comprehensive search failed: {e}")
            return {"error": str(e), "results": []}

    async def _bing_comprehensive_search(self, target_name: str) -> Dict[str, Any]:
        """Execute comprehensive Bing search"""

        try:
            # Use Bing's comprehensive search functionality
            return await self.bing.comprehensive_search(target_name)
        except Exception as e:
            self.logger.error(f"❌ Bing comprehensive search failed: {e}")
            return {"error": str(e), "results": []}

    async def _ares_comprehensive_search(self, target_name: str) -> Dict[str, Any]:
        """Execute comprehensive ARES business registry search"""

        try:
            # Use ARES comprehensive search functionality
            return await self.ares.comprehensive_business_search(target_name)
        except Exception as e:
            self.logger.error(f"❌ ARES comprehensive search failed: {e}")
            return {"error": str(e), "results": []}

    async def _justice_comprehensive_search(self, target_name: str) -> Dict[str, Any]:
        """Execute comprehensive Justice.cz search"""

        try:
            # Use Justice comprehensive search functionality
            return await self.justice.comprehensive_justice_search(target_name)
        except Exception as e:
            self.logger.error(f"❌ Justice comprehensive search failed: {e}")
            return {"error": str(e), "results": []}

    async def _fallback_comprehensive_search(self, target_name: str) -> Dict[str, Any]:
        """Execute comprehensive fallback web search"""

        try:
            # Use fallback multi-search functionality
            result = await self.fallback.multi_search(target_name)

            # Convert to format compatible with other engines
            formatted_result = {
                "target": target_name,
                "timestamp": result.get("timestamp"),
                "searches": {
                    "fallback_web": {
                        "results": result.get("consolidated_results", []),
                        "total_found": result.get("total_unique_results", 0),
                        "engines_used": list(result.get("engines", {}).keys()),
                        "note": "Fallback web scraping - no API keys required"
                    }
                },
                "summary": {
                    "total_web_results": result.get("total_unique_results", 0),
                    "engines_successful": len([e for e in result.get("engines", {}).values() if "error" not in e]),
                    "engines_used": list(result.get("engines", {}).keys())
                }
            }

            return formatted_result

        except Exception as e:
            self.logger.error(f"❌ Fallback comprehensive search failed: {e}")
            return {"error": str(e), "results": []}

    def _correlate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correlate and deduplicate results across search engines

        Args:
            raw_results: Raw results from each search engine

        Returns:
            Correlated and deduplicated results
        """

        correlated = {
            "unique_urls": {},
            "duplicate_urls": {},
            "domain_coverage": {},
            "content_types": {},
            "consolidated_results": []
        }

        all_results = []
        url_sources = {}

        # Collect all results and track sources
        for engine, engine_results in raw_results.items():
            if engine_results.get("error"):
                continue

            # Handle different result structures
            results_list = []

            if engine == "google":
                # Google advanced search structure
                for query_data in engine_results.get("queries", []):
                    results_list.extend(query_data.get("results", []))

            elif engine == "bing":
                # Bing comprehensive search structure
                for search_type, search_data in engine_results.get("searches", {}).items():
                    results_list.extend(search_data.get("results", []))

            elif engine == "fallback":
                # Fallback search structure
                for search_type, search_data in engine_results.get("searches", {}).items():
                    results_list.extend(search_data.get("results", []))

            # Process each result
            for result in results_list:
                url = result.get("url") or result.get("link", "")
                if url:
                    # Track which engines found this URL
                    if url not in url_sources:
                        url_sources[url] = []
                    url_sources[url].append(engine)

                    # Add engine source to result
                    result["source_engine"] = engine
                    all_results.append(result)

        # Categorize URLs by uniqueness
        for url, sources in url_sources.items():
            if len(sources) == 1:
                correlated["unique_urls"][url] = sources[0]
            else:
                correlated["duplicate_urls"][url] = sources

        # Analyze domain coverage
        domains = {}
        for result in all_results:
            url = result.get("url") or result.get("link", "")
            if url:
                try:
                    domain = urlparse(url).netloc
                    if domain:
                        if domain not in domains:
                            domains[domain] = {
                                "count": 0,
                                "engines": set(),
                                "results": []
                            }
                        domains[domain]["count"] += 1
                        domains[domain]["engines"].add(result["source_engine"])
                        domains[domain]["results"].append(result)
                except:
                    continue

        # Convert sets to lists for JSON serialization
        for domain, data in domains.items():
            data["engines"] = list(data["engines"])

        correlated["domain_coverage"] = domains

        # Deduplicate and consolidate results
        seen_urls = set()
        for result in all_results:
            url = result.get("url") or result.get("link", "")
            if url and url not in seen_urls:
                seen_urls.add(url)

                # Enhance result with correlation data
                result["found_by_engines"] = url_sources.get(url, [])
                result["confidence_boost"] = len(url_sources.get(url, [])) > 1

                correlated["consolidated_results"].append(result)

        # Sort by relevance (multi-engine results first, then by source)
        correlated["consolidated_results"].sort(
            key=lambda x: (
                -len(x.get("found_by_engines", [])),  # Multi-engine results first
                x.get("title", "").lower()  # Then by title
            )
        )

        return correlated

    def _generate_summary(self, correlated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate investigation summary"""

        summary = {
            "total_unique_results": len(correlated_results.get("consolidated_results", [])),
            "unique_domains": len(correlated_results.get("domain_coverage", {})),
            "multi_engine_confirmations": len(correlated_results.get("duplicate_urls", {})),
            "single_engine_discoveries": len(correlated_results.get("unique_urls", {})),
            "top_domains": [],
            "content_type_distribution": {},
            "source_reliability": {}
        }

        # Top domains by result count
        domain_coverage = correlated_results.get("domain_coverage", {})
        top_domains = sorted(
            domain_coverage.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:10]

        summary["top_domains"] = [
            {
                "domain": domain,
                "result_count": data["count"],
                "engines": data["engines"]
            }
            for domain, data in top_domains
        ]

        # Content type analysis
        content_types = {}
        for result in correlated_results.get("consolidated_results", []):
            # Try to determine content type from URL or other indicators
            url = result.get("url", "")

            if ".pdf" in url.lower():
                content_type = "PDF"
            elif any(social in url.lower() for social in ["facebook", "linkedin", "twitter", "instagram"]):
                content_type = "Social Media"
            elif any(news in url.lower() for news in ["novinky", "idnes", "aktualne", "denik"]):
                content_type = "News"
            elif "justice.cz" in url.lower():
                content_type = "Legal Records"
            elif "ares.gov.cz" in url.lower():
                content_type = "Business Registry"
            else:
                content_type = "Web Page"

            content_types[content_type] = content_types.get(content_type, 0) + 1

        summary["content_type_distribution"] = content_types

        # Source reliability (based on multi-engine confirmation)
        confirmed_results = sum(
            1 for result in correlated_results.get("consolidated_results", [])
            if len(result.get("found_by_engines", [])) > 1
        )

        total_results = len(correlated_results.get("consolidated_results", []))

        if total_results > 0:
            summary["source_reliability"] = {
                "multi_engine_confirmation_rate": confirmed_results / total_results,
                "high_confidence_results": confirmed_results,
                "total_results": total_results
            }

        return summary

    def _calculate_confidence_scores(self, correlated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence scores for results"""

        confidence_scores = {
            "methodology": "Multi-engine confirmation + domain authority + content type",
            "results": []
        }

        for result in correlated_results.get("consolidated_results", []):
            score = 0.5  # Base score
            factors = []

            # Multi-engine confirmation boost
            engines_found = len(result.get("found_by_engines", []))
            if engines_found > 1:
                score += 0.3
                factors.append(f"Found by {engines_found} engines (+0.3)")

            # Domain authority boost
            url = result.get("url", "")
            domain = urlparse(url).netloc if url else ""

            # High-authority domains
            high_authority_domains = [
                "justice.cz", "ares.gov.cz", "firmy.cz",
                "linkedin.com", "facebook.com",
                "idnes.cz", "novinky.cz", "aktualne.cz"
            ]

            if any(auth_domain in domain for auth_domain in high_authority_domains):
                score += 0.2
                factors.append("High-authority domain (+0.2)")

            # Content type relevance
            if "PDF" in url or "document" in result.get("title", "").lower():
                score += 0.1
                factors.append("Document content (+0.1)")

            # Title/snippet quality
            title = result.get("title", "")
            snippet = result.get("snippet", "") or result.get("description", "")

            if title and len(title) > 10:
                score += 0.05
                factors.append("Quality title (+0.05)")

            if snippet and len(snippet) > 20:
                score += 0.05
                factors.append("Quality description (+0.05)")

            # Cap at 1.0
            score = min(score, 1.0)

            confidence_scores["results"].append({
                "url": url,
                "title": title,
                "confidence_score": round(score, 3),
                "contributing_factors": factors,
                "engines": result.get("found_by_engines", [])
            })

        # Sort by confidence score
        confidence_scores["results"].sort(
            key=lambda x: x["confidence_score"],
            reverse=True
        )

        return confidence_scores

    async def quick_search(self, query: str, max_results: int = 20) -> Dict[str, Any]:
        """
        Perform quick search across available engines

        Args:
            query: Search query
            max_results: Maximum results per engine

        Returns:
            Quick search results
        """

        self.logger.info(f"⚡ Quick search: {query}")

        if not self.available_engines:
            return {"error": "No search engines available"}

        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "engines": {}
        }

        search_tasks = []

        if 'google' in self.available_engines:
            search_tasks.append(
                ("google", self.google.search(query, num_results=max_results))
            )

        if 'bing' in self.available_engines:
            search_tasks.append(
                ("bing", self.bing.web_search(query, count=max_results))
            )

        # Execute searches concurrently
        engine_results = await asyncio.gather(
            *[task[1] for task in search_tasks],
            return_exceptions=True
        )

        # Process results
        for i, (engine_name, _) in enumerate(search_tasks):
            result = engine_results[i]

            if isinstance(result, Exception):
                results["engines"][engine_name] = {"error": str(result)}
            else:
                results["engines"][engine_name] = result

        return results

    async def specialized_czech_search(self, target_name: str) -> Dict[str, Any]:
        """
        Specialized search focused on Czech OSINT sources

        Args:
            target_name: Target name to investigate

        Returns:
            Czech-focused search results
        """

        self.logger.info(f"🇨🇿 Czech specialized search for: {target_name}")

        results = {
            "target": target_name,
            "timestamp": datetime.now().isoformat(),
            "czech_sources": {}
        }

        search_tasks = []

        if 'bing' in self.available_engines:
            search_tasks.append(
                ("bing", self.bing.czech_osint_search(target_name))
            )

        if 'google' in self.available_engines:
            # Google Czech-specific searches
            czech_queries = [
                f'{target_name} site:justice.cz',
                f'{target_name} site:ares.gov.cz',
                f'{target_name} site:firmy.cz'
            ]

            for query in czech_queries:
                search_tasks.append(
                    (f"google_{query.split('site:')[1]}", self.google.search(query, num_results=10))
                )

        # Execute searches
        engine_results = await asyncio.gather(
            *[task[1] for task in search_tasks],
            return_exceptions=True
        )

        # Process results
        for i, (engine_name, _) in enumerate(search_tasks):
            result = engine_results[i]

            if isinstance(result, Exception):
                results["czech_sources"][engine_name] = {"error": str(result)}
            else:
                results["czech_sources"][engine_name] = result

        return results

    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all search engines and databases"""

        return {
            "available_engines": self.available_engines,
            "available_databases": self.available_databases,
            "google": {
                "available": 'google' in self.available_engines,
                "api_key_configured": bool(self.google.api_key)
            },
            "bing": {
                "available": 'bing' in self.available_engines,
                "api_key_configured": bool(self.bing.api_key)
            },
            "fallback": {
                "available": 'fallback' in self.available_engines,
                "description": "Web scraping fallback (no API keys required)"
            },
            "ares": {
                "available": 'ares' in self.available_databases,
                "description": "Czech business registry (IČO/DIČ)"
            },
            "justice": {
                "available": 'justice' in self.available_databases,
                "description": "Czech court records and legal proceedings"
            },
            "total_engines": len(self.available_engines),
            "total_databases": len(self.available_databases),
            "total_sources": len(self.available_engines) + len(self.available_databases)
        }

# Example usage and testing
async def test_search_orchestrator():
    """Test search orchestrator functionality"""

    orchestrator = SearchOrchestrator()

    print("🔍 Search Engine Status:")
    status = orchestrator.get_engine_status()
    print(f"Available engines: {status['available_engines']}")
    print(f"Total engines: {status['total_engines']}")

    if not status['available_engines']:
        print("⚠️ No search engines available - check API keys")
        return

    # Test quick search
    print("\n⚡ Testing quick search...")
    quick_result = await orchestrator.quick_search("OSINT nástroje", max_results=5)

    for engine, engine_results in quick_result["engines"].items():
        if "error" in engine_results:
            print(f"❌ {engine}: {engine_results['error']}")
        else:
            print(f"✅ {engine}: {len(engine_results.get('results', []))} results")

    # Test comprehensive investigation
    if len(status['available_engines']) > 1:
        print("\n🎯 Testing comprehensive investigation...")
        comp_result = await orchestrator.comprehensive_investigation("Jan Novák")

        print(f"📊 Total unique results: {comp_result['summary']['total_unique_results']}")
        print(f"📊 Unique domains: {comp_result['summary']['unique_domains']}")
        print(f"📊 Multi-engine confirmations: {comp_result['summary']['multi_engine_confirmations']}")

        # Show top confidence results
        top_results = comp_result["confidence_scores"]["results"][:3]
        print(f"\n🏆 Top {len(top_results)} confidence results:")
        for result in top_results:
            print(f"  {result['confidence_score']:.3f} - {result['title'][:50]}...")

    # Test Czech search
    print("\n🇨🇿 Testing Czech specialized search...")
    czech_result = await orchestrator.specialized_czech_search("Test Firma")
    print(f"📊 Czech sources searched: {len(czech_result['czech_sources'])}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_search_orchestrator())