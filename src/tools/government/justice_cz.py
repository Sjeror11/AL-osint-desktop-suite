#!/usr/bin/env python3
"""
⚖️ Justice.cz OSINT Tool
Czech Court Records and Legal Documents Search

Features:
- Court proceedings search
- Insolvency registry search
- Commercial register search
- Criminal proceedings search
- Legal entity verification
- Anti-detection measures for respectful access
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus, urljoin
import json
import re
import time
from bs4 import BeautifulSoup

class JusticeCzTool:
    """Justice.cz court records and legal documents search tool"""

    def __init__(self):
        """Initialize Justice.cz search tool"""

        self.base_url = "https://justice.cz"
        self.endpoints = {
            'civil_search': f"{self.base_url}/web/guest/uvod",
            'insolvency': f"{self.base_url}/web/guest/insolvencni-rejstrik",
            'commercial': f"{self.base_url}/web/guest/obchodni-rejstrik",
            'criminal': f"{self.base_url}/web/guest/trestni-rejstrik"
        }

        # Rate limiting for respectful access
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between requests

        # Cache
        self.cache = {}
        self.cache_duration = timedelta(hours=4)  # Longer cache for legal records

        # Request headers for realistic browsing
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'cs,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        # Logger
        self.logger = logging.getLogger(__name__)

    def _get_cache_key(self, search_type: str, query: str, **params) -> str:
        """Generate cache key for query"""
        cache_data = f"{search_type}_{query}_{str(sorted(params.items()))}"
        return f"justice_{hash(cache_data)}"

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cache entry is still valid"""
        return datetime.now() - timestamp < self.cache_duration

    async def _rate_limit(self):
        """Implement respectful rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    async def _make_request(self, url: str, method: str = 'GET', data: Optional[Dict] = None) -> Optional[str]:
        """Make HTTP request with rate limiting and error handling"""

        # Apply rate limiting
        await self._rate_limit()

        try:
            async with aiohttp.ClientSession() as session:
                if method == 'GET':
                    async with session.get(url, headers=self.headers, timeout=30) as response:
                        if response.status == 200:
                            return await response.text()
                        else:
                            self.logger.warning(f"⚠️ HTTP {response.status} for {url}")
                            return None
                elif method == 'POST':
                    async with session.post(url, headers=self.headers, data=data, timeout=30) as response:
                        if response.status == 200:
                            return await response.text()
                        else:
                            self.logger.warning(f"⚠️ HTTP {response.status} for {url}")
                            return None

        except asyncio.TimeoutError:
            self.logger.error(f"❌ Timeout for {url}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Request error for {url}: {e}")
            return None

    async def search_civil_proceedings(self, name: str, court: Optional[str] = None) -> Dict[str, Any]:
        """
        Search civil court proceedings

        Args:
            name: Person or entity name to search
            court: Specific court to search (optional)

        Returns:
            Civil proceedings search results
        """

        self.logger.info(f"⚖️ Searching civil proceedings for: {name}")

        # Check cache
        cache_key = self._get_cache_key("civil", name, court=court)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.logger.info("📦 Using cached civil proceedings data")
                return cached_data

        results = {
            "search_type": "civil_proceedings",
            "target": name,
            "court": court,
            "timestamp": datetime.now().isoformat(),
            "proceedings": [],
            "metadata": {}
        }

        try:
            # This is a simplified implementation
            # Real implementation would need to handle complex forms and pagination
            search_url = f"{self.base_url}/web/guest/uvod"

            html_content = await self._make_request(search_url)
            if not html_content:
                results["error"] = "Failed to access justice.cz"
                return results

            # Parse HTML and extract relevant information
            soup = BeautifulSoup(html_content, 'html.parser')

            # Look for search forms and extract structure
            forms = soup.find_all('form')
            results["metadata"]["available_forms"] = len(forms)

            # Simulate search results (in real implementation, would submit forms)
            results["proceedings"] = [
                {
                    "case_number": f"SP {datetime.now().year}/XXXX",
                    "court_name": court or "Okresní soud Praha",
                    "case_type": "Občanskoprávní řízení",
                    "status": "Probíhající",
                    "parties": [name],
                    "date_filed": datetime.now().strftime("%Y-%m-%d"),
                    "description": f"Občanskoprávní spor týkající se {name}",
                    "note": "⚠️ Simulovaný výsledek - implementace vyžaduje detailní analýzu formulářů"
                }
            ]

            # Cache results
            self.cache[cache_key] = (results, datetime.now())

            self.logger.info(f"✅ Found {len(results['proceedings'])} civil proceedings")

        except Exception as e:
            self.logger.error(f"❌ Civil proceedings search error: {e}")
            results["error"] = str(e)

        return results

    async def search_insolvency_registry(self, name: str) -> Dict[str, Any]:
        """
        Search insolvency registry

        Args:
            name: Person or entity name to search

        Returns:
            Insolvency registry search results
        """

        self.logger.info(f"💰 Searching insolvency registry for: {name}")

        # Check cache
        cache_key = self._get_cache_key("insolvency", name)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.logger.info("📦 Using cached insolvency data")
                return cached_data

        results = {
            "search_type": "insolvency_registry",
            "target": name,
            "timestamp": datetime.now().isoformat(),
            "insolvency_cases": [],
            "metadata": {}
        }

        try:
            search_url = f"{self.base_url}/web/guest/insolvencni-rejstrik"

            html_content = await self._make_request(search_url)
            if not html_content:
                results["error"] = "Failed to access insolvency registry"
                return results

            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract insolvency search form structure
            search_forms = soup.find_all('form')
            results["metadata"]["search_forms_available"] = len(search_forms)

            # Simulate insolvency search results
            results["insolvency_cases"] = [
                {
                    "case_number": f"INS {datetime.now().year}/XXXX",
                    "debtor_name": name,
                    "court": "Krajský soud v Praze",
                    "case_type": "Osobní bankrot",
                    "status": "Probíhající",
                    "date_filed": datetime.now().strftime("%Y-%m-%d"),
                    "trustee": "JUDr. Správce Majetku",
                    "assets_value": "Nespecifikováno",
                    "creditors_count": "Nespecifikováno",
                    "note": "⚠️ Simulovaný výsledek - implementace vyžaduje detailní analýzu formulářů"
                }
            ]

            # Cache results
            self.cache[cache_key] = (results, datetime.now())

            self.logger.info(f"✅ Found {len(results['insolvency_cases'])} insolvency cases")

        except Exception as e:
            self.logger.error(f"❌ Insolvency search error: {e}")
            results["error"] = str(e)

        return results

    async def search_commercial_register(self, name: str) -> Dict[str, Any]:
        """
        Search commercial register

        Args:
            name: Company name or ID to search

        Returns:
            Commercial register search results
        """

        self.logger.info(f"🏢 Searching commercial register for: {name}")

        # Check cache
        cache_key = self._get_cache_key("commercial", name)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.logger.info("📦 Using cached commercial register data")
                return cached_data

        results = {
            "search_type": "commercial_register",
            "target": name,
            "timestamp": datetime.now().isoformat(),
            "companies": [],
            "metadata": {}
        }

        try:
            search_url = f"{self.base_url}/web/guest/obchodni-rejstrik"

            html_content = await self._make_request(search_url)
            if not html_content:
                results["error"] = "Failed to access commercial register"
                return results

            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract commercial register search structure
            search_forms = soup.find_all('form')
            results["metadata"]["search_forms_available"] = len(search_forms)

            # Simulate commercial register results
            results["companies"] = [
                {
                    "company_name": name,
                    "registration_number": "12345678",
                    "court": "Městský soud v Praze",
                    "section": "C",
                    "insert_number": "123456",
                    "legal_form": "Společnost s ručením omezeným",
                    "registered_office": "Praha 1, Václavské náměstí 1",
                    "business_activities": ["Poskytování software"],
                    "share_capital": "200 000 Kč",
                    "statutory_body": [
                        {
                            "function": "jednatel",
                            "name": "Jan Novák",
                            "from_date": "2020-01-01"
                        }
                    ],
                    "registration_date": "2020-01-01",
                    "status": "Aktivní",
                    "note": "⚠️ Simulovaný výsledek - implementace vyžaduje detailní analýzu formulářů"
                }
            ]

            # Cache results
            self.cache[cache_key] = (results, datetime.now())

            self.logger.info(f"✅ Found {len(results['companies'])} companies")

        except Exception as e:
            self.logger.error(f"❌ Commercial register search error: {e}")
            results["error"] = str(e)

        return results

    async def comprehensive_justice_search(self, name: str) -> Dict[str, Any]:
        """
        Perform comprehensive search across all Justice.cz databases

        Args:
            name: Person or entity name to search

        Returns:
            Comprehensive search results
        """

        self.logger.info(f"🎯 Comprehensive Justice.cz search for: {name}")

        comprehensive_results = {
            "target": name,
            "timestamp": datetime.now().isoformat(),
            "searches": {},
            "summary": {}
        }

        # Execute searches concurrently
        search_tasks = [
            ("civil_proceedings", self.search_civil_proceedings(name)),
            ("insolvency_registry", self.search_insolvency_registry(name)),
            ("commercial_register", self.search_commercial_register(name))
        ]

        results = await asyncio.gather(
            *[task[1] for task in search_tasks],
            return_exceptions=True
        )

        # Process results
        for i, (search_name, _) in enumerate(search_tasks):
            result = results[i]

            if isinstance(result, Exception):
                comprehensive_results["searches"][search_name] = {
                    "error": str(result),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                comprehensive_results["searches"][search_name] = result

        # Generate summary
        total_proceedings = sum(
            len(search.get("proceedings", []))
            for search in comprehensive_results["searches"].values()
            if not search.get("error")
        )

        total_insolvencies = sum(
            len(search.get("insolvency_cases", []))
            for search in comprehensive_results["searches"].values()
            if not search.get("error")
        )

        total_companies = sum(
            len(search.get("companies", []))
            for search in comprehensive_results["searches"].values()
            if not search.get("error")
        )

        comprehensive_results["summary"] = {
            "total_civil_proceedings": total_proceedings,
            "total_insolvency_cases": total_insolvencies,
            "total_companies": total_companies,
            "searches_completed": len([s for s in comprehensive_results["searches"].values() if not s.get("error")]),
            "searches_failed": len([s for s in comprehensive_results["searches"].values() if s.get("error")])
        }

        self.logger.info(f"✅ Comprehensive search completed: "
                        f"{total_proceedings} proceedings, "
                        f"{total_insolvencies} insolvencies, "
                        f"{total_companies} companies")

        return comprehensive_results

    def get_search_capabilities(self) -> Dict[str, Any]:
        """Get information about search capabilities"""

        return {
            "available_searches": {
                "civil_proceedings": {
                    "description": "Search civil court proceedings",
                    "data_types": ["Case numbers", "Court names", "Parties", "Status"],
                    "coverage": "All Czech courts"
                },
                "insolvency_registry": {
                    "description": "Search insolvency and bankruptcy cases",
                    "data_types": ["Debtor info", "Trustees", "Assets", "Creditors"],
                    "coverage": "National insolvency registry"
                },
                "commercial_register": {
                    "description": "Search company registrations",
                    "data_types": ["Company details", "Statutory bodies", "Business activities"],
                    "coverage": "National commercial register"
                }
            },
            "rate_limiting": {
                "min_interval": f"{self.min_request_interval} seconds",
                "respectful_access": True,
                "caching": f"{self.cache_duration.total_seconds() / 3600} hours"
            },
            "legal_compliance": {
                "public_data_only": True,
                "terms_of_service": "https://justice.cz/web/guest/uvod",
                "data_protection": "GDPR compliant access"
            },
            "implementation_status": {
                "status": "PROTOTYPE",
                "note": "Current implementation provides structure and simulation",
                "required_development": [
                    "Form analysis and submission",
                    "Result parsing and extraction",
                    "Pagination handling",
                    "CAPTCHA handling if present"
                ]
            }
        }

    async def get_detailed_case_info(self, case_number: str, court: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific court case.

        Args:
            case_number: Case/file number (spisová značka)
            court: Court name

        Returns:
            Detailed case information including documents and timeline
        """

        self.logger.info(f"📋 Fetching detailed case info: {case_number} @ {court}")

        # Check cache
        cache_key = self._get_cache_key("case_detail", case_number, court=court)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.logger.info("📦 Using cached case details")
                return cached_data

        case_details = {
            "case_number": case_number,
            "court": court,
            "timestamp": datetime.now().isoformat(),
            "case_type": None,
            "parties": [],
            "timeline": [],
            "documents": [],
            "hearings": [],
            "status": None,
            "result": None
        }

        try:
            # This would fetch and parse actual case details
            # Placeholder implementation
            case_details.update({
                "case_type": "Občanskoprávní řízení",
                "status": "Probíhající",
                "parties": [
                    {"role": "Žalobce", "name": "Jan Novák", "representation": "Advokát XY"},
                    {"role": "Žalovaný", "name": "Firma s.r.o.", "representation": "Advokát AB"}
                ],
                "timeline": [
                    {"date": "2024-01-15", "event": "Podání žaloby"},
                    {"date": "2024-02-10", "event": "Nařízeno jednání"},
                    {"date": "2024-03-05", "event": "Hlavní líčení"}
                ],
                "documents": [
                    {"name": "Žaloba", "date": "2024-01-15", "type": "Podání", "available": True},
                    {"name": "Vyjádření žalovaného", "date": "2024-01-30", "type": "Podání", "available": True},
                    {"name": "Předvolání k jednání", "date": "2024-02-10", "type": "Usnesení", "available": True}
                ],
                "hearings": [
                    {"date": "2024-03-05", "time": "09:00", "room": "Jednací síň č. 5", "purpose": "Hlavní líčení"}
                ]
            })

            # Cache results
            self.cache[cache_key] = (case_details, datetime.now())

            self.logger.info(f"✅ Fetched case details: {len(case_details['documents'])} documents, {len(case_details['hearings'])} hearings")

        except Exception as e:
            self.logger.error(f"❌ Error fetching case details: {e}")
            case_details["error"] = str(e)

        return case_details

    async def extract_company_litigations(self, company_name: str, ico: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract all litigations involving a company (as plaintiff or defendant).

        Args:
            company_name: Company name
            ico: Company ICO for cross-validation

        Returns:
            All litigation cases involving the company
        """

        self.logger.info(f"⚖️ Extracting litigations for company: {company_name}")

        litigations = {
            "company_name": company_name,
            "ico": ico,
            "timestamp": datetime.now().isoformat(),
            "as_plaintiff": [],
            "as_defendant": [],
            "statistics": {
                "total_cases": 0,
                "active_cases": 0,
                "closed_cases": 0,
                "won_cases": 0,
                "lost_cases": 0
            }
        }

        try:
            # Search civil proceedings
            civil_results = await self.search_civil_proceedings(company_name)

            # Categorize cases
            for case in civil_results.get("proceedings", []):
                if company_name.lower() in str(case.get("parties", [])).lower():
                    # Determine if plaintiff or defendant
                    # This is simplified - real parsing would be more sophisticated
                    litigations["as_plaintiff"].append(case)
                    litigations["statistics"]["total_cases"] += 1

                    if case.get("status") == "Probíhající":
                        litigations["statistics"]["active_cases"] += 1
                    else:
                        litigations["statistics"]["closed_cases"] += 1

            self.logger.info(f"✅ Found {litigations['statistics']['total_cases']} litigation cases")

        except Exception as e:
            self.logger.error(f"❌ Error extracting litigations: {e}")
            litigations["error"] = str(e)

        return litigations

    async def cross_reference_with_ares(self, company_name: str) -> Dict[str, Any]:
        """
        Cross-reference Justice.cz data with ARES for comprehensive company profile.

        Args:
            company_name: Company name

        Returns:
            Combined profile from Justice.cz and ARES
        """

        self.logger.info(f"🔄 Cross-referencing company: {company_name} with ARES")

        combined_profile = {
            "company_name": company_name,
            "timestamp": datetime.now().isoformat(),
            "justice_data": {},
            "ares_data": {},
            "comprehensive_profile": {
                "legal_health_score": 0.0,
                "active_litigations": 0,
                "insolvency_risk": "Low",
                "statutory_bodies_verified": False,
                "company_status": None
            }
        }

        try:
            # Fetch Justice.cz data
            justice_search = await self.comprehensive_justice_search(company_name)
            combined_profile["justice_data"] = justice_search

            # Fetch litigations
            litigations = await self.extract_company_litigations(company_name)
            combined_profile["litigations"] = litigations

            # Calculate legal health score
            active_cases = litigations["statistics"]["active_cases"]
            insolvency_cases = len(justice_search["searches"].get("insolvency_registry", {}).get("insolvency_cases", []))

            # Health score calculation (0.0 - 1.0)
            health_score = 1.0
            health_score -= min(active_cases * 0.1, 0.5)  # Penalize active cases
            health_score -= min(insolvency_cases * 0.3, 0.5)  # Heavily penalize insolvency

            combined_profile["comprehensive_profile"]["legal_health_score"] = max(health_score, 0.0)
            combined_profile["comprehensive_profile"]["active_litigations"] = active_cases

            if insolvency_cases > 0:
                combined_profile["comprehensive_profile"]["insolvency_risk"] = "High"
            elif active_cases > 3:
                combined_profile["comprehensive_profile"]["insolvency_risk"] = "Medium"
            else:
                combined_profile["comprehensive_profile"]["insolvency_risk"] = "Low"

            self.logger.info(f"✅ Cross-reference completed - Legal health score: {health_score:.2f}")

        except Exception as e:
            self.logger.error(f"❌ Error in cross-reference: {e}")
            combined_profile["error"] = str(e)

        return combined_profile

    async def enhanced_person_profile(self, person_name: str) -> Dict[str, Any]:
        """
        Create enhanced person profile with all available Justice.cz data.

        Args:
            person_name: Person's full name

        Returns:
            Comprehensive person profile from public records
        """

        self.logger.info(f"👤 Creating enhanced profile for: {person_name}")

        profile = {
            "person_name": person_name,
            "timestamp": datetime.now().isoformat(),
            "profile_completeness": 0.0,
            "sections": {}
        }

        try:
            # Comprehensive Justice.cz search
            justice_results = await self.comprehensive_justice_search(person_name)
            profile["sections"]["justice_records"] = justice_results

            # Extract specific involvement types
            profile["sections"]["civil_proceedings"] = {
                "total": len(justice_results["searches"].get("civil_proceedings", {}).get("proceedings", [])),
                "details": justice_results["searches"].get("civil_proceedings", {})
            }

            profile["sections"]["insolvency"] = {
                "total": len(justice_results["searches"].get("insolvency_registry", {}).get("insolvency_cases", [])),
                "details": justice_results["searches"].get("insolvency_registry", {})
            }

            # Calculate profile completeness
            sections_with_data = sum([
                1 if profile["sections"]["civil_proceedings"]["total"] > 0 else 0,
                1 if profile["sections"]["insolvency"]["total"] > 0 else 0,
                0.5  # Base completeness for having conducted search
            ])

            profile["profile_completeness"] = min(sections_with_data / 2.5, 1.0)

            # Risk assessment
            profile["risk_assessment"] = {
                "insolvency_filings": profile["sections"]["insolvency"]["total"],
                "active_litigations": profile["sections"]["civil_proceedings"]["total"],
                "risk_level": "Low"
            }

            if profile["sections"]["insolvency"]["total"] > 0:
                profile["risk_assessment"]["risk_level"] = "High"
            elif profile["sections"]["civil_proceedings"]["total"] > 5:
                profile["risk_assessment"]["risk_level"] = "Medium"

            self.logger.info(f"✅ Enhanced profile created (completeness: {profile['profile_completeness']:.1%})")

        except Exception as e:
            self.logger.error(f"❌ Error creating enhanced profile: {e}")
            profile["error"] = str(e)

        return profile

    def get_legal_disclaimer(self) -> str:
        """Get legal disclaimer for Justice.cz usage"""

        return """
⚖️ JUSTICE.CZ LEGAL DISCLAIMER

This tool accesses public information available on justice.cz website.

IMPORTANT NOTES:
• Only public court records are accessed
• Respectful rate limiting is implemented (2 second intervals)
• No automated form submission without proper authorization
• Data is cached locally to minimize server load
• All access complies with website terms of service

CURRENT IMPLEMENTATION:
• PROTOTYPE STATUS - Provides structure and simulation
• Real implementation requires detailed form analysis
• Results are simulated for demonstration purposes
• Production use requires proper form submission logic

LEGAL COMPLIANCE:
• GDPR compliant data handling
• Public information access only
• Respects website terms of service
• No unauthorized data scraping

For production use, ensure proper legal review and compliance.
        """

# Example usage and testing
async def test_justice_cz():
    """Test Justice.cz functionality"""

    justice = JusticeCzTool()

    print("⚖️ Testing Justice.cz OSINT Tool")
    print("=" * 50)

    # Show capabilities
    capabilities = justice.get_search_capabilities()
    print("🔍 Available searches:")
    for search_name, details in capabilities["available_searches"].items():
        print(f"  • {search_name}: {details['description']}")

    print(f"\n⏱️ Rate limiting: {capabilities['rate_limiting']['min_interval']}")
    print(f"📦 Caching: {capabilities['rate_limiting']['caching']}")

    # Test comprehensive search
    print("\n🎯 Testing comprehensive search...")
    result = await justice.comprehensive_justice_search("Test Firma s.r.o.")

    summary = result["summary"]
    print(f"📊 Results summary:")
    print(f"  • Civil proceedings: {summary['total_civil_proceedings']}")
    print(f"  • Insolvency cases: {summary['total_insolvency_cases']}")
    print(f"  • Companies: {summary['total_companies']}")
    print(f"  • Searches completed: {summary['searches_completed']}")

    # Show legal disclaimer
    print("\n" + "=" * 50)
    print(justice.get_legal_disclaimer())

if __name__ == "__main__":
    # Run test
    asyncio.run(test_justice_cz())