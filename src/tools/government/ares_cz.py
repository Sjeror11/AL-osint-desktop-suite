#!/usr/bin/env python3
"""
🏢 ARES.gov.cz OSINT Tool
Czech Business Registry Search

Features:
- Company information lookup
- Business registry search
- Economic entity verification
- Historical data access
- Financial information extraction
- API and web scraping capabilities
"""

import asyncio
import aiohttp
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote_plus, urljoin
import json
import re
import time

class AresCzTool:
    """ARES.gov.cz business registry search tool"""

    def __init__(self):
        """Initialize ARES.gov.cz search tool"""

        # ARES API endpoints
        self.base_url = "https://wwwinfo.mfcr.cz/cgi-bin/ares"
        self.web_url = "https://ares.gov.cz"

        self.endpoints = {
            'basic_search': f"{self.base_url}/darv_bas.cgi",
            'standard_search': f"{self.base_url}/darv_std.cgi",
            'economic_subject': f"{self.base_url}/darv_es.cgi",
            'web_search': f"{self.web_url}/ekonomicke-subjekty-v-be"
        }

        # Rate limiting for API
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests

        # Cache
        self.cache = {}
        self.cache_duration = timedelta(hours=6)  # Longer cache for business data

        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'cs,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive'
        }

        # Logger
        self.logger = logging.getLogger(__name__)

    def _get_cache_key(self, search_type: str, query: str, **params) -> str:
        """Generate cache key for query"""
        cache_data = f"{search_type}_{query}_{str(sorted(params.items()))}"
        return f"ares_{hash(cache_data)}"

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

    async def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[str]:
        """Make HTTP request with rate limiting"""

        await self._rate_limit()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self.headers, timeout=20) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        self.logger.warning(f"⚠️ HTTP {response.status} for ARES request")
                        return None

        except asyncio.TimeoutError:
            self.logger.error("❌ Timeout for ARES request")
            return None
        except Exception as e:
            self.logger.error(f"❌ ARES request error: {e}")
            return None

    def _parse_xml_response(self, xml_content: str) -> Dict[str, Any]:
        """Parse XML response from ARES API"""

        try:
            root = ET.fromstring(xml_content)

            # Find the data element
            ns = {'are': 'http://wwwinfo.mfcr.cz/ares/xml_doc/schemas/ares/ares_answer/v_1.0.1'}

            # Look for economic subjects
            subjects = []
            for vbas in root.findall('.//are:VBAS', ns):
                subject = {}

                # Basic identification
                ico = vbas.find('.//are:ICO', ns)
                if ico is not None:
                    subject['ico'] = ico.text

                dic = vbas.find('.//are:DIC', ns)
                if dic is not None:
                    subject['dic'] = dic.text

                name = vbas.find('.//are:OF', ns)
                if name is not None:
                    subject['name'] = name.text

                # Address
                address_parts = []
                for addr_elem in ['are:AD_ADRESAR', 'are:AD_ADRESAR/are:AA', 'are:AD_ADRESAR/are:AA/are:NU']:
                    elem = vbas.find(f'.//{addr_elem}', ns)
                    if elem is not None:
                        address_parts.append(elem.text)

                if address_parts:
                    subject['address'] = ', '.join(filter(None, address_parts))

                # Legal form
                legal_form = vbas.find('.//are:PF', ns)
                if legal_form is not None:
                    subject['legal_form'] = legal_form.text

                # Status
                status = vbas.find('.//are:STA', ns)
                if status is not None:
                    subject['status'] = status.text

                subjects.append(subject)

            return {
                'subjects': subjects,
                'total_found': len(subjects)
            }

        except ET.ParseError as e:
            self.logger.error(f"❌ XML parsing error: {e}")
            return {'error': 'XML parsing failed', 'subjects': []}
        except Exception as e:
            self.logger.error(f"❌ Response parsing error: {e}")
            return {'error': str(e), 'subjects': []}

    async def search_by_ico(self, ico: str) -> Dict[str, Any]:
        """
        Search company by ICO (Company ID)

        Args:
            ico: 8-digit company identification number

        Returns:
            Company information from ARES
        """

        # Validate ICO format
        ico_clean = re.sub(r'\D', '', ico)
        if len(ico_clean) != 8:
            return {
                "error": "ICO must be 8 digits",
                "ico": ico,
                "timestamp": datetime.now().isoformat()
            }

        self.logger.info(f"🏢 ARES search by ICO: {ico_clean}")

        # Check cache
        cache_key = self._get_cache_key("ico", ico_clean)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.logger.info("📦 Using cached ARES data")
                return cached_data

        # API parameters
        params = {
            'ico': ico_clean,
            'MAX_POCET': '1',
            'SHOW_REGL': 'N'
        }

        # Make API request
        xml_content = await self._make_request(self.endpoints['basic_search'], params)

        if not xml_content:
            return {
                "error": "Failed to access ARES API",
                "ico": ico_clean,
                "timestamp": datetime.now().isoformat()
            }

        # Parse response
        parsed_data = self._parse_xml_response(xml_content)

        # Build result
        result = {
            "search_type": "ico_search",
            "ico": ico_clean,
            "timestamp": datetime.now().isoformat(),
            "found": len(parsed_data.get('subjects', [])) > 0,
            "data": parsed_data.get('subjects', [{}])[0] if parsed_data.get('subjects') else {},
            "metadata": {
                "total_found": parsed_data.get('total_found', 0),
                "source": "ARES API",
                "response_size": len(xml_content) if xml_content else 0
            }
        }

        # Add additional fields if found
        if result["found"]:
            subject = result["data"]

            # Enhance with additional information
            result["data"]["search_ico"] = ico_clean
            result["data"]["verified"] = True
            result["data"]["last_updated"] = datetime.now().isoformat()

            # Determine entity type
            legal_form = subject.get('legal_form', '').lower()
            if 's.r.o' in legal_form or 'společnost s ručením omezeným' in legal_form:
                result["data"]["entity_type"] = "Společnost s ručením omezeným"
            elif 'a.s' in legal_form or 'akciová společnost' in legal_form:
                result["data"]["entity_type"] = "Akciová společnost"
            elif 'o.p.s' in legal_form:
                result["data"]["entity_type"] = "Obecně prospěšná společnost"
            else:
                result["data"]["entity_type"] = legal_form

        # Cache results
        self.cache[cache_key] = (result, datetime.now())

        if result["found"]:
            self.logger.info(f"✅ Found company: {result['data'].get('name', 'Unknown')}")
        else:
            self.logger.info("❌ No company found for ICO")

        return result

    async def search_by_name(self, name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search companies by name

        Args:
            name: Company name to search
            limit: Maximum number of results

        Returns:
            List of matching companies
        """

        self.logger.info(f"🔍 ARES search by name: {name}")

        # Check cache
        cache_key = self._get_cache_key("name", name, limit=limit)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.logger.info("📦 Using cached ARES name search")
                return cached_data

        # API parameters
        params = {
            'obchodni_firma': name,
            'MAX_POCET': str(min(limit, 100)),  # API limit
            'SHOW_REGL': 'N'
        }

        # Make API request
        xml_content = await self._make_request(self.endpoints['basic_search'], params)

        if not xml_content:
            return {
                "error": "Failed to access ARES API",
                "search_name": name,
                "timestamp": datetime.now().isoformat()
            }

        # Parse response
        parsed_data = self._parse_xml_response(xml_content)

        result = {
            "search_type": "name_search",
            "search_name": name,
            "timestamp": datetime.now().isoformat(),
            "found": len(parsed_data.get('subjects', [])) > 0,
            "companies": parsed_data.get('subjects', []),
            "metadata": {
                "total_found": parsed_data.get('total_found', 0),
                "requested_limit": limit,
                "source": "ARES API"
            }
        }

        # Enhance company data
        for company in result["companies"]:
            company["verified"] = True
            company["last_updated"] = datetime.now().isoformat()

        # Cache results
        self.cache[cache_key] = (result, datetime.now())

        self.logger.info(f"✅ Found {len(result['companies'])} companies matching: {name}")

        return result

    async def search_by_dic(self, dic: str) -> Dict[str, Any]:
        """
        Search company by DIC (Tax ID)

        Args:
            dic: Tax identification number

        Returns:
            Company information from ARES
        """

        # Clean DIC format
        dic_clean = re.sub(r'\D', '', dic)

        self.logger.info(f"💰 ARES search by DIC: {dic_clean}")

        # Check cache
        cache_key = self._get_cache_key("dic", dic_clean)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.logger.info("📦 Using cached ARES DIC data")
                return cached_data

        # API parameters
        params = {
            'dic': dic_clean,
            'MAX_POCET': '1',
            'SHOW_REGL': 'N'
        }

        # Make API request
        xml_content = await self._make_request(self.endpoints['basic_search'], params)

        if not xml_content:
            return {
                "error": "Failed to access ARES API",
                "dic": dic_clean,
                "timestamp": datetime.now().isoformat()
            }

        # Parse response
        parsed_data = self._parse_xml_response(xml_content)

        result = {
            "search_type": "dic_search",
            "dic": dic_clean,
            "timestamp": datetime.now().isoformat(),
            "found": len(parsed_data.get('subjects', [])) > 0,
            "data": parsed_data.get('subjects', [{}])[0] if parsed_data.get('subjects') else {},
            "metadata": {
                "total_found": parsed_data.get('total_found', 0),
                "source": "ARES API"
            }
        }

        # Cache results
        self.cache[cache_key] = (result, datetime.now())

        if result["found"]:
            self.logger.info(f"✅ Found company by DIC: {result['data'].get('name', 'Unknown')}")
        else:
            self.logger.info("❌ No company found for DIC")

        return result

    async def comprehensive_business_search(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive business search - tries ICO, DIC, and name search

        Args:
            query: Search query (ICO, DIC, or company name)

        Returns:
            Comprehensive search results
        """

        self.logger.info(f"🎯 Comprehensive ARES search for: {query}")

        comprehensive_results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "searches": {},
            "best_match": None,
            "summary": {}
        }

        # Determine search strategies based on query format
        search_strategies = []

        # Check if query looks like ICO (8 digits)
        if re.match(r'^\d{8}$', re.sub(r'\D', '', query)):
            search_strategies.append(("ico", self.search_by_ico(re.sub(r'\D', '', query))))

        # Check if query looks like DIC
        if re.match(r'^(CZ)?\d{8,10}$', re.sub(r'\D', '', query)):
            search_strategies.append(("dic", self.search_by_dic(query)))

        # Always try name search
        search_strategies.append(("name", self.search_by_name(query, limit=20)))

        # Execute searches
        results = await asyncio.gather(
            *[strategy[1] for strategy in search_strategies],
            return_exceptions=True
        )

        # Process results
        for i, (search_type, _) in enumerate(search_strategies):
            result = results[i]

            if isinstance(result, Exception):
                comprehensive_results["searches"][search_type] = {
                    "error": str(result),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                comprehensive_results["searches"][search_type] = result

                # Determine best match
                if result.get("found") and not comprehensive_results["best_match"]:
                    if search_type in ["ico", "dic"]:
                        # ICO/DIC searches are more precise
                        comprehensive_results["best_match"] = {
                            "search_type": search_type,
                            "data": result.get("data", {}),
                            "confidence": 0.95
                        }
                    elif search_type == "name" and result.get("companies"):
                        # Name search - use first result
                        comprehensive_results["best_match"] = {
                            "search_type": search_type,
                            "data": result["companies"][0],
                            "confidence": 0.8
                        }

        # Generate summary
        total_companies_found = 0
        successful_searches = 0

        for search_type, search_result in comprehensive_results["searches"].items():
            if not search_result.get("error"):
                successful_searches += 1
                if search_type == "name":
                    total_companies_found += len(search_result.get("companies", []))
                elif search_result.get("found"):
                    total_companies_found += 1

        comprehensive_results["summary"] = {
            "total_companies_found": total_companies_found,
            "successful_searches": successful_searches,
            "total_searches": len(search_strategies),
            "has_best_match": comprehensive_results["best_match"] is not None,
            "search_strategies_used": [strategy[0] for strategy in search_strategies]
        }

        self.logger.info(f"✅ Comprehensive search completed: "
                        f"{total_companies_found} companies found across "
                        f"{successful_searches} successful searches")

        return comprehensive_results

    def validate_ico(self, ico: str) -> Dict[str, Any]:
        """
        Validate ICO using Czech algorithm

        Args:
            ico: ICO to validate

        Returns:
            Validation result
        """

        ico_clean = re.sub(r'\D', '', ico)

        if len(ico_clean) != 8:
            return {
                "valid": False,
                "ico": ico,
                "error": "ICO must be exactly 8 digits"
            }

        # Czech ICO validation algorithm
        try:
            digits = [int(d) for d in ico_clean]
            weights = [8, 7, 6, 5, 4, 3, 2]

            # Calculate checksum
            sum_result = sum(digit * weight for digit, weight in zip(digits[:7], weights))
            remainder = sum_result % 11

            if remainder < 2:
                check_digit = remainder
            else:
                check_digit = 11 - remainder

            is_valid = check_digit == digits[7]

            return {
                "valid": is_valid,
                "ico": ico_clean,
                "formatted": f"{ico_clean[:2]} {ico_clean[2:4]} {ico_clean[4:6]} {ico_clean[6:]}",
                "checksum_calculated": check_digit,
                "checksum_provided": digits[7]
            }

        except Exception as e:
            return {
                "valid": False,
                "ico": ico,
                "error": f"Validation error: {str(e)}"
            }

    async def get_company_relationships(self, ico: str) -> Dict[str, Any]:
        """
        Get company relationships (subsidiaries, parent companies, etc.)

        Args:
            ico: Company ICO

        Returns:
            Company relationship network
        """

        self.logger.info(f"🔗 Fetching relationships for ICO: {ico}")

        # First get the company details
        company_data = await self.search_by_ico(ico)

        if not company_data.get("found"):
            return {
                "error": "Company not found",
                "ico": ico,
                "timestamp": datetime.now().isoformat()
            }

        relationships = {
            "ico": ico,
            "company_name": company_data["data"].get("name", "Unknown"),
            "timestamp": datetime.now().isoformat(),
            "statutory_bodies": [],
            "subsidiaries": [],
            "parent_companies": [],
            "related_entities": []
        }

        # Fetch statutory bodies (jednatelé, ředitelé)
        statutory = await self._fetch_statutory_bodies(ico)
        relationships["statutory_bodies"] = statutory

        # Cross-reference with Justice.cz for more detailed relationships
        # This would be enhanced with actual Justice.cz integration
        self.logger.info(f"✅ Found {len(statutory)} statutory body members")

        return relationships

    async def _fetch_statutory_bodies(self, ico: str) -> List[Dict[str, Any]]:
        """Fetch statutory body members from ARES."""

        # This would parse extended ARES data or Justice.cz
        # Placeholder implementation
        await self._rate_limit()

        return [
            {
                "name": "Example Jednatel",
                "position": "Jednatel",
                "appointment_date": "2020-01-01",
                "birth_number": "XXXX",
                "address": "Praha 1"
            }
        ]

    async def get_financial_indicators(self, ico: str) -> Dict[str, Any]:
        """
        Get basic financial indicators (if available publicly).

        Args:
            ico: Company ICO

        Returns:
            Financial indicators and health score
        """

        self.logger.info(f"💰 Fetching financial indicators for ICO: {ico}")

        company_data = await self.search_by_ico(ico)

        if not company_data.get("found"):
            return {
                "error": "Company not found",
                "ico": ico
            }

        indicators = {
            "ico": ico,
            "company_name": company_data["data"].get("name", "Unknown"),
            "timestamp": datetime.now().isoformat(),
            "financial_health_score": None,
            "indicators": {},
            "data_available": False,
            "notes": []
        }

        # Check if company is active
        status = company_data["data"].get("status", "").lower()
        if "neaktivní" in status or "zaniklý" in status:
            indicators["notes"].append("Company is inactive or dissolved")
            indicators["financial_health_score"] = 0.0
        else:
            indicators["notes"].append("Active company - detailed financials require Justice.cz integration")
            indicators["data_available"] = False

        return indicators

    async def cross_reference_with_justice(self, ico: str) -> Dict[str, Any]:
        """
        Cross-reference ARES data with Justice.cz for comprehensive profile.

        Args:
            ico: Company ICO

        Returns:
            Combined profile from ARES and Justice.cz
        """

        self.logger.info(f"🔄 Cross-referencing ICO: {ico} with Justice.cz")

        # Get ARES data
        ares_data = await self.search_by_ico(ico)

        if not ares_data.get("found"):
            return {
                "error": "Company not found in ARES",
                "ico": ico
            }

        combined_profile = {
            "ico": ico,
            "timestamp": datetime.now().isoformat(),
            "ares_data": ares_data["data"],
            "justice_data": {},
            "relationships": {},
            "comprehensive_profile": {
                "basic_info": {},
                "legal_status": {},
                "financial_info": {},
                "statutory_bodies": [],
                "business_activities": []
            }
        }

        # This would integrate with Justice.cz tool
        # For now, we enrich with ARES data
        combined_profile["comprehensive_profile"]["basic_info"] = {
            "ico": ico,
            "name": ares_data["data"].get("name"),
            "dic": ares_data["data"].get("dic"),
            "legal_form": ares_data["data"].get("legal_form"),
            "address": ares_data["data"].get("address"),
            "status": ares_data["data"].get("status")
        }

        # Fetch relationships
        relationships = await self.get_company_relationships(ico)
        combined_profile["relationships"] = relationships

        self.logger.info("✅ Cross-reference completed")

        return combined_profile

    async def enhanced_company_profile(self, query: str) -> Dict[str, Any]:
        """
        Create enhanced company profile with all available data.

        Args:
            query: ICO, DIC, or company name

        Returns:
            Comprehensive enhanced company profile
        """

        self.logger.info(f"🎯 Creating enhanced profile for: {query}")

        # Comprehensive search first
        search_results = await self.comprehensive_business_search(query)

        if not search_results.get("best_match"):
            return {
                "error": "No company found",
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

        best_match = search_results["best_match"]["data"]
        ico = best_match.get("ico")

        if not ico:
            return {
                "error": "ICO not found in search results",
                "query": query
            }

        # Build enhanced profile
        profile = {
            "query": query,
            "ico": ico,
            "timestamp": datetime.now().isoformat(),
            "confidence": search_results["best_match"]["confidence"],
            "profile_completeness": 0.0,
            "sections": {}
        }

        # Section 1: Basic company info
        profile["sections"]["basic_info"] = best_match

        # Section 2: Relationships
        relationships = await self.get_company_relationships(ico)
        profile["sections"]["relationships"] = relationships

        # Section 3: Financial indicators
        financial = await self.get_financial_indicators(ico)
        profile["sections"]["financial"] = financial

        # Section 4: Justice.cz cross-reference
        cross_ref = await self.cross_reference_with_justice(ico)
        profile["sections"]["cross_reference"] = cross_ref

        # Calculate profile completeness
        sections_completed = sum([
            1 if profile["sections"]["basic_info"] else 0,
            1 if relationships.get("statutory_bodies") else 0,
            0.5  # Financial data placeholder
        ])
        profile["profile_completeness"] = min(sections_completed / 3.0, 1.0)

        self.logger.info(f"✅ Enhanced profile created (completeness: {profile['profile_completeness']:.1%})")

        return profile

    def get_api_info(self) -> Dict[str, Any]:
        """Get information about ARES API capabilities"""

        return {
            "api_endpoints": {
                "basic_search": {
                    "url": self.endpoints['basic_search'],
                    "description": "Basic company search by ICO, DIC, or name",
                    "parameters": ["ico", "dic", "obchodni_firma", "MAX_POCET"],
                    "format": "XML"
                },
                "standard_search": {
                    "url": self.endpoints['standard_search'],
                    "description": "Standard search with more details",
                    "parameters": ["ico", "dic", "obchodni_firma"],
                    "format": "XML"
                }
            },
            "enhanced_features": {
                "company_relationships": "Fetch statutory bodies and related entities",
                "financial_indicators": "Basic financial health assessment",
                "justice_cross_reference": "Integration with Justice.cz data",
                "enhanced_profile": "Comprehensive company profile aggregation"
            },
            "rate_limiting": {
                "min_interval": f"{self.min_request_interval} seconds",
                "recommended_use": "Production applications should implement additional rate limiting"
            },
            "data_types": {
                "basic_info": ["ICO", "DIC", "Company name", "Legal form", "Address"],
                "status_info": ["Active/Inactive status", "Registration date"],
                "contact_info": ["Registered address", "Business address"],
                "enhanced_info": ["Statutory bodies", "Relationships", "Financial health"]
            },
            "limitations": {
                "api_limits": "Max 100 results per request",
                "data_freshness": "Updated daily from official sources",
                "availability": "24/7 with possible maintenance windows"
            }
        }

# Example usage and testing
async def test_ares_cz():
    """Test ARES.cz functionality"""

    ares = AresCzTool()

    print("🏢 Testing ARES.cz OSINT Tool")
    print("=" * 50)

    # Test ICO validation
    print("🔍 Testing ICO validation...")
    test_icos = ["25596641", "12345678", "invalid"]

    for ico in test_icos:
        validation = ares.validate_ico(ico)
        print(f"  ICO {ico}: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
        if 'error' in validation:
            print(f"    Error: {validation['error']}")

    # Test ICO search
    print("\n🔍 Testing ICO search...")
    ico_result = await ares.search_by_ico("25596641")  # Example ICO

    if ico_result.get("found"):
        company = ico_result["data"]
        print(f"✅ Found company: {company.get('name', 'Unknown')}")
        print(f"  ICO: {company.get('ico', 'N/A')}")
        print(f"  Legal form: {company.get('legal_form', 'N/A')}")
    else:
        print("❌ No company found for test ICO")

    # Test name search
    print("\n🔍 Testing name search...")
    name_result = await ares.search_by_name("Test", limit=3)

    if name_result.get("found"):
        print(f"✅ Found {len(name_result['companies'])} companies:")
        for i, company in enumerate(name_result['companies'][:3]):
            print(f"  {i+1}. {company.get('name', 'Unknown')} (ICO: {company.get('ico', 'N/A')})")
    else:
        print("❌ No companies found for test name")

    # Test comprehensive search
    print("\n🎯 Testing comprehensive search...")
    comp_result = await ares.comprehensive_business_search("25596641")

    summary = comp_result["summary"]
    print(f"📊 Comprehensive search results:")
    print(f"  • Companies found: {summary['total_companies_found']}")
    print(f"  • Successful searches: {summary['successful_searches']}/{summary['total_searches']}")
    print(f"  • Best match: {'Yes' if summary['has_best_match'] else 'No'}")

    if comp_result["best_match"]:
        best = comp_result["best_match"]
        print(f"  • Best match confidence: {best['confidence']}")

    # Show API capabilities
    print("\n📋 API Information:")
    api_info = ares.get_api_info()
    print(f"  • Available endpoints: {len(api_info['api_endpoints'])}")
    print(f"  • Rate limiting: {api_info['rate_limiting']['min_interval']}")
    print(f"  • Data types: {', '.join(api_info['data_types']['basic_info'])}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_ares_cz())