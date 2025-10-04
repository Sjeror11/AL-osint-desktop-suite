"""
Czech OSINT Orchestrator
========================

Unified orchestrator for all Czech OSINT sources.

Coordinates:
- ARES (Business Registry)
- Justice.cz (Court Records)
- Cadastre (Property Registry)

Features:
- Comprehensive entity profiling
- Cross-source data correlation
- Automated investigation workflows
- Risk assessment and scoring

Author: AL-OSINT Suite
Created: 2025-10-04
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from .ares_cz import AresCzTool
from .justice_cz import JusticeCzTool
from .cadastre_cz import CadastreCzTool


logger = logging.getLogger(__name__)


class InvestigationTargetType(Enum):
    """Types of investigation targets."""
    PERSON = "person"
    COMPANY = "company"
    PROPERTY = "property"
    UNKNOWN = "unknown"


@dataclass
class CzechOSINTResult:
    """Unified result from Czech OSINT investigation."""
    target: str
    target_type: InvestigationTargetType
    timestamp: datetime = field(default_factory=datetime.now)

    # Data from sources
    ares_data: Dict[str, Any] = field(default_factory=dict)
    justice_data: Dict[str, Any] = field(default_factory=dict)
    cadastre_data: Dict[str, Any] = field(default_factory=dict)

    # Cross-referenced insights
    comprehensive_profile: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    sources_queried: List[str] = field(default_factory=list)
    profile_completeness: float = 0.0
    confidence_score: float = 0.0

    success: bool = False
    errors: List[str] = field(default_factory=list)


class CzechOSINTOrchestrator:
    """
    Unified orchestrator for Czech OSINT sources.

    Provides comprehensive investigation capabilities by coordinating
    ARES, Justice.cz, and Cadastre searches with intelligent
    cross-referencing and data correlation.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        enable_caching: bool = True
    ):
        """
        Initialize Czech OSINT Orchestrator.

        Args:
            output_dir: Directory for saving results
            enable_caching: Enable response caching across tools
        """
        self.output_dir = Path(output_dir) if output_dir else Path("data/czech_osint")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tools
        self.ares = AresCzTool()
        self.justice = JusticeCzTool()
        self.cadastre = CadastreCzTool(
            output_dir=self.output_dir / "cadastre",
            enable_caching=enable_caching
        )

        # Statistics
        self.stats = {
            "investigations_performed": 0,
            "ares_queries": 0,
            "justice_queries": 0,
            "cadastre_queries": 0,
            "cross_references": 0
        }

        logger.info("CzechOSINTOrchestrator initialized")

    async def investigate_entity(
        self,
        target: str,
        target_type: Optional[InvestigationTargetType] = None,
        include_properties: bool = True,
        include_legal_records: bool = True
    ) -> CzechOSINTResult:
        """
        Perform comprehensive investigation of an entity.

        Args:
            target: Target name (person, company, or property address)
            target_type: Type of target (auto-detected if None)
            include_properties: Include property ownership search
            include_legal_records: Include legal/court records

        Returns:
            Comprehensive investigation results
        """
        logger.info(f"🔍 Starting Czech OSINT investigation: {target}")

        # Auto-detect target type if not specified
        if target_type is None:
            target_type = self._detect_target_type(target)
            logger.info(f"Auto-detected target type: {target_type.value}")

        result = CzechOSINTResult(
            target=target,
            target_type=target_type
        )

        try:
            # Execute searches based on target type
            if target_type == InvestigationTargetType.COMPANY:
                await self._investigate_company(target, result, include_properties, include_legal_records)
            elif target_type == InvestigationTargetType.PERSON:
                await self._investigate_person(target, result, include_properties, include_legal_records)
            elif target_type == InvestigationTargetType.PROPERTY:
                await self._investigate_property(target, result)
            else:
                # Unknown type - try all approaches
                await self._investigate_unknown(target, result)

            # Cross-reference and consolidate data
            await self._cross_reference_data(result)

            # Calculate completeness and confidence
            self._calculate_metrics(result)

            result.success = True

            # Update stats
            self.stats["investigations_performed"] += 1

            logger.info(f"✅ Investigation completed (completeness: {result.profile_completeness:.1%})")

        except Exception as e:
            logger.error(f"❌ Investigation error: {e}")
            result.errors.append(str(e))
            result.success = False

        return result

    async def _investigate_company(
        self,
        company_name: str,
        result: CzechOSINTResult,
        include_properties: bool,
        include_legal_records: bool
    ) -> None:
        """Investigate a company using all available sources."""
        logger.info(f"🏢 Investigating company: {company_name}")

        tasks = []

        # ARES search (required for companies)
        tasks.append(("ares", self.ares.enhanced_company_profile(company_name)))
        result.sources_queried.append("ARES")
        self.stats["ares_queries"] += 1

        # Justice.cz search (if enabled)
        if include_legal_records:
            tasks.append(("justice", self.justice.cross_reference_with_ares(company_name)))
            result.sources_queried.append("Justice.cz")
            self.stats["justice_queries"] += 1

        # Cadastre search (if enabled)
        if include_properties:
            # Search by company name as owner
            tasks.append(("cadastre", self.cadastre.search_by_owner(company_name)))
            result.sources_queried.append("Cadastre")
            self.stats["cadastre_queries"] += 1

        # Execute searches concurrently
        results_list = await asyncio.gather(
            *[task[1] for task in tasks],
            return_exceptions=True
        )

        # Process results
        for i, (source_name, _) in enumerate(tasks):
            data = results_list[i]

            if isinstance(data, Exception):
                result.errors.append(f"{source_name}: {str(data)}")
                logger.error(f"Error from {source_name}: {data}")
            else:
                if source_name == "ares":
                    result.ares_data = data
                elif source_name == "justice":
                    result.justice_data = data
                elif source_name == "cadastre":
                    result.cadastre_data = data

    async def _investigate_person(
        self,
        person_name: str,
        result: CzechOSINTResult,
        include_properties: bool,
        include_legal_records: bool
    ) -> None:
        """Investigate a person using all available sources."""
        logger.info(f"👤 Investigating person: {person_name}")

        tasks = []

        # Justice.cz search (primary for persons)
        if include_legal_records:
            tasks.append(("justice", self.justice.enhanced_person_profile(person_name)))
            result.sources_queried.append("Justice.cz")
            self.stats["justice_queries"] += 1

        # Cadastre search (property ownership)
        if include_properties:
            tasks.append(("cadastre", self.cadastre.search_by_owner(person_name)))
            result.sources_queried.append("Cadastre")
            self.stats["cadastre_queries"] += 1

        # ARES search (if person is entrepreneur - IČO search)
        # This is optional for persons
        tasks.append(("ares", self.ares.search_by_name(person_name, limit=5)))
        result.sources_queried.append("ARES")
        self.stats["ares_queries"] += 1

        # Execute searches concurrently
        results_list = await asyncio.gather(
            *[task[1] for task in tasks],
            return_exceptions=True
        )

        # Process results
        for i, (source_name, _) in enumerate(tasks):
            data = results_list[i]

            if isinstance(data, Exception):
                result.errors.append(f"{source_name}: {str(data)}")
            else:
                if source_name == "ares":
                    result.ares_data = data
                elif source_name == "justice":
                    result.justice_data = data
                elif source_name == "cadastre":
                    result.cadastre_data = data

    async def _investigate_property(
        self,
        property_address: str,
        result: CzechOSINTResult
    ) -> None:
        """Investigate a property using cadastre."""
        logger.info(f"🏠 Investigating property: {property_address}")

        # Cadastre search
        cadastre_result = await self.cadastre.search_by_address(property_address)
        result.cadastre_data = cadastre_result
        result.sources_queried.append("Cadastre")
        self.stats["cadastre_queries"] += 1

        # If property has owners, investigate them
        if cadastre_result.success and cadastre_result.properties:
            for prop in cadastre_result.properties:
                for owner in prop.owners:
                    if owner.is_current:
                        # Recursive investigation of owner
                        logger.info(f"🔗 Investigating property owner: {owner.name}")

                        # Determine if owner is person or company
                        if owner.ic:  # Company (has IČO)
                            ares_data = await self.ares.search_by_ico(owner.ic)
                            result.ares_data[owner.name] = ares_data
                            self.stats["ares_queries"] += 1
                        else:  # Person
                            justice_data = await self.justice.search_civil_proceedings(owner.name)
                            result.justice_data[owner.name] = justice_data
                            self.stats["justice_queries"] += 1

    async def _investigate_unknown(
        self,
        target: str,
        result: CzechOSINTResult
    ) -> None:
        """Investigate unknown target type - try all approaches."""
        logger.info(f"❓ Investigating unknown target: {target}")

        # Try comprehensive search across all sources
        tasks = [
            ("ares", self.ares.comprehensive_business_search(target)),
            ("justice", self.justice.comprehensive_justice_search(target)),
            ("cadastre", self.cadastre.search_by_address(target))
        ]

        result.sources_queried.extend(["ARES", "Justice.cz", "Cadastre"])
        self.stats["ares_queries"] += 1
        self.stats["justice_queries"] += 1
        self.stats["cadastre_queries"] += 1

        results_list = await asyncio.gather(
            *[task[1] for task in tasks],
            return_exceptions=True
        )

        # Process results
        for i, (source_name, _) in enumerate(tasks):
            data = results_list[i]

            if not isinstance(data, Exception):
                if source_name == "ares":
                    result.ares_data = data
                elif source_name == "justice":
                    result.justice_data = data
                elif source_name == "cadastre":
                    result.cadastre_data = data

    async def _cross_reference_data(self, result: CzechOSINTResult) -> None:
        """Cross-reference data from all sources to build comprehensive profile."""
        logger.info("🔄 Cross-referencing data from all sources")

        self.stats["cross_references"] += 1

        profile = {
            "basic_info": {},
            "business_info": {},
            "legal_status": {},
            "property_ownership": {},
            "relationships": {},
            "timeline": []
        }

        # Extract basic info from ARES
        if result.ares_data:
            ares_profile = result.ares_data.get("sections", {})
            if ares_profile:
                basic = ares_profile.get("basic_info", {})
                profile["basic_info"] = {
                    "name": basic.get("name"),
                    "ico": basic.get("ico"),
                    "dic": basic.get("dic"),
                    "legal_form": basic.get("legal_form"),
                    "address": basic.get("address")
                }

                # Business relationships
                relationships = ares_profile.get("relationships", {})
                profile["relationships"]["statutory_bodies"] = relationships.get("statutory_bodies", [])

        # Extract legal info from Justice.cz
        if result.justice_data:
            if result.target_type == InvestigationTargetType.COMPANY:
                justice_profile = result.justice_data.get("comprehensive_profile", {})
                profile["legal_status"] = {
                    "legal_health_score": justice_profile.get("legal_health_score", 0.0),
                    "active_litigations": justice_profile.get("active_litigations", 0),
                    "insolvency_risk": justice_profile.get("insolvency_risk", "Unknown")
                }
            else:  # Person
                justice_profile = result.justice_data.get("risk_assessment", {})
                profile["legal_status"] = {
                    "insolvency_filings": justice_profile.get("insolvency_filings", 0),
                    "active_litigations": justice_profile.get("active_litigations", 0),
                    "risk_level": justice_profile.get("risk_level", "Unknown")
                }

        # Extract property info from Cadastre
        if result.cadastre_data:
            cadastre_result = result.cadastre_data
            if hasattr(cadastre_result, 'properties'):
                properties = cadastre_result.properties
                profile["property_ownership"] = {
                    "total_properties": len(properties),
                    "properties": [
                        {
                            "address": prop.address,
                            "type": prop.property_type.value,
                            "lv_number": prop.lv_number,
                            "cadastral_area": prop.cadastral_area
                        }
                        for prop in properties
                    ]
                }

        result.comprehensive_profile = profile

        # Calculate risk assessment
        result.risk_assessment = self._calculate_risk_assessment(result)

    def _calculate_risk_assessment(self, result: CzechOSINTResult) -> Dict[str, Any]:
        """Calculate overall risk assessment from all sources."""
        risk = {
            "overall_risk_score": 0.0,
            "risk_level": "Low",
            "risk_factors": []
        }

        # Legal risks from Justice.cz
        legal_status = result.comprehensive_profile.get("legal_status", {})

        if isinstance(legal_status.get("legal_health_score"), (int, float)):
            # Company risk
            health_score = legal_status.get("legal_health_score", 1.0)
            risk["overall_risk_score"] += (1.0 - health_score) * 0.6  # 60% weight

            if health_score < 0.5:
                risk["risk_factors"].append("Low legal health score")
        else:
            # Person risk
            insolvency = legal_status.get("insolvency_filings", 0)
            litigations = legal_status.get("active_litigations", 0)

            if insolvency > 0:
                risk["overall_risk_score"] += 0.5
                risk["risk_factors"].append(f"{insolvency} insolvency filing(s)")

            if litigations > 3:
                risk["overall_risk_score"] += 0.3
                risk["risk_factors"].append(f"{litigations} active litigation(s)")

        # Determine risk level
        if risk["overall_risk_score"] > 0.7:
            risk["risk_level"] = "High"
        elif risk["overall_risk_score"] > 0.4:
            risk["risk_level"] = "Medium"
        else:
            risk["risk_level"] = "Low"

        return risk

    def _calculate_metrics(self, result: CzechOSINTResult) -> None:
        """Calculate profile completeness and confidence scores."""

        # Completeness: how many sources returned data
        sources_with_data = sum([
            1 if result.ares_data else 0,
            1 if result.justice_data else 0,
            1 if result.cadastre_data else 0
        ])

        result.profile_completeness = sources_with_data / len(result.sources_queried) if result.sources_queried else 0.0

        # Confidence: based on data quality and cross-validation
        confidence_factors = []

        # ARES confidence
        if result.ares_data:
            ares_confidence = result.ares_data.get("confidence", 0.0)
            if ares_confidence > 0:
                confidence_factors.append(ares_confidence)

        # Cadastre confidence
        if result.cadastre_data:
            if hasattr(result.cadastre_data, 'properties'):
                avg_confidence = sum(p.confidence for p in result.cadastre_data.properties) / len(result.cadastre_data.properties) if result.cadastre_data.properties else 0.0
                confidence_factors.append(avg_confidence)

        # Average confidence
        result.confidence_score = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    def _detect_target_type(self, target: str) -> InvestigationTargetType:
        """Auto-detect target type based on format."""

        # Check if ICO format (8 digits)
        import re
        if re.match(r'^\d{8}$', re.sub(r'\D', '', target)):
            return InvestigationTargetType.COMPANY

        # Check if contains company indicators
        company_indicators = ['s.r.o', 'a.s', 'o.p.s', 'v.o.s', 'k.s', 'spol.']
        if any(indicator in target.lower() for indicator in company_indicators):
            return InvestigationTargetType.COMPANY

        # Check if looks like address
        address_indicators = ['ul.', 'ulice', 'náměstí', 'třída', 'nám.', ',']
        if any(indicator in target.lower() for indicator in address_indicators):
            return InvestigationTargetType.PROPERTY

        # Check if looks like person name (has space, no company indicators)
        if ' ' in target and not any(indicator in target.lower() for indicator in company_indicators):
            return InvestigationTargetType.PERSON

        # Default to unknown
        return InvestigationTargetType.UNKNOWN

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator usage statistics."""
        return {
            **self.stats,
            "ares_cache_stats": self.ares.cache if hasattr(self.ares, 'cache') else {},
            "justice_cache_stats": self.justice.cache if hasattr(self.justice, 'cache') else {},
            "cadastre_stats": self.cadastre.get_statistics()
        }


# Convenience functions

async def investigate_company(
    company_name: str,
    output_dir: Optional[Path] = None
) -> CzechOSINTResult:
    """
    Quick company investigation.

    Args:
        company_name: Company name or ICO
        output_dir: Output directory

    Returns:
        Investigation results
    """
    orchestrator = CzechOSINTOrchestrator(output_dir=output_dir)
    return await orchestrator.investigate_entity(
        company_name,
        target_type=InvestigationTargetType.COMPANY
    )


async def investigate_person(
    person_name: str,
    output_dir: Optional[Path] = None
) -> CzechOSINTResult:
    """
    Quick person investigation.

    Args:
        person_name: Person's full name
        output_dir: Output directory

    Returns:
        Investigation results
    """
    orchestrator = CzechOSINTOrchestrator(output_dir=output_dir)
    return await orchestrator.investigate_entity(
        person_name,
        target_type=InvestigationTargetType.PERSON
    )


async def investigate_property(
    property_address: str,
    output_dir: Optional[Path] = None
) -> CzechOSINTResult:
    """
    Quick property investigation.

    Args:
        property_address: Property address
        output_dir: Output directory

    Returns:
        Investigation results
    """
    orchestrator = CzechOSINTOrchestrator(output_dir=output_dir)
    return await orchestrator.investigate_entity(
        property_address,
        target_type=InvestigationTargetType.PROPERTY
    )
