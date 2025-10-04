"""
Test Suite for Czech OSINT Phase 4 Components
==============================================

Tests:
- Cadastre property search
- Enhanced ARES features
- Enhanced Justice.cz features
- Czech OSINT Orchestrator

Author: AL-OSINT Suite
Created: 2025-10-04
"""

import asyncio
import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools.government.cadastre_cz import (
    CadastreCzTool,
    PropertyType,
    OwnershipType
)
from tools.government.ares_cz import AresCzTool
from tools.government.justice_cz import JusticeCzTool
from tools.government.czech_osint_orchestrator import (
    CzechOSINTOrchestrator,
    InvestigationTargetType,
    investigate_company,
    investigate_person
)


class TestCadastreCz(unittest.TestCase):
    """Test Cadastre property search tool."""

    def setUp(self):
        """Set up test fixtures."""
        self.cadastre = CadastreCzTool()

    def test_cadastre_initialization(self):
        """Test cadastre tool initializes correctly."""
        self.assertIsNotNone(self.cadastre)
        self.assertTrue(self.cadastre.output_dir.exists())
        self.assertEqual(self.cadastre.stats["searches_performed"], 0)

    def test_search_by_address(self):
        """Test address-based property search."""
        async def run_test():
            result = await self.cadastre.search_by_address(
                "Test Street 1",
                city="Praha"
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.query, "Test Street 1")
            self.assertEqual(result.search_type, "address")
            self.assertTrue(result.search_time_ms >= 0)

            return result

        result = asyncio.run(run_test())
        print(f"✅ Address search: {result.total_found} properties found")

    def test_search_by_owner(self):
        """Test owner-based property search."""
        async def run_test():
            result = await self.cadastre.search_by_owner("Jan Novák")

            self.assertIsNotNone(result)
            self.assertEqual(result.search_type, "owner")
            self.assertIn("properties", dir(result))

            return result

        result = asyncio.run(run_test())
        print(f"✅ Owner search: {result.total_found} properties found")

    def test_search_by_lv(self):
        """Test LV number search."""
        async def run_test():
            result = await self.cadastre.search_by_lv("1234", "Praha 1")

            self.assertIsNotNone(result)
            self.assertEqual(result.search_type, "lv")

            if result.success:
                self.assertGreater(len(result.properties), 0)
                prop = result.properties[0]
                self.assertEqual(prop.lv_number, "1234")

            return result

        result = asyncio.run(run_test())
        print(f"✅ LV search: {result.success}")

    def test_ownership_history(self):
        """Test ownership history retrieval."""
        async def run_test():
            owners = await self.cadastre.get_ownership_history("1234", "Praha 1")

            self.assertIsInstance(owners, list)

            if owners:
                # Check chronological ordering (newest first)
                for i in range(len(owners) - 1):
                    if owners[i].date_from and owners[i+1].date_from:
                        self.assertGreaterEqual(
                            owners[i].date_from,
                            owners[i+1].date_from
                        )

            return owners

        owners = asyncio.run(run_test())
        print(f"✅ Ownership history: {len(owners)} owners")

    def test_statistics(self):
        """Test statistics tracking."""
        stats = self.cadastre.get_statistics()

        self.assertIn("searches_performed", stats)
        self.assertIn("cache_size", stats)
        self.assertIn("cache_hit_rate", stats)

        print(f"✅ Statistics: {stats['searches_performed']} searches")


class TestEnhancedARES(unittest.TestCase):
    """Test enhanced ARES features."""

    def setUp(self):
        """Set up test fixtures."""
        self.ares = AresCzTool()

    def test_company_relationships(self):
        """Test company relationships extraction."""
        async def run_test():
            # Use a test ICO
            relationships = await self.ares.get_company_relationships("25596641")

            self.assertIsNotNone(relationships)
            self.assertIn("statutory_bodies", relationships)
            self.assertIn("subsidiaries", relationships)
            self.assertIn("related_entities", relationships)

            return relationships

        result = asyncio.run(run_test())
        print(f"✅ Relationships: {len(result.get('statutory_bodies', []))} statutory bodies")

    def test_financial_indicators(self):
        """Test financial indicators extraction."""
        async def run_test():
            indicators = await self.ares.get_financial_indicators("25596641")

            self.assertIsNotNone(indicators)
            self.assertIn("financial_health_score", indicators)
            self.assertIn("data_available", indicators)

            return indicators

        result = asyncio.run(run_test())
        print(f"✅ Financial indicators: {result.get('data_available', False)}")

    def test_cross_reference_justice(self):
        """Test ARES-Justice.cz cross-reference."""
        async def run_test():
            profile = await self.ares.cross_reference_with_justice("25596641")

            self.assertIsNotNone(profile)
            self.assertIn("ares_data", profile)
            self.assertIn("justice_data", profile)
            self.assertIn("comprehensive_profile", profile)

            return profile

        result = asyncio.run(run_test())
        print(f"✅ Cross-reference completed")

    def test_enhanced_profile(self):
        """Test enhanced company profile."""
        async def run_test():
            profile = await self.ares.enhanced_company_profile("25596641")

            self.assertIsNotNone(profile)
            self.assertIn("profile_completeness", profile)
            self.assertIn("sections", profile)

            # Check all sections present
            sections = profile["sections"]
            self.assertIn("basic_info", sections)
            self.assertIn("relationships", sections)
            self.assertIn("financial", sections)

            return profile

        result = asyncio.run(run_test())
        print(f"✅ Enhanced profile: {result.get('profile_completeness', 0):.1%} complete")


class TestEnhancedJustice(unittest.TestCase):
    """Test enhanced Justice.cz features."""

    def setUp(self):
        """Set up test fixtures."""
        self.justice = JusticeCzTool()

    def test_detailed_case_info(self):
        """Test detailed case information retrieval."""
        async def run_test():
            case_info = await self.justice.get_detailed_case_info(
                "12 C 34/2024",
                "Okresní soud Praha"
            )

            self.assertIsNotNone(case_info)
            self.assertEqual(case_info["case_number"], "12 C 34/2024")
            self.assertIn("parties", case_info)
            self.assertIn("documents", case_info)
            self.assertIn("timeline", case_info)

            return case_info

        result = asyncio.run(run_test())
        print(f"✅ Case details: {len(result.get('documents', []))} documents")

    def test_company_litigations(self):
        """Test company litigations extraction."""
        async def run_test():
            litigations = await self.justice.extract_company_litigations("Test Firma s.r.o.")

            self.assertIsNotNone(litigations)
            self.assertIn("as_plaintiff", litigations)
            self.assertIn("as_defendant", litigations)
            self.assertIn("statistics", litigations)

            stats = litigations["statistics"]
            self.assertIn("total_cases", stats)
            self.assertIn("active_cases", stats)

            return litigations

        result = asyncio.run(run_test())
        print(f"✅ Litigations: {result['statistics']['total_cases']} cases")

    def test_cross_reference_ares(self):
        """Test Justice-ARES cross-reference."""
        async def run_test():
            profile = await self.justice.cross_reference_with_ares("Test Firma s.r.o.")

            self.assertIsNotNone(profile)
            self.assertIn("justice_data", profile)
            self.assertIn("comprehensive_profile", profile)

            comp_profile = profile["comprehensive_profile"]
            self.assertIn("legal_health_score", comp_profile)
            self.assertIn("insolvency_risk", comp_profile)

            return profile

        result = asyncio.run(run_test())
        comp = result["comprehensive_profile"]
        print(f"✅ Cross-reference: Legal health {comp.get('legal_health_score', 0):.2f}")

    def test_enhanced_person_profile(self):
        """Test enhanced person profile."""
        async def run_test():
            profile = await self.justice.enhanced_person_profile("Jan Novák")

            self.assertIsNotNone(profile)
            self.assertIn("profile_completeness", profile)
            self.assertIn("risk_assessment", profile)

            risk = profile["risk_assessment"]
            self.assertIn("risk_level", risk)

            return profile

        result = asyncio.run(run_test())
        print(f"✅ Person profile: {result.get('profile_completeness', 0):.1%} complete")


class TestCzechOSINTOrchestrator(unittest.TestCase):
    """Test unified Czech OSINT orchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = CzechOSINTOrchestrator()

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        self.assertIsNotNone(self.orchestrator.ares)
        self.assertIsNotNone(self.orchestrator.justice)
        self.assertIsNotNone(self.orchestrator.cadastre)

        stats = self.orchestrator.stats
        self.assertEqual(stats["investigations_performed"], 0)

    def test_target_type_detection(self):
        """Test automatic target type detection."""
        # Company detection
        company_type = self.orchestrator._detect_target_type("Test s.r.o.")
        self.assertEqual(company_type, InvestigationTargetType.COMPANY)

        # ICO detection
        ico_type = self.orchestrator._detect_target_type("25596641")
        self.assertEqual(ico_type, InvestigationTargetType.COMPANY)

        # Person detection
        person_type = self.orchestrator._detect_target_type("Jan Novák")
        self.assertEqual(person_type, InvestigationTargetType.PERSON)

        # Property detection
        property_type = self.orchestrator._detect_target_type("Václavské náměstí 1, Praha")
        self.assertEqual(property_type, InvestigationTargetType.PROPERTY)

        print("✅ Target type detection working")

    def test_investigate_company(self):
        """Test comprehensive company investigation."""
        async def run_test():
            result = await self.orchestrator.investigate_entity(
                "Test s.r.o.",
                target_type=InvestigationTargetType.COMPANY
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.target_type, InvestigationTargetType.COMPANY)
            self.assertGreater(len(result.sources_queried), 0)

            # Check data was collected
            self.assertTrue(
                result.ares_data or result.justice_data or result.cadastre_data
            )

            # Check profile was built
            self.assertIn("basic_info", result.comprehensive_profile)

            return result

        result = asyncio.run(run_test())
        print(f"✅ Company investigation: {result.profile_completeness:.1%} complete")
        print(f"   Sources: {', '.join(result.sources_queried)}")

    def test_investigate_person(self):
        """Test comprehensive person investigation."""
        async def run_test():
            result = await self.orchestrator.investigate_entity(
                "Jan Novák",
                target_type=InvestigationTargetType.PERSON
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.target_type, InvestigationTargetType.PERSON)

            # Person investigations should query Justice and Cadastre primarily
            self.assertIn("Justice.cz", result.sources_queried)

            return result

        result = asyncio.run(run_test())
        print(f"✅ Person investigation: {result.profile_completeness:.1%} complete")

    def test_investigate_property(self):
        """Test property investigation."""
        async def run_test():
            result = await self.orchestrator.investigate_entity(
                "Test Street 1, Praha",
                target_type=InvestigationTargetType.PROPERTY
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.target_type, InvestigationTargetType.PROPERTY)

            # Property investigations should query Cadastre
            self.assertIn("Cadastre", result.sources_queried)

            return result

        result = asyncio.run(run_test())
        print(f"✅ Property investigation: {result.profile_completeness:.1%} complete")

    def test_cross_reference_integration(self):
        """Test cross-referencing across all sources."""
        async def run_test():
            result = await self.orchestrator.investigate_entity(
                "Test s.r.o.",
                target_type=InvestigationTargetType.COMPANY,
                include_properties=True,
                include_legal_records=True
            )

            # Should have comprehensive profile with all sections
            profile = result.comprehensive_profile

            self.assertIn("basic_info", profile)
            self.assertIn("legal_status", profile)
            self.assertIn("property_ownership", profile)

            # Should have risk assessment
            self.assertIn("overall_risk_score", result.risk_assessment)
            self.assertIn("risk_level", result.risk_assessment)

            return result

        result = asyncio.run(run_test())
        print(f"✅ Cross-reference integration: Risk level {result.risk_assessment['risk_level']}")

    def test_convenience_functions(self):
        """Test convenience wrapper functions."""
        async def run_test():
            # Test company investigation
            company_result = await investigate_company("Test s.r.o.")
            self.assertEqual(company_result.target_type, InvestigationTargetType.COMPANY)

            # Test person investigation
            person_result = await investigate_person("Jan Novák")
            self.assertEqual(person_result.target_type, InvestigationTargetType.PERSON)

            return company_result, person_result

        company, person = asyncio.run(run_test())
        print(f"✅ Convenience functions working")


def run_tests():
    """Run all Czech OSINT tests."""
    print("\n" + "=" * 80)
    print("🧪 CZECH OSINT PHASE 4 TEST SUITE")
    print("=" * 80 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCadastreCz))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedARES))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedJustice))
    suite.addTests(loader.loadTestsFromTestCase(TestCzechOSINTOrchestrator))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("🧪 CZECH OSINT PHASE 4 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped)}")
    print("=" * 80)

    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        success_rate = 100.0
    else:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
        print(f"⚠️  SUCCESS RATE: {success_rate:.1f}%")

    print("=" * 80 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
