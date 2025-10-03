#!/usr/bin/env python3
"""
🧪 Orchestrator Integration Test - OSINT Desktop Suite
Testing Enhanced Orchestrator integration with Social Media components
"""

import sys
import asyncio
from datetime import datetime

print("🧪 OSINT Orchestrator Integration Test")
print("=" * 60)

async def test_orchestrator_initialization():
    """Test Enhanced Orchestrator initialization with social media components"""
    print("\n🤖 Testing Enhanced Orchestrator Initialization...")

    try:
        from src.core.enhanced_orchestrator import EnhancedInvestigationOrchestrator, InvestigationTarget, InvestigationType, InvestigationPriority

        # Initialize orchestrator
        orchestrator = EnhancedInvestigationOrchestrator()
        await orchestrator.initialize()
        print("✅ Enhanced Orchestrator initialized successfully")

        # Check social media integration
        if orchestrator.social_media_available and orchestrator.social_media_orchestrator:
            print("✅ Social media orchestration available and initialized")
        else:
            print("⚠️ Social media orchestration not available (expected in test environment)")

        return True, orchestrator

    except Exception as e:
        print(f"❌ Orchestrator initialization failed: {e}")
        return False, None

async def test_social_media_investigation():
    """Test social media investigation through orchestrator"""
    print("\n🔍 Testing Social Media Investigation...")

    try:
        from src.core.enhanced_orchestrator import EnhancedInvestigationOrchestrator, InvestigationTarget, InvestigationType

        # Initialize orchestrator
        orchestrator = EnhancedInvestigationOrchestrator()
        await orchestrator.initialize()

        # Create test target
        target = InvestigationTarget(
            name="John Doe",
            target_type=InvestigationType.PERSON,
            location="Prague",
            investigation_scope="social_media_focused"
        )

        print(f"🎯 Created investigation target: {target.name}")

        # Start investigation
        investigation_id = await orchestrator.start_investigation(target)
        print(f"✅ Investigation started with ID: {investigation_id}")

        return True

    except Exception as e:
        print(f"❌ Social media investigation test failed: {e}")
        return False

async def test_social_media_orchestrator_direct():
    """Test social media orchestrator directly"""
    print("\n📱 Testing Direct Social Media Orchestrator...")

    try:
        from src.core.social_media_orchestration import SocialMediaOrchestrator, SocialMediaPhase
        from src.core.enhanced_orchestrator import InvestigationTarget, InvestigationType

        # Initialize social media orchestrator
        sm_orchestrator = SocialMediaOrchestrator()
        await sm_orchestrator.initialize()
        print("✅ Social Media Orchestrator initialized successfully")

        # Create test target
        target = InvestigationTarget(
            name="Jane Smith",
            target_type=InvestigationType.PERSON,
            location="Brno"
        )

        # Create custom investigation phases
        custom_phases = [
            SocialMediaPhase(
                phase_name="quick_search",
                platforms=["facebook", "linkedin"],
                search_strategies=["name_search"],
                ai_enhancement=True
            )
        ]

        # Execute investigation
        results = await sm_orchestrator.execute_social_media_investigation(
            target=target,
            custom_phases=custom_phases
        )

        print(f"✅ Investigation completed for: {results.target_name}")
        print(f"📊 Platforms searched: {len(results.platforms_searched)}")
        print(f"👤 Profiles found: {len(results.profiles_found)}")
        print(f"🔗 Correlations: {len(results.correlations)}")
        print(f"📈 Confidence score: {results.confidence_score}")
        print(f"⏱️ Duration: {results.investigation_duration:.1f}s")

        return True

    except Exception as e:
        print(f"❌ Direct social media orchestrator test failed: {e}")
        return False

async def test_investigation_phase_detection():
    """Test social media phase detection logic"""
    print("\n🔍 Testing Investigation Phase Detection...")

    try:
        from src.core.enhanced_orchestrator import EnhancedInvestigationOrchestrator

        orchestrator = EnhancedInvestigationOrchestrator()

        # Test phase detection
        test_cases = [
            ("social_media_reconnaissance", ["facebook.com", "linkedin.com"], True),
            ("facebook_search", ["various_sources"], True),
            ("social_analysis", [], True),
            ("web_scraping", ["google.com", "bing.com"], False),
            ("database_search", ["justice.cz", "ares.cz"], False),
        ]

        for phase_name, sources, expected in test_cases:
            result = orchestrator._is_social_media_phase(phase_name, sources)
            if result == expected:
                print(f"✅ Phase detection: '{phase_name}' -> {result} (correct)")
            else:
                print(f"❌ Phase detection: '{phase_name}' -> {result} (expected {expected})")
                return False

        return True

    except Exception as e:
        print(f"❌ Phase detection test failed: {e}")
        return False

async def test_integration_components():
    """Test all integration components"""
    print("\n🔗 Testing Integration Components...")

    try:
        # Test imports
        from src.core.enhanced_orchestrator import EnhancedInvestigationOrchestrator
        from src.core.social_media_orchestration import SocialMediaOrchestrator, get_social_media_orchestrator
        from src.core.browser_integration import BrowserIntegrationAdapter

        print("✅ All integration imports successful")

        # Test component availability
        components = {
            "Enhanced Orchestrator": EnhancedInvestigationOrchestrator,
            "Social Media Orchestrator": SocialMediaOrchestrator,
            "Browser Integration": BrowserIntegrationAdapter
        }

        for name, component_class in components.items():
            try:
                instance = component_class()
                print(f"✅ {name}: instantiation successful")
            except Exception as e:
                print(f"⚠️ {name}: instantiation issue - {e}")

        return True

    except Exception as e:
        print(f"❌ Integration components test failed: {e}")
        return False

async def main():
    """Run all orchestrator integration tests"""
    print(f"🚀 Starting orchestrator integration tests at {datetime.now()}")

    tests = [
        ("Orchestrator Initialization", test_orchestrator_initialization),
        ("Social Media Investigation", test_social_media_investigation),
        ("Direct Social Media Orchestrator", test_social_media_orchestrator_direct),
        ("Phase Detection Logic", test_investigation_phase_detection),
        ("Integration Components", test_integration_components),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            if name == "Orchestrator Initialization":
                # Handle special case that returns orchestrator instance
                success, orchestrator = await test_func()
                if success:
                    print(f"✅ {name} PASSED")
                    passed += 1
                else:
                    print(f"❌ {name} FAILED")
            else:
                if await test_func():
                    print(f"✅ {name} PASSED")
                    passed += 1
                else:
                    print(f"❌ {name} FAILED")
        except Exception as e:
            print(f"❌ {name} FAILED with exception: {e}")

    print(f"\n{'='*60}")
    print(f"📊 RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All orchestrator integration tests PASSED!")
        print("✅ Enhanced Orchestrator is properly integrated with Social Media OSINT components")
        return True
    else:
        print("⚠️ Some integration tests failed - check orchestrator configuration")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)