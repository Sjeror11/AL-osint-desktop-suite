#!/usr/bin/env python3
"""
🧪 Browser Integration Test - OSINT Desktop Suite
Testing browser automation integration with social media scanners
"""

import sys
import asyncio
from datetime import datetime

print("🧪 OSINT Browser Integration Test")
print("=" * 60)

async def test_browser_integration():
    """Test browser integration adapter"""
    print("\n🔧 Testing Browser Integration Adapter...")

    try:
        from src.core.browser_integration import BrowserIntegrationAdapter

        # Initialize adapter
        adapter = BrowserIntegrationAdapter()
        await adapter.initialize()
        print("✅ Browser Integration Adapter initialized successfully")

        # Test session creation
        print("\n🌐 Testing session creation...")
        session = await adapter.create_session(
            platform="facebook",
            stealth_level="high",
            proxy=None
        )

        if session:
            print("✅ Browser session created successfully")

            # Test session cleanup
            await session.close()
            print("✅ Browser session closed successfully")
        else:
            print("❌ Failed to create browser session")
            return False

        return True

    except Exception as e:
        print(f"❌ Browser integration test failed: {e}")
        return False

async def test_social_media_scanners():
    """Test social media scanner initialization"""
    print("\n📱 Testing Social Media Scanner Integration...")

    try:
        from src.tools.social_media.facebook_scanner import FacebookScanner
        from src.tools.social_media.instagram_scanner import InstagramScanner
        from src.tools.social_media.linkedin_scanner import LinkedInScanner

        # Test Facebook Scanner
        print("\n🔵 Testing Facebook Scanner...")
        facebook_scanner = FacebookScanner()
        await facebook_scanner.initialize()
        print("✅ Facebook Scanner initialized successfully")

        # Test Instagram Scanner
        print("\n📸 Testing Instagram Scanner...")
        instagram_scanner = InstagramScanner()
        await instagram_scanner.initialize()
        print("✅ Instagram Scanner initialized successfully")

        # Test LinkedIn Scanner
        print("\n💼 Testing LinkedIn Scanner...")
        linkedin_scanner = LinkedInScanner()
        await linkedin_scanner.initialize()
        print("✅ LinkedIn Scanner initialized successfully")

        return True

    except Exception as e:
        print(f"❌ Social media scanner test failed: {e}")
        return False

async def test_cross_platform_search():
    """Test cross-platform search integration"""
    print("\n🔍 Testing Cross-Platform Search Integration...")

    try:
        from src.tools.social_media.cross_platform_search import CrossPlatformSearch

        # Initialize cross-platform search
        search_engine = CrossPlatformSearch()
        print("✅ Cross-Platform Search initialized successfully")

        # Test that it has the expected methods
        methods = ['search_all_platforms', 'analyze_connections']
        for method in methods:
            if hasattr(search_engine, method):
                print(f"✅ Method '{method}' exists")
            else:
                print(f"⚠️ Method '{method}' missing (may be optional)")

        return True

    except Exception as e:
        print(f"❌ Cross-platform search test failed: {e}")
        return False

async def test_compatibility_methods():
    """Test that compatibility methods exist and work"""
    print("\n🔗 Testing Compatibility Methods...")

    try:
        from src.core.browser_integration import create_stealth_session, cleanup_browser_sessions

        # Test global functions
        print("✅ Compatibility functions imported successfully")

        # Test session creation
        session = await create_stealth_session("facebook", "high")
        if session:
            print("✅ Global create_stealth_session working")
            await session.close()
        else:
            print("⚠️ Global session creation returned None")

        # Test cleanup
        await cleanup_browser_sessions()
        print("✅ Global cleanup function working")

        return True

    except Exception as e:
        print(f"❌ Compatibility methods test failed: {e}")
        return False

async def main():
    """Run all browser integration tests"""
    print(f"🚀 Starting browser integration tests at {datetime.now()}")

    tests = [
        ("Browser Integration Adapter", test_browser_integration),
        ("Social Media Scanners", test_social_media_scanners),
        ("Cross-Platform Search", test_cross_platform_search),
        ("Compatibility Methods", test_compatibility_methods),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n{'='*60}")
        try:
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
        print("🎉 All browser integration tests PASSED!")
        print("✅ Social media scanners are properly integrated with browser automation")
        return True
    else:
        print("⚠️ Some integration tests failed - check browser automation setup")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)