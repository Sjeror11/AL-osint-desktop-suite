#!/usr/bin/env python3
"""
🛡️ Complete Anti-Detection Test - Desktop OSINT Suite
LakyLuk Enhanced Edition - 27.9.2025

Kompletní test anti-detection systému s proxy managementem
"""

import asyncio
import json
import logging
from pathlib import Path

# Setup project path
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from src.core.browser_manager import EnhancedBrowserManager, BrowserType, StealthLevel
from src.core.proxy_manager import AntiDetectionManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class AntiDetectionTester:
    """Test kompletní anti-detection capabilities"""

    def __init__(self):
        self.test_results = {}

    async def test_proxy_manager_standalone(self):
        """Test standalone proxy manager"""
        logger.info("🛡️ Testing Proxy Manager standalone...")

        try:
            manager = AntiDetectionManager()
            await manager.initialize()

            # Test proxy rotation
            rotation_success = await manager.rotate_proxy()

            # Test fingerprint rotation
            fingerprint_success = await manager.rotate_fingerprint()

            # Test timing patterns
            nav_delay = await manager.get_human_delay('navigation_delay')
            click_delay = await manager.get_human_delay('click_delay')

            # Get current config
            config = manager.get_current_config()
            stats = manager.get_statistics()

            self.test_results['proxy_manager'] = {
                'status': 'success',
                'proxy_rotation': rotation_success,
                'fingerprint_rotation': fingerprint_success,
                'navigation_delay': nav_delay,
                'click_delay': click_delay,
                'available_proxies': stats.get('available_proxies', 0),
                'fingerprint_profiles': stats.get('fingerprint_profiles', 0),
                'tor_available': stats.get('tor_available', False)
            }

            logger.info(f"✅ Proxy Manager: {stats['available_proxies']} proxies, {stats['fingerprint_profiles']} profiles")
            logger.info(f"   🔄 Rotation: Proxy: {rotation_success}, Fingerprint: {fingerprint_success}")
            logger.info(f"   ⏱️ Delays: Nav: {nav_delay:.2f}s, Click: {click_delay:.2f}s")
            logger.info(f"   🧅 TOR: {'Available' if stats['tor_available'] else 'Not available'}")

            return True

        except Exception as e:
            logger.error(f"❌ Proxy Manager test failed: {e}")
            self.test_results['proxy_manager'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    async def test_browser_anti_detection_integration(self):
        """Test browser manager s anti-detection integrací"""
        logger.info("🌐🛡️ Testing Browser + Anti-Detection integration...")

        try:
            # Test různé stealth úrovně
            stealth_levels = [
                StealthLevel.MINIMAL,
                StealthLevel.MODERATE,
                StealthLevel.MAXIMUM
            ]

            integration_results = {}

            for stealth_level in stealth_levels:
                logger.info(f"🔒 Testing stealth level: {stealth_level.value}")

                browser_manager = EnhancedBrowserManager(stealth_level=stealth_level)
                await browser_manager.initialize()

                # Test browser session creation
                session_created = await browser_manager.create_browser_session(
                    browser_type=BrowserType.PLAYWRIGHT_CHROMIUM
                )

                # Check anti-detection manager availability
                has_anti_detection = browser_manager.anti_detection_manager is not None

                integration_results[stealth_level.value] = {
                    'session_created': session_created,
                    'anti_detection_available': has_anti_detection,
                    'stealth_level': stealth_level.value
                }

                if session_created:
                    # Quick navigation test
                    nav_success = await browser_manager.navigate_to_url("https://httpbin.org/headers")
                    integration_results[stealth_level.value]['navigation_success'] = nav_success

                    # Extract headers to check fingerprint
                    if nav_success:
                        headers_data = await browser_manager.extract_data({"headers": "pre"})
                        has_headers = bool(headers_data.get('headers'))
                        integration_results[stealth_level.value]['headers_extracted'] = has_headers

                await browser_manager.close_session()
                await asyncio.sleep(1)

            self.test_results['browser_integration'] = {
                'status': 'success',
                'stealth_levels_tested': len(integration_results),
                'results_by_level': integration_results
            }

            # Summary
            successful_levels = sum(1 for result in integration_results.values()
                                  if result.get('session_created', False))

            logger.info(f"✅ Browser Integration: {successful_levels}/{len(stealth_levels)} stealth levels working")

            for level, result in integration_results.items():
                session_ok = "✅" if result.get('session_created') else "❌"
                anti_det_ok = "🛡️" if result.get('anti_detection_available') else "⚠️"
                logger.info(f"   {level}: {session_ok} Session, {anti_det_ok} Anti-Detection")

            return True

        except Exception as e:
            logger.error(f"❌ Browser integration test failed: {e}")
            self.test_results['browser_integration'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    async def test_stealth_effectiveness(self):
        """Test účinnosti stealth opatření"""
        logger.info("🕵️ Testing stealth effectiveness...")

        try:
            # Test s maximum stealth
            browser_manager = EnhancedBrowserManager(stealth_level=StealthLevel.MAXIMUM)
            await browser_manager.initialize()

            session_created = await browser_manager.create_browser_session(
                browser_type=BrowserType.PLAYWRIGHT_CHROMIUM
            )

            if not session_created:
                raise Exception("Failed to create browser session")

            # Test na bot detection stránce
            logger.info("🤖 Testing against bot detection...")
            nav_success = await browser_manager.navigate_to_url("https://httpbin.org/user-agent")

            if nav_success:
                # Extract user agent
                ua_data = await browser_manager.extract_data({"user_agent": "pre"})
                user_agent = ua_data.get('user_agent', '')

                # Analýza user agent na realistické znaky
                ua_realistic = (
                    'Mozilla' in user_agent and
                    'Chrome' in user_agent and
                    len(user_agent) > 50
                )

                # Test headers
                await browser_manager.navigate_to_url("https://httpbin.org/headers")
                headers_data = await browser_manager.extract_data({"headers": "pre"})
                headers_text = headers_data.get('headers', '').lower()

                headers_realistic = (
                    'accept' in headers_text and
                    'accept-language' in headers_text and
                    'user-agent' in headers_text
                )

                self.test_results['stealth_effectiveness'] = {
                    'status': 'success',
                    'user_agent_realistic': ua_realistic,
                    'headers_realistic': headers_realistic,
                    'user_agent_sample': user_agent[:100] + "..." if len(user_agent) > 100 else user_agent,
                    'overall_stealth_score': (ua_realistic + headers_realistic) / 2
                }

                logger.info(f"✅ Stealth Analysis:")
                logger.info(f"   🎭 User Agent: {'Realistic' if ua_realistic else 'Basic'}")
                logger.info(f"   📋 Headers: {'Realistic' if headers_realistic else 'Basic'}")
                logger.info(f"   🏆 Overall Score: {(ua_realistic + headers_realistic) / 2:.1f}/1.0")

            await browser_manager.close_session()
            return True

        except Exception as e:
            logger.error(f"❌ Stealth effectiveness test failed: {e}")
            self.test_results['stealth_effectiveness'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    async def run_all_tests(self):
        """Spustí všechny anti-detection testy"""
        logger.info("🚀 Starting Complete Anti-Detection Tests...")
        logger.info("=" * 60)

        test_methods = [
            ("Proxy Manager Standalone", self.test_proxy_manager_standalone),
            ("Browser Anti-Detection Integration", self.test_browser_anti_detection_integration),
            ("Stealth Effectiveness", self.test_stealth_effectiveness)
        ]

        successful_tests = 0
        total_tests = len(test_methods)

        for test_name, test_method in test_methods:
            try:
                logger.info(f"🧪 Running: {test_name}")
                success = await test_method()
                if success:
                    successful_tests += 1
                await asyncio.sleep(2)  # Delay between tests
            except Exception as e:
                logger.error(f"❌ Test '{test_name}' crashed: {e}")

        # Print summary
        self.print_test_summary(successful_tests, total_tests)
        return self.test_results

    def print_test_summary(self, successful_tests: int, total_tests: int):
        """Print test results summary"""
        logger.info("=" * 60)
        logger.info("🛡️ Complete Anti-Detection Test Results:")
        logger.info("=" * 60)

        for test_name, result in self.test_results.items():
            status = result.get('status', 'unknown')

            if status == 'success':
                logger.info(f"✅ {test_name.upper().replace('_', ' '):<30} | SUCCESS")

                # Additional info per test
                if test_name == 'proxy_manager':
                    proxies = result.get('available_proxies', 0)
                    profiles = result.get('fingerprint_profiles', 0)
                    tor = result.get('tor_available', False)
                    logger.info(f"   Proxies: {proxies}, Profiles: {profiles}, TOR: {'✅' if tor else '❌'}")

                elif test_name == 'browser_integration':
                    levels = result.get('stealth_levels_tested', 0)
                    logger.info(f"   Stealth levels tested: {levels}")

                elif test_name == 'stealth_effectiveness':
                    score = result.get('overall_stealth_score', 0.0)
                    logger.info(f"   Stealth score: {score:.1f}/1.0")

            else:
                error = result.get('error', 'Unknown error')
                logger.info(f"❌ {test_name.upper().replace('_', ' '):<30} | FAILED: {error[:40]}...")

        logger.info("=" * 60)
        logger.info(f"📊 Results: {successful_tests}/{total_tests} tests successful")

        if successful_tests == total_tests:
            logger.info("🎉 Complete anti-detection system fully operational!")
        elif successful_tests >= 2:
            logger.info("✅ Anti-detection system mostly functional - minor issues detected")
        elif successful_tests >= 1:
            logger.info("⚠️ Anti-detection system partially functional - needs attention")
        else:
            logger.info("❌ Anti-detection system not working - check setup")

        logger.info("=" * 60)

async def main():
    """Main test execution"""
    try:
        tester = AntiDetectionTester()
        results = await tester.run_all_tests()

        # Save results
        results_file = Path(__file__).parent / "anti_detection_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Test results saved to: {results_file}")

        # Success summary
        successful_count = sum(1 for r in results.values() if r.get('status') == 'success')
        total_count = len(results)

        print()
        if successful_count == total_count:
            print("🛡️ Anti-detection system ready for stealth OSINT operations!")
            return True
        elif successful_count >= 2:
            print("✅ Anti-detection system mostly ready!")
            return True
        else:
            print("⚠️ Anti-detection system needs improvement")
            return False

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(main())