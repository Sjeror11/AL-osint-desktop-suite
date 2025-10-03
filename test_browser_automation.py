#!/usr/bin/env python3
"""
🕷️ Test Browser Automation - Desktop OSINT Suite
LakyLuk Enhanced Edition - 27.9.2025

Testuje stealth web scraping capabilities našeho Browser Manager systému
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Setup project path
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from src.core.browser_manager import EnhancedBrowserManager, BrowserType, ScrapingTarget

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class BrowserAutomationTester:
    """Test stealth web scraping funkcionalita"""

    def __init__(self):
        self.browser_manager = EnhancedBrowserManager()
        self.test_results = {}
        self.initialized = False

    async def test_basic_navigation(self):
        """Test základní navigace a stealth funkcionalita"""
        logger.info("🔍 Testing basic stealth navigation...")

        try:
            # Initialize browser manager if not done yet
            if not self.initialized:
                await self.browser_manager.initialize()
                self.initialized = True

            # Test s httpbin.org - poskytuje informace o requestu
            session = await self.browser_manager.create_browser_session(
                browser_type=BrowserType.PLAYWRIGHT_CHROMIUM
            )

            # Navigace na httpbin.org/headers
            await self.browser_manager.navigate_to_url("https://httpbin.org/headers")

            # Extrahování headers
            headers_data = await self.browser_manager.extract_data({
                "headers": "pre"
            })

            logger.info("✅ Basic navigation successful")
            logger.info(f"📊 Headers detected: {headers_data.get('headers', 'N/A')[:200]}...")

            self.test_results['basic_navigation'] = {
                'status': 'success',
                'headers_detected': bool(headers_data.get('headers'))
            }

            await self.browser_manager.close_session()

        except Exception as e:
            logger.error(f"❌ Basic navigation test failed: {e}")
            self.test_results['basic_navigation'] = {
                'status': 'failed',
                'error': str(e)
            }

    async def test_bot_detection_evasion(self):
        """Test anti-bot detection"""
        logger.info("🤖 Testing bot detection evasion...")

        try:
            session = await self.browser_manager.create_browser_session(
                browser_type=BrowserType.PLAYWRIGHT_CHROMIUM
            )

            # Test na bot detection službu
            await self.browser_manager.navigate_to_url("https://bot.sannysoft.com/")

            # Čekání na načtení stránky
            await asyncio.sleep(5)

            # Extrahování bot detection výsledků
            detection_data = await self.browser_manager.extract_data({
                "webdriver": "[data-name='webdriver']",
                "chrome": "[data-name='chrome']",
                "permissions": "[data-name='permissions']",
                "plugins": "[data-name='plugins']"
            })

            logger.info("✅ Bot detection test completed")

            # Hodnocení stealth úspěchu
            failed_checks = sum(1 for key, value in detection_data.items()
                              if value and 'failed' in value.lower())

            self.test_results['bot_detection'] = {
                'status': 'success',
                'failed_checks': failed_checks,
                'detection_data': detection_data
            }

            await self.browser_manager.close_session()

        except Exception as e:
            logger.error(f"❌ Bot detection test failed: {e}")
            self.test_results['bot_detection'] = {
                'status': 'failed',
                'error': str(e)
            }

    async def test_data_extraction(self):
        """Test pokročilého data extraction"""
        logger.info("📊 Testing advanced data extraction...")

        try:
            session = await self.browser_manager.create_browser_session(
                browser_type=BrowserType.PLAYWRIGHT_CHROMIUM
            )

            # Test na stránce s různými daty
            await self.browser_manager.navigate_to_url("https://quotes.toscrape.com/")

            # Extrahování citátů
            quotes_data = await self.browser_manager.extract_data({
                "quotes": ".quote .text",
                "authors": ".quote .author",
                "tags": ".quote .tags a"
            })

            logger.info(f"✅ Data extraction successful")
            logger.info(f"📝 Extracted {len(quotes_data.get('quotes', []))} quotes")

            self.test_results['data_extraction'] = {
                'status': 'success',
                'quotes_count': len(quotes_data.get('quotes', [])),
                'authors_count': len(quotes_data.get('authors', [])),
                'sample_quote': quotes_data.get('quotes', [''])[0][:100] if quotes_data.get('quotes') else None
            }

            await self.browser_manager.close_session()

        except Exception as e:
            logger.error(f"❌ Data extraction test failed: {e}")
            self.test_results['data_extraction'] = {
                'status': 'failed',
                'error': str(e)
            }

    async def test_human_behavior_simulation(self):
        """Test human-like behavior simulation"""
        logger.info("👤 Testing human behavior simulation...")

        try:
            session = await self.browser_manager.create_browser_session(
                browser_type=BrowserType.PLAYWRIGHT_CHROMIUM
            )

            # Navigace s human behavior
            await self.browser_manager.navigate_to_url("https://httpbin.org/delay/2")

            # Simulace mouse movement a scrolling
            if hasattr(self.browser_manager, 'simulate_human_behavior'):
                await self.browser_manager.simulate_human_behavior()

            # Random delay simulation
            await asyncio.sleep(2)

            logger.info("✅ Human behavior simulation completed")

            self.test_results['human_behavior'] = {
                'status': 'success',
                'behavior_simulated': True
            }

            await self.browser_manager.close_session()

        except Exception as e:
            logger.error(f"❌ Human behavior test failed: {e}")
            self.test_results['human_behavior'] = {
                'status': 'failed',
                'error': str(e)
            }

    async def test_selenium_fallback(self):
        """Test Selenium fallback funkcionalita"""
        logger.info("🔄 Testing Selenium fallback...")

        try:
            session = await self.browser_manager.create_browser_session(
                browser_type=BrowserType.SELENIUM_CHROME
            )

            # Test základní funkcionalita se Selenium
            await self.browser_manager.navigate_to_url("https://httpbin.org/user-agent")

            user_agent_data = await self.browser_manager.extract_data({
                "user_agent": "pre"
            })

            logger.info("✅ Selenium fallback working")
            logger.info(f"🕵️ User agent: {user_agent_data.get('user_agent', 'N/A')[:100]}...")

            self.test_results['selenium_fallback'] = {
                'status': 'success',
                'user_agent_detected': bool(user_agent_data.get('user_agent'))
            }

            await self.browser_manager.close_session()

        except Exception as e:
            logger.error(f"❌ Selenium fallback test failed: {e}")
            self.test_results['selenium_fallback'] = {
                'status': 'failed',
                'error': str(e)
            }

    async def run_all_tests(self):
        """Spustí všechny testy browser automation"""
        logger.info("🚀 Starting Browser Automation Tests...")
        logger.info("=" * 60)

        # Postupné spuštění testů
        test_methods = [
            self.test_basic_navigation,
            self.test_bot_detection_evasion,
            self.test_data_extraction,
            self.test_human_behavior_simulation,
            self.test_selenium_fallback
        ]

        for test_method in test_methods:
            try:
                await test_method()
                await asyncio.sleep(2)  # Delay mezi testy
            except Exception as e:
                logger.error(f"❌ Test {test_method.__name__} crashed: {e}")

        # Výsledky
        self.print_test_summary()
        return self.test_results

    def print_test_summary(self):
        """Vytiskne souhrn testů"""
        logger.info("=" * 60)
        logger.info("🕷️ Browser Automation Test Results:")
        logger.info("=" * 60)

        successful_tests = []
        failed_tests = []

        for test_name, result in self.test_results.items():
            status = result.get('status', 'unknown')

            if status == 'success':
                successful_tests.append(test_name)
                logger.info(f"✅ {test_name.upper():<25} | SUCCESS")

                # Dodatečné info
                if test_name == 'bot_detection':
                    failed_checks = result.get('failed_checks', 0)
                    logger.info(f"   Failed detection checks: {failed_checks}")
                elif test_name == 'data_extraction':
                    quotes_count = result.get('quotes_count', 0)
                    logger.info(f"   Extracted quotes: {quotes_count}")

            else:
                failed_tests.append(test_name)
                error = result.get('error', 'Unknown error')
                logger.info(f"❌ {test_name.upper():<25} | FAILED: {error[:50]}...")

        logger.info("=" * 60)
        logger.info(f"📊 Results: {len(successful_tests)} successful, {len(failed_tests)} failed")

        if len(successful_tests) >= 3:
            logger.info("🎉 Browser automation ready for OSINT investigations!")
        elif len(successful_tests) >= 1:
            logger.info("✅ Basic browser automation functional.")
        else:
            logger.info("❌ Browser automation needs attention - check dependencies.")

        logger.info("=" * 60)

async def main():
    """Hlavní test execution"""
    try:
        tester = BrowserAutomationTester()
        results = await tester.run_all_tests()

        # Save results to file
        results_file = Path(__file__).parent / "browser_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 Test results saved to: {results_file}")

        # Success summary
        successful_count = sum(1 for r in results.values() if r.get('status') == 'success')

        print()
        if successful_count >= 4:
            print("🚀 Advanced browser automation ready! OSINT capabilities unlocked!")
        elif successful_count >= 2:
            print("✅ Basic browser automation confirmed!")
        else:
            print("⚠️ Browser automation needs debugging - check setup.")

        return results

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())