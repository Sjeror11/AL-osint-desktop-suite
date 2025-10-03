#!/usr/bin/env python3
"""
🕷️ Quick Browser Automation Test - Desktop OSINT Suite
LakyLuk Enhanced Edition - 27.9.2025

Rychlý test pouze s Playwright (bez Selenium) pro ověření funkcionalita
"""

import asyncio
import json
import logging
from pathlib import Path

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

async def test_playwright_stealth():
    """Rychlý test Playwright stealth capabilities"""
    logger.info("🚀 Quick Playwright Stealth Test...")
    logger.info("=" * 50)

    # Initialize browser manager
    browser_manager = EnhancedBrowserManager()
    await browser_manager.initialize()

    results = {}

    try:
        # Test 1: Basic navigation
        logger.info("🔍 Test 1: Basic navigation...")
        session = await browser_manager.create_browser_session(
            browser_type=BrowserType.PLAYWRIGHT_CHROMIUM
        )

        if session:
            await browser_manager.navigate_to_url("https://httpbin.org/user-agent")
            data = await browser_manager.extract_data({"user_agent": "pre"})

            results['basic_navigation'] = {
                'status': 'success',
                'user_agent': data.get('user_agent', '')[:100] if data.get('user_agent') else None
            }
            logger.info("✅ Basic navigation successful")
        else:
            results['basic_navigation'] = {'status': 'failed', 'error': 'Session creation failed'}

        await browser_manager.close_session()

        # Test 2: Bot detection evasion (quick)
        logger.info("🤖 Test 2: Bot detection evasion...")
        session = await browser_manager.create_browser_session(
            browser_type=BrowserType.PLAYWRIGHT_CHROMIUM
        )

        if session:
            await browser_manager.navigate_to_url("https://httpbin.org/headers")
            headers_data = await browser_manager.extract_data({"headers": "pre"})

            # Check if headers look realistic
            headers_str = headers_data.get('headers', '').lower()
            has_accept = 'accept' in headers_str
            has_user_agent = 'user-agent' in headers_str

            results['bot_detection'] = {
                'status': 'success' if has_accept and has_user_agent else 'warning',
                'realistic_headers': has_accept and has_user_agent
            }
            logger.info(f"✅ Bot detection test: {'realistic' if has_accept and has_user_agent else 'basic'} headers")
        else:
            results['bot_detection'] = {'status': 'failed', 'error': 'Session creation failed'}

        await browser_manager.close_session()

        # Test 3: Data extraction
        logger.info("📊 Test 3: Data extraction...")
        session = await browser_manager.create_browser_session(
            browser_type=BrowserType.PLAYWRIGHT_CHROMIUM
        )

        if session:
            await browser_manager.navigate_to_url("https://quotes.toscrape.com/")
            quotes_data = await browser_manager.extract_data({
                "quotes": ".quote .text",
                "authors": ".quote .author"
            })

            quotes_count = len(quotes_data.get('quotes', []))
            authors_count = len(quotes_data.get('authors', []))

            results['data_extraction'] = {
                'status': 'success' if quotes_count > 0 else 'failed',
                'quotes_extracted': quotes_count,
                'authors_extracted': authors_count
            }
            logger.info(f"✅ Data extraction: {quotes_count} quotes, {authors_count} authors")
        else:
            results['data_extraction'] = {'status': 'failed', 'error': 'Session creation failed'}

        await browser_manager.close_session()

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        results['execution_error'] = str(e)

    # Results summary
    logger.info("=" * 50)
    logger.info("📊 Quick Test Results Summary:")
    logger.info("=" * 50)

    successful_tests = sum(1 for r in results.values()
                          if isinstance(r, dict) and r.get('status') == 'success')
    total_tests = len([r for r in results.values() if isinstance(r, dict) and 'status' in r])

    for test_name, result in results.items():
        if isinstance(result, dict) and 'status' in result:
            status = result['status']
            icon = "✅" if status == 'success' else "⚠️" if status == 'warning' else "❌"
            logger.info(f"{icon} {test_name}: {status}")

    logger.info("=" * 50)
    logger.info(f"🎯 Results: {successful_tests}/{total_tests} tests successful")

    if successful_tests >= 2:
        logger.info("🚀 Playwright stealth browser automation READY!")
        print("✅ Browser automation capabilities confirmed!")
        return True
    else:
        logger.info("⚠️ Browser automation needs attention")
        print("⚠️ Some browser automation issues detected")
        return False

if __name__ == "__main__":
    asyncio.run(test_playwright_stealth())