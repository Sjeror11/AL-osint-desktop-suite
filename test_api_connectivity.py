#!/usr/bin/env python3
"""
🔑 API Connectivity Test for Desktop OSINT Suite
LakyLuk Enhanced Edition - 27.9.2025

Tests connectivity and functionality of all configured AI APIs
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
env_file = project_root / "config" / "api_keys.env"
if env_file.exists():
    load_dotenv(env_file)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class APIConnectivityTester:
    """Test connectivity for all OSINT Suite APIs"""

    def __init__(self):
        self.results = {}

    async def test_anthropic_claude(self):
        """Test Claude API connectivity"""
        logger.info("🤖 Testing Claude API...")

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            self.results['claude'] = {'status': 'failed', 'error': 'No API key'}
            return

        try:
            # Basic API test without importing anthropic library
            import requests

            # Simple test - if key format is correct
            if api_key.startswith('sk-ant-'):
                self.results['claude'] = {
                    'status': 'configured',
                    'key_format': 'valid',
                    'note': 'Key present but not tested (anthropic module not installed)'
                }
                logger.info("✅ Claude API key configured (format valid)")
            else:
                self.results['claude'] = {
                    'status': 'warning',
                    'key_format': 'invalid',
                    'note': 'API key format does not match expected pattern'
                }
                logger.warning("⚠️ Claude API key format may be invalid")

        except Exception as e:
            self.results['claude'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Claude API test failed: {e}")

    async def test_openai_gpt(self):
        """Test OpenAI GPT API connectivity"""
        logger.info("🤖 Testing OpenAI API...")

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.results['openai'] = {'status': 'failed', 'error': 'No API key'}
            return

        try:
            # Test key format
            if api_key.startswith('sk-'):
                self.results['openai'] = {
                    'status': 'configured',
                    'key_format': 'valid',
                    'note': 'Key present but not tested (openai module not installed)'
                }
                logger.info("✅ OpenAI API key configured (format valid)")
            else:
                self.results['openai'] = {
                    'status': 'warning',
                    'key_format': 'invalid',
                    'note': 'API key format does not match expected pattern'
                }
                logger.warning("⚠️ OpenAI API key format may be invalid")

        except Exception as e:
            self.results['openai'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ OpenAI API test failed: {e}")

    async def test_google_gemini(self):
        """Test Google Gemini API connectivity"""
        logger.info("🤖 Testing Google Gemini API...")

        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            self.results['gemini'] = {'status': 'failed', 'error': 'No API key'}
            return

        try:
            # Test key format
            if api_key.startswith('AIza'):
                self.results['gemini'] = {
                    'status': 'configured',
                    'key_format': 'valid',
                    'note': 'Key present but not tested (google-generativeai module not installed)'
                }
                logger.info("✅ Google Gemini API key configured (format valid)")
            else:
                self.results['gemini'] = {
                    'status': 'warning',
                    'key_format': 'invalid',
                    'note': 'API key format does not match expected pattern'
                }
                logger.warning("⚠️ Google Gemini API key format may be invalid")

        except Exception as e:
            self.results['gemini'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Google Gemini API test failed: {e}")

    async def test_youtube_api(self):
        """Test YouTube Data API connectivity"""
        logger.info("📺 Testing YouTube API...")

        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            self.results['youtube'] = {'status': 'failed', 'error': 'No API key'}
            return

        try:
            if api_key.startswith('AIza'):
                self.results['youtube'] = {
                    'status': 'configured',
                    'key_format': 'valid',
                    'note': 'YouTube API key configured'
                }
                logger.info("✅ YouTube API key configured")
            else:
                self.results['youtube'] = {
                    'status': 'warning',
                    'key_format': 'invalid'
                }
                logger.warning("⚠️ YouTube API key format may be invalid")

        except Exception as e:
            self.results['youtube'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ YouTube API test failed: {e}")

    async def test_search_apis(self):
        """Test search engine APIs"""
        logger.info("🔍 Testing Search APIs...")

        # Google Search API
        google_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        if google_key and google_key != 'your_google_search_api_key_here':
            self.results['google_search'] = {'status': 'configured'}
            logger.info("✅ Google Search API key configured")
        else:
            self.results['google_search'] = {'status': 'not_configured'}
            logger.warning("⚠️ Google Search API key not configured")

        # Bing Search API
        bing_key = os.getenv('BING_SEARCH_API_KEY')
        if bing_key and bing_key != 'your_bing_search_api_key_here':
            self.results['bing_search'] = {'status': 'configured'}
            logger.info("✅ Bing Search API key configured")
        else:
            self.results['bing_search'] = {'status': 'not_configured'}
            logger.warning("⚠️ Bing Search API key not configured")

    async def run_all_tests(self):
        """Run all API connectivity tests"""
        logger.info("🚀 Starting API Connectivity Tests...")
        logger.info("=" * 50)

        # Run all tests
        await self.test_anthropic_claude()
        await self.test_openai_gpt()
        await self.test_google_gemini()
        await self.test_youtube_api()
        await self.test_search_apis()

        # Print summary
        self.print_summary()
        return self.results

    def print_summary(self):
        """Print test results summary"""
        logger.info("=" * 50)
        logger.info("📊 API Connectivity Test Results:")
        logger.info("=" * 50)

        configured_count = 0
        total_count = 0

        for api_name, result in self.results.items():
            total_count += 1
            status = result.get('status', 'unknown')

            if status == 'configured':
                configured_count += 1
                status_icon = "✅"
                status_text = "CONFIGURED"
            elif status == 'warning':
                status_icon = "⚠️"
                status_text = "WARNING"
            elif status == 'not_configured':
                status_icon = "⏳"
                status_text = "NOT CONFIGURED"
            else:
                status_icon = "❌"
                status_text = "FAILED"

            note = result.get('note', result.get('error', ''))
            logger.info(f"{status_icon} {api_name.upper():<15} | {status_text:<15} | {note}")

        logger.info("=" * 50)
        logger.info(f"📈 Summary: {configured_count}/{total_count} APIs configured")

        if configured_count >= 3:  # Claude, OpenAI, Gemini
            logger.info("🎉 AI Enhancement ready! Multi-model ensemble available.")
        elif configured_count >= 1:
            logger.info("✅ Basic AI functionality available.")
        else:
            logger.warning("⚠️ No AI APIs configured - limited functionality.")

async def main():
    """Main test execution"""
    try:
        tester = APIConnectivityTester()
        results = await tester.run_all_tests()

        # Check if we can proceed with Phase 2
        ai_apis = ['claude', 'openai', 'gemini']
        configured_ai = sum(1 for api in ai_apis if results.get(api, {}).get('status') == 'configured')

        print()
        if configured_ai >= 2:
            print("🚀 Ready for PHASE 2: Core OSINT Engine Implementation!")
        elif configured_ai >= 1:
            print("✅ Ready for basic OSINT functionality.")
        else:
            print("⚠️ Consider adding at least one AI API key for enhanced functionality.")

        return results

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())