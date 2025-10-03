#!/usr/bin/env python3
"""
🤖 Live AI API Test for Desktop OSINT Suite
LakyLuk Enhanced Edition - 27.9.2025

Real connectivity and functionality test for all AI APIs
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

class LiveAITester:
    """Test actual AI API functionality"""

    def __init__(self):
        self.results = {}
        self.test_prompt = "What is OSINT? Respond in one sentence."

    async def test_claude_live(self):
        """Test Claude API with actual request"""
        logger.info("🤖 Testing Claude API (live)...")

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            self.results['claude'] = {'status': 'failed', 'error': 'No API key'}
            return

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)

            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": self.test_prompt
                }]
            )

            if response.content:
                self.results['claude'] = {
                    'status': 'success',
                    'response': response.content[0].text[:100] + "..." if len(response.content[0].text) > 100 else response.content[0].text,
                    'model': 'claude-3-5-sonnet'
                }
                logger.info("✅ Claude API working successfully")
            else:
                self.results['claude'] = {'status': 'failed', 'error': 'Empty response'}

        except Exception as e:
            self.results['claude'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Claude API test failed: {e}")

    async def test_openai_live(self):
        """Test OpenAI API with actual request"""
        logger.info("🤖 Testing OpenAI API (live)...")

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.results['openai'] = {'status': 'failed', 'error': 'No API key'}
            return

        try:
            import openai

            client = openai.OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model="gpt-4",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": self.test_prompt
                }]
            )

            if response.choices:
                self.results['openai'] = {
                    'status': 'success',
                    'response': response.choices[0].message.content[:100] + "..." if len(response.choices[0].message.content) > 100 else response.choices[0].message.content,
                    'model': 'gpt-4'
                }
                logger.info("✅ OpenAI API working successfully")
            else:
                self.results['openai'] = {'status': 'failed', 'error': 'Empty response'}

        except Exception as e:
            self.results['openai'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ OpenAI API test failed: {e}")

    async def test_gemini_live(self):
        """Test Gemini API with actual request"""
        logger.info("🤖 Testing Gemini API (live)...")

        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            self.results['gemini'] = {'status': 'failed', 'error': 'No API key'}
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            # Try multiple model names
            model_names = ['models/gemini-2.5-flash', 'models/gemini-2.5-pro', 'models/gemini-2.0-flash']
            model = None

            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    break
                except Exception:
                    continue

            if not model:
                model = genai.GenerativeModel('models/gemini-2.5-flash')  # Working model

            response = model.generate_content(self.test_prompt)

            if response.text:
                self.results['gemini'] = {
                    'status': 'success',
                    'response': response.text[:100] + "..." if len(response.text) > 100 else response.text,
                    'model': 'gemini-pro'
                }
                logger.info("✅ Gemini API working successfully")
            else:
                self.results['gemini'] = {'status': 'failed', 'error': 'Empty response'}

        except Exception as e:
            self.results['gemini'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Gemini API test failed: {e}")

    async def run_live_tests(self):
        """Run all live AI API tests"""
        logger.info("🚀 Starting Live AI API Tests...")
        logger.info("=" * 60)
        logger.info(f"📝 Test prompt: '{self.test_prompt}'")
        logger.info("=" * 60)

        # Run all tests
        await self.test_claude_live()
        await self.test_openai_live()
        await self.test_gemini_live()

        # Print summary
        self.print_live_summary()
        return self.results

    def print_live_summary(self):
        """Print live test results summary"""
        logger.info("=" * 60)
        logger.info("🤖 Live AI API Test Results:")
        logger.info("=" * 60)

        working_apis = []
        failed_apis = []

        for api_name, result in self.results.items():
            status = result.get('status', 'unknown')

            if status == 'success':
                working_apis.append(api_name)
                model = result.get('model', 'unknown')
                response = result.get('response', 'No response')
                logger.info(f"✅ {api_name.upper():<10} | {model:<20} | WORKING")
                logger.info(f"   Response: {response}")
            else:
                failed_apis.append(api_name)
                error = result.get('error', 'Unknown error')
                logger.info(f"❌ {api_name.upper():<10} | {'ERROR':<20} | {error}")

        logger.info("=" * 60)
        logger.info(f"📊 Results: {len(working_apis)} working, {len(failed_apis)} failed")

        if len(working_apis) >= 2:
            logger.info("🎉 Multi-model AI ensemble ready for OSINT investigations!")
        elif len(working_apis) >= 1:
            logger.info("✅ Single AI model available for basic functionality.")
        else:
            logger.info("❌ No working AI models - check API keys and internet connection.")

        logger.info("=" * 60)

async def main():
    """Main test execution"""
    try:
        tester = LiveAITester()
        results = await tester.run_live_tests()

        # Success summary
        working_count = sum(1 for r in results.values() if r.get('status') == 'success')

        print()
        if working_count >= 2:
            print("🚀 Multi-model AI ready! Phase 2 implementation can begin!")
        elif working_count >= 1:
            print("✅ Basic AI functionality confirmed!")
        else:
            print("⚠️ No working AI APIs - check keys and connectivity.")

        return results

    except Exception as e:
        logger.error(f"❌ Live test execution failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())