#!/usr/bin/env python3
"""
🤖🌐 AI + Browser Integration Test - Desktop OSINT Suite
LakyLuk Enhanced Edition - 27.9.2025

Test integrace AI orchestrátoru s browser automation
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

from src.core.enhanced_orchestrator import (
    EnhancedInvestigationOrchestrator,
    InvestigationTarget,
    InvestigationType,
    InvestigationPriority
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class AIBrowserIntegrationTester:
    """Test AI orchestrator + browser automation integration"""

    def __init__(self):
        self.orchestrator = None
        self.test_results = {}

    async def test_orchestrator_initialization(self):
        """Test AI orchestrator s browser automation initialization"""
        logger.info("🤖 Testing AI Orchestrator + Browser initialization...")

        try:
            self.orchestrator = EnhancedInvestigationOrchestrator()
            await self.orchestrator.initialize()

            # Check initialization status
            stats = self.orchestrator.get_ensemble_statistics()

            self.test_results['initialization'] = {
                'status': 'success',
                'available_models': len(stats.get('available_models', [])),
                'browser_available': stats.get('browser_available', False),
                'model_list': [str(model) for model in stats.get('available_models', [])]
            }

            logger.info(f"✅ Orchestrator initialized: {len(stats['available_models'])} AI models, Browser: {stats['browser_available']}")

            return True

        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            self.test_results['initialization'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    async def test_web_investigation(self):
        """Test AI-guided web investigation"""
        logger.info("🔍 Testing AI-guided web investigation...")

        if not self.orchestrator:
            logger.error("❌ Orchestrator not initialized")
            self.test_results['web_investigation'] = {
                'status': 'failed',
                'error': 'Orchestrator not initialized'
            }
            return False

        try:
            # Create test investigation target
            target = InvestigationTarget(
                name="Test Company",
                target_type=InvestigationType.BUSINESS,
                investigation_scope="basic web presence",
                priority=InvestigationPriority.NORMAL
            )

            # Test URLs for investigation
            test_urls = [
                "https://httpbin.org/html",  # Simple HTML page
                "https://quotes.toscrape.com/",  # Rich content for extraction
            ]

            logger.info(f"🎯 Investigating {len(test_urls)} URLs for '{target.name}'...")

            # Perform web investigation
            results = await self.orchestrator.perform_web_investigation(target, test_urls)

            # Analyze results
            urls_investigated = len(results.get('urls_investigated', []))
            confidence_score = results.get('confidence_score', 0.0)
            has_extracted_data = bool(results.get('extracted_data', {}))
            has_ai_analysis = bool(results.get('ai_analysis', {}))

            self.test_results['web_investigation'] = {
                'status': 'success',
                'urls_investigated': urls_investigated,
                'confidence_score': confidence_score,
                'has_extracted_data': has_extracted_data,
                'has_ai_analysis': has_ai_analysis,
                'sample_data': str(results.get('extracted_data', {}))[:200] + "..."
            }

            logger.info(f"✅ Web investigation completed:")
            logger.info(f"   📊 URLs investigated: {urls_investigated}")
            logger.info(f"   🎯 Confidence score: {confidence_score:.2f}")
            logger.info(f"   📦 Data extracted: {has_extracted_data}")
            logger.info(f"   🤖 AI analysis: {has_ai_analysis}")

            return True

        except Exception as e:
            logger.error(f"❌ Web investigation failed: {e}")
            self.test_results['web_investigation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    async def test_ai_selector_generation(self):
        """Test AI-generated CSS selector functionality"""
        logger.info("🎯 Testing AI selector generation...")

        if not self.orchestrator:
            self.test_results['ai_selectors'] = {
                'status': 'failed',
                'error': 'Orchestrator not initialized'
            }
            return False

        try:
            # Test target
            target = InvestigationTarget(
                name="John Doe",
                target_type=InvestigationType.PERSON,
                investigation_scope="social media presence",
                priority=InvestigationPriority.NORMAL
            )

            # Test AI selector generation
            selectors = await self.orchestrator._generate_ai_selectors(
                target, "https://example.com"
            )

            # Validate selectors
            has_selectors = bool(selectors)
            selector_count = len(selectors) if selectors else 0
            has_contact_selector = 'contact' in selectors if selectors else False

            self.test_results['ai_selectors'] = {
                'status': 'success',
                'has_selectors': has_selectors,
                'selector_count': selector_count,
                'has_contact_selector': has_contact_selector,
                'sample_selectors': selectors
            }

            logger.info(f"✅ AI selectors generated: {selector_count} selectors")
            logger.info(f"   📋 Sample selectors: {list(selectors.keys())[:3] if selectors else 'None'}")

            return True

        except Exception as e:
            logger.error(f"❌ AI selector generation failed: {e}")
            self.test_results['ai_selectors'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    async def test_performance_metrics(self):
        """Test performance metrics tracking"""
        logger.info("📊 Testing performance metrics...")

        if not self.orchestrator:
            self.test_results['performance_metrics'] = {
                'status': 'failed',
                'error': 'Orchestrator not initialized'
            }
            return False

        try:
            # Get statistics
            stats = self.orchestrator.get_ensemble_statistics()

            # Check web scraping stats
            web_stats = stats.get('web_scraping_stats', {})
            has_web_stats = bool(web_stats)
            total_requests = web_stats.get('total_requests', 0)

            self.test_results['performance_metrics'] = {
                'status': 'success',
                'has_web_stats': has_web_stats,
                'total_requests': total_requests,
                'web_scraping_stats': web_stats,
                'full_stats': stats
            }

            logger.info(f"✅ Performance metrics available")
            logger.info(f"   🌐 Web requests tracked: {total_requests}")
            logger.info(f"   📈 Stats available: {has_web_stats}")

            return True

        except Exception as e:
            logger.error(f"❌ Performance metrics test failed: {e}")
            self.test_results['performance_metrics'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False

    async def run_all_tests(self):
        """Spustí všechny testy AI + Browser integrace"""
        logger.info("🚀 Starting AI + Browser Integration Tests...")
        logger.info("=" * 60)

        # Test methods
        test_methods = [
            ("Orchestrator Initialization", self.test_orchestrator_initialization),
            ("AI Selector Generation", self.test_ai_selector_generation),
            ("Web Investigation", self.test_web_investigation),
            ("Performance Metrics", self.test_performance_metrics)
        ]

        successful_tests = 0
        total_tests = len(test_methods)

        for test_name, test_method in test_methods:
            try:
                logger.info(f"🧪 Running: {test_name}")
                success = await test_method()
                if success:
                    successful_tests += 1
                await asyncio.sleep(1)  # Delay between tests
            except Exception as e:
                logger.error(f"❌ Test '{test_name}' crashed: {e}")

        # Print summary
        self.print_test_summary(successful_tests, total_tests)
        return self.test_results

    def print_test_summary(self, successful_tests: int, total_tests: int):
        """Print test results summary"""
        logger.info("=" * 60)
        logger.info("🤖🌐 AI + Browser Integration Test Results:")
        logger.info("=" * 60)

        for test_name, result in self.test_results.items():
            status = result.get('status', 'unknown')

            if status == 'success':
                logger.info(f"✅ {test_name.upper():<25} | SUCCESS")

                # Additional info per test
                if test_name == 'initialization':
                    models = result.get('available_models', 0)
                    browser = result.get('browser_available', False)
                    logger.info(f"   AI Models: {models}, Browser: {'✅' if browser else '❌'}")

                elif test_name == 'web_investigation':
                    urls = result.get('urls_investigated', 0)
                    confidence = result.get('confidence_score', 0.0)
                    logger.info(f"   URLs: {urls}, Confidence: {confidence:.2f}")

                elif test_name == 'ai_selectors':
                    count = result.get('selector_count', 0)
                    logger.info(f"   Selectors generated: {count}")

                elif test_name == 'performance_metrics':
                    requests = result.get('total_requests', 0)
                    logger.info(f"   Web requests tracked: {requests}")

            else:
                error = result.get('error', 'Unknown error')
                logger.info(f"❌ {test_name.upper():<25} | FAILED: {error[:50]}...")

        logger.info("=" * 60)
        logger.info(f"📊 Results: {successful_tests}/{total_tests} tests successful")

        if successful_tests == total_tests:
            logger.info("🎉 AI + Browser automation integration fully functional!")
        elif successful_tests >= total_tests * 0.75:
            logger.info("✅ AI + Browser integration mostly working - minor issues detected")
        elif successful_tests >= 1:
            logger.info("⚠️ AI + Browser integration partially functional - needs attention")
        else:
            logger.info("❌ AI + Browser integration not working - check setup")

        logger.info("=" * 60)

async def main():
    """Main test execution"""
    try:
        tester = AIBrowserIntegrationTester()
        results = await tester.run_all_tests()

        # Save results
        results_file = Path(__file__).parent / "ai_browser_integration_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Test results saved to: {results_file}")

        # Success summary
        successful_count = sum(1 for r in results.values() if r.get('status') == 'success')
        total_count = len(results)

        print()
        if successful_count == total_count:
            print("🚀 AI + Browser integration ready for OSINT investigations!")
            return True
        elif successful_count >= total_count * 0.75:
            print("✅ AI + Browser integration mostly functional!")
            return True
        else:
            print("⚠️ AI + Browser integration needs debugging")
            return False

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(main())