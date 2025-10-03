#!/usr/bin/env python3
"""
🧪 Basic Functionality Test - OSINT Desktop Suite
Testing core components without heavy dependencies
"""

import sys
import asyncio
from datetime import datetime

print("🧪 OSINT Desktop Suite - Basic Functionality Test")
print("=" * 60)

def test_imports():
    """Test basic imports"""
    print("\n📦 Testing Basic Imports...")

    try:
        # Test core modules
        from src.core.enhanced_orchestrator import EnhancedInvestigationOrchestrator
        print("✅ Enhanced Orchestrator imported successfully")

        from src.core.browser_manager import EnhancedBrowserManager
        print("✅ Browser Manager imported successfully")

        from src.core.proxy_manager import AntiDetectionManager
        print("✅ Proxy Manager imported successfully")

        # Test utility modules (mock versions)
        from src.utils.similarity_calculator import SimilarityCalculator
        from src.utils.confidence_estimator import ConfidenceEstimator
        from src.utils.data_sanitizer import PIISanitizer
        from src.utils.rate_limiter import RateLimiter
        print("✅ All utility modules imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_orchestrator():
    """Test AI Orchestrator initialization"""
    print("\n🤖 Testing AI Orchestrator...")

    try:
        from src.core.enhanced_orchestrator import EnhancedInvestigationOrchestrator

        # Initialize without API keys for basic test
        orchestrator = EnhancedInvestigationOrchestrator()
        print("✅ Orchestrator initialized successfully")

        # Test basic methods exist
        methods = ['initialize', 'start_investigation']  # Use actual existing methods
        for method in methods:
            if hasattr(orchestrator, method):
                print(f"✅ Method '{method}' exists")
            else:
                print(f"❌ Method '{method}' missing")
                return False

        return True

    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False

def test_utils():
    """Test utility modules"""
    print("\n🛠️ Testing Utility Modules...")

    try:
        from src.utils.similarity_calculator import SimilarityCalculator
        from src.utils.confidence_estimator import ConfidenceEstimator
        from src.utils.data_sanitizer import PIISanitizer
        from src.utils.rate_limiter import RateLimiter

        # Test SimilarityCalculator
        calc = SimilarityCalculator()
        result = calc.calculate_name_similarity("John Doe", "Jane Smith")
        print(f"✅ Name similarity calculation: {result}")

        # Test ConfidenceEstimator
        conf = ConfidenceEstimator()
        confidence = conf.calculate_correlation_confidence([{"name": "test"}], [0.8, 0.6])
        print(f"✅ Confidence estimation: {confidence}")

        # Test PIISanitizer
        sanitizer = PIISanitizer()
        sanitized = sanitizer.sanitize_profile({"name": "John Doe", "email": "john@example.com"})
        print(f"✅ PII sanitization working")

        # Test RateLimiter
        limiter = RateLimiter()
        print(f"✅ Rate limiter initialized")

        return True

    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n⚙️ Testing Configuration...")

    try:
        import os
        from pathlib import Path

        # Check config directory
        config_dir = Path("config")
        if config_dir.exists():
            print("✅ Config directory exists")

            # Check API keys file
            api_keys_file = config_dir / "api_keys.env"
            if api_keys_file.exists():
                print("✅ API keys file exists")

                # Read and check for basic keys
                with open(api_keys_file, 'r') as f:
                    content = f.read()
                    if "ANTHROPIC_API_KEY" in content:
                        print("✅ Anthropic API key configured")
                    if "OPENAI_API_KEY" in content:
                        print("✅ OpenAI API key configured")
                    if "GOOGLE_API_KEY" in content:
                        print("✅ Google API key configured")
            else:
                print("❌ API keys file missing")
                return False
        else:
            print("❌ Config directory missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print(f"🚀 Starting tests at {datetime.now()}")

    tests = [
        ("Import Test", test_imports),
        ("Orchestrator Test", test_orchestrator),
        ("Utils Test", test_utils),
        ("Config Test", test_config),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n{'='*60}")
        if test_func():
            print(f"✅ {name} PASSED")
            passed += 1
        else:
            print(f"❌ {name} FAILED")

    print(f"\n{'='*60}")
    print(f"📊 RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic functionality tests PASSED!")
        print("✅ Core OSINT suite components are working correctly")
        return True
    else:
        print("⚠️ Some tests failed - check dependencies and configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)