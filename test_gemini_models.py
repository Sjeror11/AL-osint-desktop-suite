#!/usr/bin/env python3
"""
🔍 Test dostupných Gemini modelů
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent / "config" / "api_keys.env"
if env_file.exists():
    load_dotenv(env_file)

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("❌ No Google API key found")
    exit(1)

try:
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    print("🔍 Listing available Gemini models...")

    models = genai.list_models()
    print("\n📋 Available models:")

    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            print(f"✅ {model.name} - {model.display_name}")
        else:
            print(f"❌ {model.name} - {model.display_name} (no generateContent)")

    print("\n🤖 Testing first available model...")

    # Try first available model
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            try:
                test_model = genai.GenerativeModel(model.name)
                response = test_model.generate_content("What is OSINT? One sentence.")

                if response.text:
                    print(f"✅ SUCCESS with {model.name}")
                    print(f"Response: {response.text}")
                    break

            except Exception as e:
                print(f"❌ Failed with {model.name}: {e}")
                continue

except Exception as e:
    print(f"❌ Error: {e}")