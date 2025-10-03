#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv

env_file = Path(__file__).parent / "config" / "api_keys.env"
load_dotenv(env_file)

try:
    import google.generativeai as genai

    api_key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=api_key)

    # Test s konkrétním modelem
    model = genai.GenerativeModel('models/gemini-2.5-flash')

    response = model.generate_content("What is OSINT? One sentence.")

    print(f"✅ SUCCESS: {response.text}")

except Exception as e:
    print(f"❌ Error: {e}")