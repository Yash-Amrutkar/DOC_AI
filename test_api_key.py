#!/usr/bin/env python3
"""
Test API Key
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key loaded: {api_key[:20]}...")

# Test the API
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
headers = {"Content-Type": "application/json"}
data = {
    "contents": [{
        "parts": [{"text": "Hello, how are you?"}]
    }]
}

try:
    response = requests.post(f"{url}?key={api_key}", headers=headers, json=data, timeout=30)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        if 'candidates' in result and result['candidates']:
            answer = result['candidates'][0]['content']['parts'][0].get('text', '')
            print(f"✅ API Test Successful!")
            print(f"Answer: {answer[:100]}...")
        else:
            print("❌ No response from API")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")
