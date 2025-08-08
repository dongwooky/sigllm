#!/usr/bin/env python3
"""
Together AI Connection Test
Simple test to verify Together AI API is working correctly
"""

import os
from together import Together

def test_together_ai():
    """Test Together AI API connection"""
    
    # Check API key
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("❌ TOGETHER_API_KEY environment variable not set!")
        print("Please set your API key: export TOGETHER_API_KEY='your-api-key'")
        print("Get an API key from: https://api.together.xyz/")
        return False
    
    print("✅ TOGETHER_API_KEY found")
    
    try:
        # Initialize client
        client = Together()
        print("✅ Together AI client initialized")
        
        # Test simple completion
        print("🧪 Testing API call...")
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that performs time series predictions."
                },
                {
                    "role": "user",
                    "content": "Please continue the following sequence: 1,2,3,4,5,"
                }
            ],
            max_tokens=20,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"✅ API call successful!")
        print(f"📝 Response: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Together AI Connection Test ===")
    success = test_together_ai()
    
    if success:
        print("\n🎉 Together AI is ready to use!")
        print("You can now run: python tutorials/pipelines/mistral-detector-pipeline_together.py")
    else:
        print("\n⚠️  Please fix the issues above before running the main pipeline.")
