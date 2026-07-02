"""
Test Ollama connection with llama3.2
"""

import requests
import subprocess

print("=" * 60)
print("OLLAMA STATUS CHECK")
print("=" * 60)

# 1. Check if Ollama process is running
print("\n🔍 Checking if Ollama is running...")
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq ollama.exe'], 
                          capture_output=True, text=True)
    if 'ollama.exe' in result.stdout:
        print("✅ Ollama process is running")
    else:
        print("⚠️  Ollama process not found, but might be running as service")
except:
    pass

# 2. Test API connection
print("\n🔄 Testing Ollama API...")
try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": "Say 'I am working!' in exactly 3 words.",
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=30
    )
    
    if response.status_code == 200:
        print("✅ Ollama API is responding!")
        print(f"Response: {response.json()['response']}")
    else:
        print(f"❌ Error: Status code {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to Ollama!")
    print("   Start it with: ollama serve")
except requests.exceptions.Timeout:
    print("❌ Ollama request timed out!")
except Exception as e:
    print(f"❌ Error: {e}")

# 3. List available models
print("\n📦 Checking available models...")
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        models = response.json().get('models', [])
        if models:
            print("✅ Available models:")
            for model in models:
                print(f"   - {model['name']} ({model['size']/1e9:.1f} GB)")
        else:
            print("⚠️  No models found. Run: ollama pull llama3.2")
    else:
        print(f"❌ Error: {response.status_code}")
except:
    print("❌ Could not fetch models")

print("\n" + "=" * 60)