"""
Complete setup verification
"""

import sys
import subprocess
import requests
from pathlib import Path

print("=" * 60)
print("COMPLETE RAG PROJECT SETUP VERIFICATION")
print("=" * 60)

# 1. Check Python
print(f"\n🐍 Python version: {sys.version}")

# 2. Check environment
if hasattr(sys, 'prefix'):
    if sys.prefix != sys.base_prefix:
        print(f"✅ Virtual environment: {sys.prefix}")
    else:
        print(f"⚠️  System Python: {sys.prefix}")

# 3. Check key packages
print("\n📦 Checking installed packages...")
packages = ["langchain", "langchain_community", "langchain_text_splitters", 
            "faiss", "sentence_transformers", "streamlit", "PyPDF2", 
            "requests", "pandas", "numpy"]

for pkg in packages:
    try:
        __import__(pkg.replace("-", "_"))
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg}")

# 4. Check Ollama
print("\n🤖 Checking Ollama...")
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    print("✅ Ollama is available")
    print("Available models:")
    print(result.stdout)
except Exception as e:
    print(f"❌ Ollama not found: {e}")

# 5. Test Ollama API
print("\n🔄 Testing Ollama API with llama3.2...")
try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": "Say 'I am working!' in 3 words.",
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Ollama is working!")
        print(f"   Response: {result['response']}")
    else:
        print(f"❌ Ollama returned status: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to Ollama!")
    print("   Start Ollama with: ollama serve")
except Exception as e:
    print(f"❌ Error: {e}")

# 6. Check project structure
print("\n📁 Checking project structure...")
project_files = [
    "src/config.py",
    "src/__init__.py",
    "data/",
    "vectorstore/"
]

for item in project_files:
    path = Path(item)
    if path.exists():
        print(f"  ✅ {item}")
    else:
        print(f"  ❌ {item} missing")

print("\n" + "=" * 60)