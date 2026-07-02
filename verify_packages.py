"""
Verify all packages are installed correctly
"""

import sys
import subprocess
import importlib

print("=" * 60)
print("VERIFYING INSTALLED PACKAGES")
print("=" * 60)

# Package names: (import_name, display_name)
packages = [
    ("langchain", "langchain"),
    ("langchain_community", "langchain-community"),
    ("langchain_text_splitters", "langchain-text-splitters"),
    ("faiss", "faiss-cpu"),
    ("sentence_transformers", "sentence-transformers"),
    ("streamlit", "streamlit"),
    ("PyPDF2", "PyPDF2"),
    ("dotenv", "python-dotenv"),
    ("tiktoken", "tiktoken"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("requests", "requests")
]

print("\nChecking installed packages:")
print("-" * 40)

installed = []
missing = []
error_packages = []

for import_name, display_name in packages:
    try:
        module = importlib.import_module(import_name)
        print(f"✅ {display_name} - installed (version: {getattr(module, '__version__', 'unknown')})")
        installed.append(display_name)
    except ImportError as e:
        print(f"❌ {display_name} - NOT installed")
        missing.append(display_name)
    except Exception as e:
        print(f"⚠️  {display_name} - installed but has issues: {str(e)[:50]}...")
        error_packages.append(display_name)

print("\n" + "-" * 40)
print(f"Total: {len(installed)} installed, {len(missing)} missing, {len(error_packages)} with issues")

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("   Run: pip install -r requirements.txt")

if error_packages:
    print(f"\n⚠️  Packages with issues: {', '.join(error_packages)}")
    print("   These packages are installed but have import problems.")

print("\n" + "=" * 60)

# Check if we're in virtual environment
if hasattr(sys, 'prefix'):
    if sys.prefix != sys.base_prefix:
        print(f"✅ Running in virtual environment: {sys.prefix}")
    else:
        print(f"⚠️  Not in virtual environment (using system Python: {sys.prefix})")

# Check Ollama
print("\n🤖 Checking Ollama installation...")
try:
    result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Ollama {result.stdout.strip()} installed")
    else:
        print("❌ Ollama not found")
except FileNotFoundError:
    print("❌ Ollama not found in PATH")

print("=" * 60)