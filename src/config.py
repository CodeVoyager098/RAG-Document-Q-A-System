"""
Configuration settings for the RAG Document Q&A System - FINAL OPTIMIZED VERSION
"""

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# Chunking parameters - OPTIMIZED for speed and quality
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_DIMENSION = 384

# Retrieval parameters - BALANCED for speed and coverage
TOP_K_RESULTS = 10  # Reduced from 8 to 5 for faster responses

# Ollama Configuration
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# Supported file types
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}

# LLM Configuration
LLM_TEMPERATURE = 0.1
MAX_RESPONSE_TOKENS = 2500
OLLAMA_TIMEOUT = 300  # Increased timeout for Ollama