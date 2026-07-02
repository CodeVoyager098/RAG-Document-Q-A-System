#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Additional packages
pip install PyPDF2 langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers pandas numpy requests python-dotenv tiktoken

echo "✅ All dependencies installed successfully!"