📄 Complete README.md
markdown
# 📚 RAG Document Q&A System

A professional **Retrieval-Augmented Generation (RAG)** Document Q&A System built with LangChain, FAISS, Sentence-Transformers, Ollama, and Streamlit.

## 🌟 Features

- 📄 **Multi-format Document Support** - Upload PDF, TXT, MD files
- ✂️ **Intelligent Chunking** - Recursive text splitting with metadata
- 🔢 **Vector Embeddings** - Sentence-Transformers (all-MiniLM-L6-v2)
- 🔍 **Semantic Search** - FAISS k-nearest neighbor retrieval (k=5)
- 🤖 **LLM Integration** - Ollama (llama3.2) with grounded responses
- 💬 **Chat Interface** - Modern UI with user/assistant bubbles
- 📚 **Source Citations** - Every answer includes sources with page numbers
- 📊 **Confidence Bars** - Visual relevance scores for each source
- 📤 **Chat Export** - Download conversations as Markdown
- 🗑️ **Document Management** - Upload, delete, and manage files

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain, Streamlit |
| Vector DB | FAISS (IndexFlatL2) |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| LLM | Ollama (llama3.2) |
| Document Parsing | PyPDF2 |
| Language | Python 3.12 |

## 📁 Project Structure
rag-document-qa/
├── app/
│ └── streamlit_app.py # Main UI
├── src/
│ ├── config.py # Configuration
│ ├── document_loader.py # PDF/TXT/MD loader
│ ├── text_processor.py # Text chunking
│ ├── embedding_indexer.py # Embeddings & FAISS
│ ├── retriever.py # Search & retrieval
│ ├── llm_responder.py # LLM grounding
│ └── evaluator.py # Evaluation metrics
├── data/ # Uploaded documents
├── vectorstore/ # FAISS index
├── requirements.txt # Dependencies
└── README.md

text

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-document-qa-system.git
cd rag-document-qa-system
2. Create Virtual Environment
bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
3. Install Dependencies
bash
pip install -r requirements.txt
4. Install Ollama
bash
# Download from: https://ollama.ai/download
ollama pull llama3.2
5. Run the Application
bash
# Start Ollama (in a separate terminal)
ollama serve

# Run Streamlit
streamlit run app/streamlit_app.py
6. Open in Browser
text
http://localhost:8501
📊 Evaluation
Run evaluation tests:

bash
python test_evaluation.py
🤝 Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

📄 License
This project is licensed under the MIT License.

🙏 Acknowledgments
LangChain

FAISS

Sentence-Transformers

Ollama

Streamlit

Built with ❤️ using Python, LangChain, FAISS, and Ollama