# 📚 RAG Document Q&A System

A professional **Retrieval-Augmented Generation (RAG)** Document Q&A System built with LangChain, FAISS, Sentence-Transformers, Ollama, and Streamlit.

## 🌟 Features

- 📄 **Multi-format Document Support** — Upload PDF, TXT, MD files
- ✂️ **Intelligent Chunking** — Recursive text splitting with metadata
- 🔢 **Vector Embeddings** — Sentence-Transformers (`all-MiniLM-L6-v2`)
- 🔍 **Semantic Search** — FAISS k-nearest neighbor retrieval (k=5)
- 🤖 **LLM Integration** — Ollama (`llama3.2`) with grounded responses
- 💬 **Chat Interface** — Modern UI with user/assistant bubbles
- 📚 **Source Citations** — Every answer includes sources with page numbers
- 📊 **Confidence Bars** — Visual relevance scores for each source
- 📤 **Chat Export** — Download conversations as Markdown
- 🗑️ **Document Management** — Upload, delete, and manage files



## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain, Streamlit |
| Vector DB | FAISS (IndexFlatL2) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| LLM | Ollama (`llama3.2`) |
| Document Parsing | PyPDF2 |
| Language | Python 3.12 |


## 📁 Project Structure


<img width="901" height="782" alt="rag_project_structure drawio" src="https://github.com/user-attachments/assets/2e5935ce-3c0c-4905-8401-7585a0e5af7b" />



## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-document-qa-system.git
cd rag-document-qa-system
```

### 2. Create Virtual Environment

bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux


### 3. Install Dependencies

bash
pip install -r requirements.txt


### 4. Install Ollama

Download from [ollama.ai/download](https://ollama.ai/download), then pull the model:

bash
ollama pull llama3.2

### 5. Run the Application

bash
# Start Ollama (in a separate terminal)
ollama serve

# Run Streamlit
streamlit run app/streamlit_app.py


### 6. Open in Browser

http://localhost:8501


## 📊 Evaluation

Run evaluation tests:

bash
python test_evaluation.py


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request


## 📄 License

This project is licensed under the MIT License.


## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence-Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.ai/)
- [Streamlit](https://streamlit.io/)

---

Built with ❤️ using Python, LangChain, FAISS, and Ollama.
