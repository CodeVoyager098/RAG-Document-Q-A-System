📄 Complete documentation.md
markdown
# 📚 RAG Document Q&A System - Technical Documentation

## Architecture Overview

The RAG (Retrieval-Augmented Generation) Document Q&A System is built using a modular architecture that processes documents, creates searchable embeddings, and generates grounded responses using a local LLM.

### System Architecture Diagram
<img width="1081" height="501" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/a28982f4-8e71-4783-af58-b2d3a00b2609" />


---

## Module Descriptions

### 1. Document Loader Module (`document_loader.py`)

**Purpose**: Extract text from uploaded documents

**Supported Formats**:
- PDF (.pdf) - Using PyPDF2
- Text (.txt) - UTF-8 encoding support
- Markdown (.md) - With markdown cleaning

**Key Functions**:
- `load_document()` - Loads a single document
- `load_multiple_documents()` - Loads multiple documents
- `load_directory()` - Loads all documents in a directory

**Metadata Extraction**:
- Source file name
- File type
- Page numbers (for PDFs)
- Character and word count

---

### 2. Text Processor Module (`text_processor.py`)

**Purpose**: Clean and chunk text for embedding

**Chunking Strategy**:
- **Method**: RecursiveCharacterTextSplitter
- **Chunk Size**: 600 characters
- **Chunk Overlap**: 100 characters
- **Separators**: Paragraph breaks, sentences, words

**Why This Strategy**:
- Preserves semantic meaning
- Maintains context between chunks
- Balances between retrieval accuracy and speed

**Text Cleaning**:
- Removes extra whitespace
- Normalizes quotes and punctuation
- Removes page numbers and headers/footers
- Handles special characters

---

### 3. Embedding & Indexing Module (`embedding_indexer.py`)

**Purpose**: Create vector embeddings and build FAISS index

**Embedding Model**:
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimension**: 384
- **Why**: Balances performance and accuracy

**FAISS Index**:
- **Type**: IndexFlatL2 (Flat L2 distance)
- **Why**: Exact nearest neighbor search
- **Persistence**: Saves index and chunks to disk

**Key Functions**:
- `create_embeddings()` - Generate embeddings
- `build_index()` - Build FAISS index
- `search()` - Search for similar chunks
- `save_index()` - Save to disk
- `load_index()` - Load from disk

---

### 4. Retrieval Module (`retriever.py`)

**Purpose**: Retrieve relevant chunks for a query

**Retrieval Parameters**:
- **k**: 5 (number of chunks to retrieve)
- **Metric**: L2 distance (Euclidean)
- **Score**: Lower is better (distance)

**Context Formatting**:
- Includes source file name
- Includes page number
- Includes relevance score
- Combines chunks with clear separation

---

### 5. LLM Responder Module (`llm_responder.py`)

**Purpose**: Generate grounded responses using Ollama

**LLM Configuration**:
- **Model**: llama3.2 (via Ollama)
- **Temperature**: 0.1 (for deterministic responses)
- **Max Tokens**: 3000
- **Timeout**: 300 seconds

**Prompt Engineering**:
- **Strict Grounding**: "Answer ONLY from the context"
- **List Extraction**: Forces all items in a list
- **Citation Enforcement**: "Cite source: (filename, Page X)"
- **Fallback**: "I don't know" when context is insufficient

---

### 6. Evaluation Module (`evaluator.py`)

**Purpose**: Evaluate system performance

**Metrics**:
- **Faithfulness**: Does the answer appear in the context?
- **Relevance**: Is the answer relevant to the question?
- **Source Accuracy**: Does it cite the correct source?

**QA Set**:
- Manually created golden QA pairs
- Customizable for different documents

---

## Design Decisions

### 1. Why FAISS?
- ✅ Fast similarity search
- ✅ Handles large datasets
- ✅ Works with 384-dim embeddings

### 2. Why Sentence-Transformers?
- ✅ Pre-trained on semantic similarity
- ✅ Good balance of speed and accuracy
- ✅ 384-dim is lightweight

### 3. Why Ollama?
- ✅ Runs locally (no API costs)
- ✅ Private and secure
- ✅ No internet required

### 4. Why Streamlit?
- ✅ Rapid prototyping
- ✅ Python-based UI
- ✅ Easy deployment

---

## Data Flow
User Uploads Document
↓

Document Loader Extracts Text
↓

Text Processor Chunks Text
↓

Embedding Indexer Creates Embeddings
↓

FAISS Index Created and Saved
↓

User Asks Question
↓

Retriever Finds Relevant Chunks
↓

LLM Responder Generates Answer
↓

Response Displayed with Citations

text

---

## Performance Optimizations

1. **Chunk Size**: 600 characters (optimized for retrieval speed)
2. **Batch Processing**: Embeddings created in batches
3. **Caching**: FAISS index persisted to disk
4. **Timeout**: 300 seconds for LLM responses

---

## Future Improvements

- [ ] User authentication
- [ ] Multi-language support
- [ ] Web scraping for URL documents
- [ ] Cloud deployment (Streamlit Cloud)
- [ ] Advanced analytics dashboard
- [ ] Voice input support

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Ollama not responding | Start Ollama: `ollama serve` |
| FAISS index not found | Run: `python rebuild_final_index.py` |
| Import errors | Activate venv: `venv\Scripts\activate` |
| Timeout errors | Increase timeout in config |
| "I don't know" responses | Check if document contains answer |

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Ollama Documentation](https://ollama.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
