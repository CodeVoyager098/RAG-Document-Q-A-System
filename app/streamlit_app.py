"""
Streamlit UI for RAG Document Q&A System - WITH ENHANCED CONFIDENCE BARS
"""

import streamlit as st
import time
import shutil
from pathlib import Path
import sys
import tempfile
import os
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.document_loader import DocumentLoader
from src.text_processor import TextProcessor
from src.embedding_indexer import EmbeddingIndexer
from src.retriever import Retriever
from src.llm_responder import LLMResponder
from src.config import (
    DATA_DIR, 
    VECTORSTORE_DIR,
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    OLLAMA_MODEL,
    SUPPORTED_EXTENSIONS
)

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== MODERN CSS STYLING =====
st.markdown("""
<style>
    /* ===== PROFESSIONAL COLOR SCHEME ===== */
    :root {
        --primary: #2563eb;
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --secondary: #7c3aed;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --background: #f1f5f9;
        --card-bg: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-light: #94a3b8;
        --border: #e2e8f0;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --radius: 12px;
        --radius-sm: 8px;
        --radius-lg: 16px;
    }
    
    /* ===== HEADER STYLING - BLACK ===== */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #000000 !important;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        -webkit-text-fill-color: #000000 !important;
        background: none !important;
        -webkit-background-clip: unset !important;
        background-clip: unset !important;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* ===== CHAT BUBBLES - MODERN ===== */
    .user-message {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 4px 16px;
        margin-bottom: 0.75rem;
        color: #ffffff;
        border-left: none;
        box-shadow: var(--shadow);
        max-width: 85%;
        margin-left: auto;
        animation: slideInRight 0.3s ease;
    }
    
    .user-message strong {
        color: #bfdbfe;
        font-size: 0.8rem;
        display: block;
        margin-bottom: 0.2rem;
    }
    
    .assistant-message {
        background: var(--card-bg);
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 16px 4px;
        margin-bottom: 0.75rem;
        white-space: pre-wrap;
        line-height: 1.7;
        border-left: 4px solid #10b981;
        color: var(--text-primary);
        box-shadow: var(--shadow);
        max-width: 85%;
        margin-right: auto;
        border: 1px solid var(--border);
        animation: slideInLeft 0.3s ease;
    }
    
    .assistant-message strong {
        color: #10b981;
        font-size: 0.8rem;
        display: block;
        margin-bottom: 0.2rem;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 60%, 100% { opacity: 0.3; }
        30% { opacity: 1; }
    }
    
    /* ===== TYPING INDICATOR ===== */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        background: var(--card-bg);
        border-radius: 16px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        max-width: 120px;
        margin-bottom: 0.75rem;
    }
    
    .typing-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #94a3b8;
        animation: pulse 1.5s infinite;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    /* ===== ENHANCED CONFIDENCE BARS ===== */
    .source-item {
        padding: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        transition: background 0.2s ease;
        border-radius: 8px;
    }
    
    .source-item:hover {
        background: #f8fafc;
    }
    
    .source-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.3rem;
    }
    
    .source-name {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    
    .source-score {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        color: white;
    }
    
    .source-score.high { background: #10b981; }
    .source-score.medium { background: #f59e0b; }
    .source-score.low { background: #ef4444; }
    
    .source-page {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.1rem;
    }
    
    .source-preview {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.3rem;
        font-style: italic;
        line-height: 1.4;
    }
    
    /* ===== CONFIDENCE BAR CONTAINER ===== */
    .confidence-container {
        margin-top: 0.5rem;
        padding: 0.3rem 0;
    }
    
    .confidence-bar-track {
        height: 6px;
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
        width: 0%;
    }
    
    .confidence-bar-fill.high {
        background: linear-gradient(90deg, #10b981, #34d399);
    }
    
    .confidence-bar-fill.medium {
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
    }
    
    .confidence-bar-fill.low {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }
    
    .confidence-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: var(--text-light);
        margin-top: 0.2rem;
    }
    
    .confidence-labels span:last-child {
        font-weight: 600;
        color: var(--text-secondary);
    }
    
    /* ===== INFO & SUCCESS BOXES ===== */
    .success-box {
        padding: 1rem 1.25rem;
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 1px solid #6ee7b7;
        border-radius: var(--radius);
        color: #065f46;
        box-shadow: var(--shadow);
    }
    
    .info-box {
        padding: 1rem 1.25rem;
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        border: 1px solid #93c5fd;
        border-radius: var(--radius);
        color: #1e40af;
        box-shadow: var(--shadow);
    }
    
    /* ===== SIDEBAR STYLING ===== */
    .stSidebar {
        background: #ffffff !important;
        border-right: 1px solid var(--border) !important;
    }
    
    .stSidebar .stMarkdown h1,
    .stSidebar .stMarkdown h2,
    .stSidebar .stMarkdown h3,
    .stSidebar .stMarkdown h4 {
        color: var(--text-primary) !important;
    }
    
    /* ===== BUTTONS - ORANGE ===== */
    .stButton > button {
        background: #ff4b4b !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        box-shadow: var(--shadow) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
        background: #e04343 !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    .stFileUploader > div > div {
        border: 2px dashed #cbd5e1 !important;
        border-radius: var(--radius) !important;
        background: #f8fafc !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #ff4b4b !important;
        background: #fff5f5 !important;
    }
    
    /* ===== METRICS ===== */
    .stMetric {
        background: var(--card-bg);
        padding: 0.75rem;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }
    
    .stMetric label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    .stMetric div {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--card-bg) !important;
        border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
    }
    
    /* ===== DIVIDER ===== */
    .stDivider {
        border-color: var(--border) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ===== RADIO BUTTONS ===== */
    .stRadio > div {
        gap: 0.5rem !important;
    }
    
    .stRadio > label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    /* ===== TEXT INPUT ===== */
    .stTextInput > div > div > input {
        border-radius: var(--radius-sm) !important;
        border: 2px solid var(--border) !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        background: white !important;
        color: var(--text-primary) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ff4b4b !important;
        box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.2) !important;
    }
    
    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-color: #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== TYPING INDICATOR FUNCTION =====
def show_typing():
    """Display typing indicator"""
    return st.markdown("""
        <div class="typing-indicator">
            <span style="font-size:0.8rem; color:#94a3b8;">Thinking</span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    """, unsafe_allow_html=True)

# ===== CHAT EXPORT FUNCTION =====
def export_chat_history(messages):
    """Generate markdown export of chat history"""
    if not messages:
        return None
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    export_text = f"""# 📚 RAG Document Q&A - Chat History

**Exported on:** {timestamp}
**Total Messages:** {len(messages)}

---
"""
    
    for i, msg in enumerate(messages, 1):
        if msg['role'] == 'user':
            export_text += f"\n## 🧑 User Question {i}\n\n{msg['content']}\n"
        else:
            export_text += f"\n## 🤖 Assistant Response {i}\n\n{msg['content']}\n"
            
            # Add citations if available
            if 'citations' in msg and msg['citations']:
                export_text += "\n### 📚 Sources:\n\n"
                for j, citation in enumerate(msg['citations'][:5], 1):
                    export_text += f"{j}. **{citation['source']}** (Page {citation['page']}) - Score: {citation['similarity_score']:.4f}\n"
                    export_text += f"   *{citation['text_preview']}*\n"
        
        export_text += "\n---\n"
    
    export_text += f"""
---
**Powered by:** LangChain, FAISS, Sentence-Transformers, and Ollama
**Document Q&A System**
"""
    
    return export_text

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'responder' not in st.session_state:
    st.session_state.responder = None
if 'index_loaded' not in st.session_state:
    st.session_state.index_loaded = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_documents' not in st.session_state:
    st.session_state.current_documents = []
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []

# Title - BLACK HEADING
st.markdown('<div class="main-header">📚 RAG Document Q&A System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your documents with grounded responses</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ollama status
    st.subheader("🤖 LLM Status")
    try:
        responder = LLMResponder()
        status = responder.check_ollama_status()
        
        if status['status'] == 'running':
            st.success(f"✅ Ollama is running")
            st.info(f"Model: {OLLAMA_MODEL}")
        else:
            st.error("❌ Ollama is not running")
            st.warning("Start Ollama with: `ollama serve`")
    except Exception as e:
        st.error(f"❌ Error checking Ollama: {str(e)}")
    
    st.divider()
    
    # File upload
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=['pdf', 'txt', 'md'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        index_action = st.radio(
            "Index action:",
            ["Replace existing index (recommended)", "Add to existing index"],
            help="Replace: Removes old documents and uses only new ones. Add: Keeps old documents and adds new ones."
        )
    
    if uploaded_files:
        if st.button("📥 Process Documents", type="primary", use_container_width=True):
            st.session_state.processing = True
            with st.spinner("Processing documents... This may take a moment..."):
                try:
                    # Save uploaded files to data directory
                    saved_paths = []
                    for file in uploaded_files:
                        file_path = DATA_DIR / file.name
                        with open(file_path, 'wb') as f:
                            f.write(file.getbuffer())
                        saved_paths.append(str(file_path))
                        st.info(f"✅ Saved: {file.name}")
                    
                    # Load documents
                    loader = DocumentLoader()
                    documents = loader.load_multiple_documents(saved_paths)
                    
                    # Chunk documents with smaller size for better retrieval
                    processor = TextProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                    chunks = processor.chunk_documents(documents)
                    
                    # Check if we should replace or add
                    index_dir = VECTORSTORE_DIR / "main_index"
                    
                    if index_action == "Replace existing index (recommended)":
                        # Delete old index
                        if index_dir.exists():
                            shutil.rmtree(index_dir)
                            st.info("🗑️ Old index removed")
                    
                    # Build or update index
                    indexer = EmbeddingIndexer()
                    
                    if index_dir.exists() and index_action == "Add to existing index":
                        # Load existing and add
                        indexer.load_index(index_name="main_index")
                        indexer.add_documents(chunks)
                    else:
                        # Build new index
                        indexer.build_index(chunks)
                    
                    # Save index
                    indexer.save_index(index_name="main_index")
                    
                    # Initialize retriever and responder
                    st.session_state.retriever = Retriever(indexer)
                    st.session_state.responder = LLMResponder(temperature=0.1)
                    st.session_state.index_loaded = True
                    st.session_state.current_documents = [f.name for f in uploaded_files]
                    
                    st.success(f"✅ Processed {len(documents)} documents with {len(chunks)} chunks")
                    st.info(f"📄 Documents: {', '.join([f.name for f in uploaded_files])}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            st.session_state.processing = False
    
    st.divider()
    
    # Manage Documents
    st.subheader("🗑️ Manage Documents")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear All Documents", use_container_width=True):
            try:
                for file in DATA_DIR.glob("*"):
                    if file.is_file():
                        file.unlink()
                st.success("✅ All documents cleared")
                # Also clear index
                index_dir = VECTORSTORE_DIR / "main_index"
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                    st.session_state.index_loaded = False
                    st.session_state.retriever = None
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Load existing index
    st.subheader("📂 Load Index")
    index_dir = VECTORSTORE_DIR / "main_index"
    if index_dir.exists():
        if st.button("🔄 Load Existing Index", use_container_width=True):
            with st.spinner("Loading index..."):
                try:
                    indexer = EmbeddingIndexer()
                    indexer.load_index(index_name="main_index")
                    st.session_state.retriever = Retriever(indexer)
                    st.session_state.responder = LLMResponder(temperature=0.1)
                    st.session_state.index_loaded = True
                    st.success(f"✅ Index loaded with {len(indexer.chunks)} chunks")
                except Exception as e:
                    st.error(f"❌ Error loading index: {str(e)}")
    else:
        st.info("No existing index found. Upload documents first.")
    
    st.divider()
    
    # Stats
    st.subheader("📊 System Stats")
    if st.session_state.retriever and st.session_state.retriever.indexer:
        info = st.session_state.retriever.indexer.get_index_info()
        st.metric("Total Chunks", info['total_chunks'])
        st.metric("Embedding Dimension", info['dimension'])
        st.metric("Model", info['model_name'].split('/')[-1])
        
        # Show current documents
        if st.session_state.current_documents:
            st.write("📄 Current Documents:")
            for doc in st.session_state.current_documents[:5]:
                st.write(f"   - {doc}")
    else:
        st.info("No index loaded")
    
    st.divider()
    
    # ===== CHAT ACTIONS SECTION =====
    st.subheader("💬 Chat Actions")
    
    # Export Chat Button
    if st.session_state.messages:
        export_text = export_chat_history(st.session_state.messages)
        if export_text:
            st.download_button(
                label="📤 Export Chat",
                data=export_text,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
                help="Download the entire chat history as a Markdown file"
            )
    else:
        st.info("No messages to export")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Debug info
    with st.expander("🔧 Debug Info"):
        st.write(f"TOP_K_RESULTS: {TOP_K_RESULTS}")
        st.write(f"CHUNK_SIZE: {CHUNK_SIZE}")
        st.write(f"CHUNK_OVERLAP: {CHUNK_OVERLAP}")
        st.write(f"MAX_RESPONSE_TOKENS: 3000")
        if st.session_state.messages:
            st.write(f"Messages: {len(st.session_state.messages)}")

# Main chat area
chat_container = st.container()

with chat_container:
    # Display chat history
    if not st.session_state.messages:
        st.markdown("""
            <div class="info-box">
                💡 <strong>Welcome to RAG Document Q&A!</strong><br>
                Upload your documents in the sidebar and start asking questions.
                The system will search through your documents and provide grounded answers with citations.
            </div>
        """, unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="user-message">
                    <strong>🧑 You</strong>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant</strong>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)
            
            # ===== ENHANCED CITATIONS WITH CONFIDENCE BARS =====
            if 'citations' in message and message['citations']:
                with st.expander("📚 View Sources"):
                    for i, citation in enumerate(message['citations'][:5], 1):
                        score = citation['similarity_score']
                        
                        # Determine score class and color
                        if score > 0.9:
                            score_class = "high"
                            score_label = "High Confidence"
                        elif score > 0.7:
                            score_class = "medium"
                            score_label = "Medium Confidence"
                        else:
                            score_class = "low"
                            score_label = "Low Confidence"
                        
                        # Determine bar color
                        bar_color = '#10b981' if score > 0.9 else '#f59e0b' if score > 0.7 else '#ef4444'
                        
                        st.markdown(f"""
                            <div class="source-item">
                                <div class="source-header">
                                    <span class="source-name">{i}. {citation['source']}</span>
                                    <span class="source-score {score_class}">{score_label} ({score:.3f})</span>
                                </div>
                                <div class="source-page">📄 Page {citation['page']}</div>
                                <div class="source-preview">"{citation['text_preview']}"</div>
                                <div class="confidence-container">
                                    <div class="confidence-bar-track">
                                        <div class="confidence-bar-fill {score_class}" style="width: {min(score * 100, 100)}%;"></div>
                                    </div>
                                    <div class="confidence-labels">
                                        <span>Relevance</span>
                                        <span>{score:.1%}</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

# Chat input at bottom
if st.session_state.index_loaded:
    with st.container():
        st.divider()
        col1, col2 = st.columns([6, 1])
        with col1:
            query = st.text_input(
                "Ask a question about your documents:",
                placeholder="Type your question here...",
                key="query_input",
                disabled=st.session_state.processing
            )
        with col2:
            submit = st.button(
                "Send 📤",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing or not query
            )
        
        if submit and query:
            # Add user message
            st.session_state.messages.append({
                'role': 'user',
                'content': query
            })
            
            # Process query
            with st.spinner("🧠 Searching and generating detailed response..."):
                try:
                    # Show typing indicator
                    typing_placeholder = st.empty()
                    with typing_placeholder:
                        show_typing()
                    
                    # Get chunks for coverage (using TOP_K_RESULTS from config)
                    context = st.session_state.retriever.get_context(query, k=TOP_K_RESULTS)
                    results = st.session_state.retriever.last_results
                    citations = st.session_state.retriever.get_citations()
                    
                    # Clear typing indicator
                    typing_placeholder.empty()
                    
                    # Debug info
                    st.session_state.debug_info.append({
                        'query': query,
                        'chunks_found': len(results),
                        'context_length': len(context)
                    })
                    
                    if not context or len(context.strip()) < 10:
                        st.session_state.messages.append({
                            'role': 'assistant',
                            'content': "I don't know - no relevant information found in the documents."
                        })
                    else:
                        # First try with strict response
                        response_data = st.session_state.responder.generate_strict_response(query, context)
                        
                        if response_data['success']:
                            response_text = response_data['response']
                            
                            # If response says "don't know" or is too short, try with regular prompt
                            if ("don't know" in response_text.lower() or len(response_text) < 100) and len(context) > 300:
                                response_data2 = st.session_state.responder.generate_response(query, context)
                                if response_data2['success'] and "don't know" not in response_data2['response'].lower():
                                    response_text = response_data2['response']
                                elif response_data2['success'] and len(response_data2['response']) > len(response_text):
                                    response_text = response_data2['response']
                            
                            # If still "don't know" but we have context, show the context
                            if "don't know" in response_text.lower() and len(context) > 300:
                                response_text = f"""I found information in the documents but the system had trouble generating a response. Here's what was found:

{context[:1500]}

Please try rephrasing your question or be more specific."""
                            
                            st.session_state.messages.append({
                                'role': 'assistant',
                                'content': response_text,
                                'citations': citations[:5] if citations else []
                            })
                        else:
                            st.session_state.messages.append({
                                'role': 'assistant',
                                'content': f"❌ Error: {response_data.get('error', 'Unknown error')}"
                            })
                        
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': f"❌ Error: {str(e)}\n\nPlease check if Ollama is running."
                    })
                    st.error(f"Error details: {error_details}")
            
            # Rerun to update chat
            st.rerun()
else:
    st.info("ℹ️ Please upload documents or load an existing index to start asking questions.")
    st.markdown("""
        **Quick Start:**
        1. Upload PDF, TXT, or MD files in the sidebar
        2. Choose "Replace existing index" for fresh start
        3. Click "Process Documents"
        4. Start asking questions!
    """)

# Footer
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("🔒 Powered by LangChain, FAISS, Sentence-Transformers, and Ollama")