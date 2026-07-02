"""
Rebuild index with FINAL settings - larger chunks, more context
"""

import sys
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.document_loader import DocumentLoader
from src.text_processor import TextProcessor
from src.embedding_indexer import EmbeddingIndexer
from src.config import DATA_DIR, VECTORSTORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def rebuild():
    print("=" * 60)
    print("REBUILDING WITH FINAL SETTINGS")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"Chunk Overlap: {CHUNK_OVERLAP}")
    print("=" * 60)
    
    # Delete old index
    index_dir = VECTORSTORE_DIR / "main_index"
    if index_dir.exists():
        shutil.rmtree(index_dir)
        print("🗑️ Old index removed")
    
    # Load documents
    loader = DocumentLoader()
    all_files = list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.md"))
    
    if not all_files:
        print("⚠️  No files found in data directory")
        print(f"   Please add PDF, TXT, or MD files to: {DATA_DIR}")
        return
    
    print(f"\n📁 Found {len(all_files)} files:")
    for f in all_files:
        print(f"   - {f.name}")
    
    # Load documents
    print("\n📂 Loading documents...")
    documents = loader.load_multiple_documents([str(f) for f in all_files])
    
    if not documents:
        print("❌ No documents could be loaded")
        return
    
    # Chunk with final settings
    print(f"\n✂️ Chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
    processor = TextProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = processor.chunk_documents(documents)
    
    if not chunks:
        print("❌ No chunks created")
        return
    
    print(f"\n📊 Created {len(chunks)} chunks")
    
    # Show chunk statistics
    stats = processor.get_chunk_stats(chunks)
    print(f"\n📊 Chunk Statistics:")
    print(f"   Average chunk size: {stats['avg_chunk_size']:.0f} characters")
    print(f"   Min chunk size: {stats['min_chunk_size']} characters")
    print(f"   Max chunk size: {stats['max_chunk_size']} characters")
    
    # Show sample chunks
    print("\n📄 Sample Chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nChunk {i}:")
        print(f"   Source: {chunk['metadata'].get('file_name', 'unknown')}")
        print(f"   Length: {len(chunk['text'])} characters")
        print(f"   Page: {chunk['metadata'].get('page', 'N/A')}")
        print(f"   Preview: {chunk['text'][:200]}...")
    
    # Build index
    print("\n🔨 Building FAISS index...")
    indexer = EmbeddingIndexer()
    indexer.build_index(chunks)
    indexer.save_index(index_name="main_index")
    
    print("\n" + "=" * 60)
    print("✅ INDEX REBUILT SUCCESSFULLY!")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Index saved at: {VECTORSTORE_DIR / 'main_index'}")
    print("=" * 60)

if __name__ == "__main__":
    rebuild()