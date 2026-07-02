"""
Embedding and Indexing Module
Creates vector embeddings and builds FAISS index for document chunks
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss

from .config import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIMENSION,
    TOP_K_RESULTS
)

class EmbeddingIndexer:
    """Handle embedding creation and FAISS indexing"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_ids = []
        self.dimension = FAISS_INDEX_DIMENSION
        
        # Load the model
        self._load_model()
        
    def _load_model(self):
        """Load the sentence-transformers model"""
        try:
            print(f"🔄 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded successfully")
            
            # Verify dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            self.dimension = len(test_embedding[0])
            print(f"📊 Embedding dimension: {self.dimension}")
            
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def create_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Create embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        print(f"🔄 Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False  # Keep raw embeddings for FAISS
        )
        
        print(f"✅ Embeddings created: shape {embeddings.shape}")
        return embeddings
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> faiss.Index:
        """
        Build FAISS index from chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            FAISS index
        """
        if not chunks:
            raise ValueError("No chunks provided to build index")
        
        print(f"\n📊 Building FAISS index from {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Create FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and IDs
        self.chunks = chunks
        self.chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
        print(f"   Index dimension: {self.dimension}")
        
        return self.index
    
    def search(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using a query
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of chunks with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if not query:
            return []
        
        # Create query embedding
        query_embedding = self.create_embeddings([query], show_progress=False)
        
        # Search in FAISS
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(k, self.index.ntotal)
        )
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(distances[0][i])
                chunk['chunk_id'] = self.chunk_ids[idx]
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def save_index(self, index_name: str = "faiss_index") -> None:
        """
        Save FAISS index and chunk data to disk
        
        Args:
            index_name: Name for the index files
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        # Create directory
        index_dir = VECTORSTORE_DIR / index_name
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = index_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        print(f"✅ FAISS index saved: {index_path}")
        
        # Save chunks
        chunks_path = index_dir / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_ids': self.chunk_ids,
                'dimension': self.dimension,
                'model_name': self.model_name
            }, f)
        print(f"✅ Chunks saved: {chunks_path}")
        
        # Save metadata
        metadata = {
            'index_name': index_name,
            'num_chunks': len(self.chunks),
            'dimension': self.dimension,
            'model_name': self.model_name,
            'chunk_size': 500,  # From config
            'chunk_overlap': 50  # From config
        }
        
        metadata_path = index_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved: {metadata_path}")
    
    def load_index(self, index_name: str = "faiss_index") -> None:
        """
        Load FAISS index and chunk data from disk
        
        Args:
            index_name: Name of the index to load
        """
        index_dir = VECTORSTORE_DIR / index_name
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")
        
        # Load FAISS index
        index_path = index_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        print(f"✅ FAISS index loaded: {index_path}")
        print(f"   Total vectors: {self.index.ntotal}")
        
        # Load chunks
        chunks_path = index_dir / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.chunk_ids = data.get('chunk_ids', [f"chunk_{i}" for i in range(len(self.chunks))])
                self.dimension = data.get('dimension', self.dimension)
                self.model_name = data.get('model_name', self.model_name)
            print(f"✅ Chunks loaded: {len(self.chunks)}")
        
        # Load metadata
        metadata_path = index_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded")
            print(f"   Index name: {metadata.get('index_name', 'unknown')}")
            print(f"   Model: {metadata.get('model_name', 'unknown')}")
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index
        
        Returns:
            Dictionary with index information
        """
        return {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'model_name': self.model_name,
            'has_index': self.index is not None,
            'index_size': self.index.ntotal if self.index else 0
        }
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add new documents to existing index
        
        Args:
            chunks: List of chunk dictionaries to add
        """
        if self.index is None:
            self.build_index(chunks)
            return
        
        # Extract texts from new chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Add to FAISS
        self.index.add(embeddings.astype('float32'))
        
        # Update chunks
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        self.chunk_ids.extend([f"chunk_{i}" for i in range(start_idx, start_idx + len(chunks))])
        
        print(f"✅ Added {len(chunks)} chunks to index")
        print(f"   Total chunks now: {self.index.ntotal}")