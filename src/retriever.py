"""
Retrieval Module - FINAL WORKING VERSION
Handles query processing and retrieval of relevant chunks from FAISS index
"""

from typing import List, Dict, Any, Optional
from src.embedding_indexer import EmbeddingIndexer
from src.config import TOP_K_RESULTS

class Retriever:
    """Retrieve relevant chunks from FAISS index"""
    
    def __init__(self, indexer: Optional[EmbeddingIndexer] = None):
        self.indexer = indexer
        self.last_query = None
        self.last_results = []
    
    def set_indexer(self, indexer: EmbeddingIndexer) -> None:
        self.indexer = indexer
    
    def retrieve(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        if self.indexer is None:
            raise ValueError("No indexer set.")
        
        if not query:
            return []
        
        results = self.indexer.search(query, k)
        self.last_query = query
        self.last_results = results
        
        return results
    
    def get_context(self, query: str, k: int = TOP_K_RESULTS) -> str:
        """
        Get formatted context string from retrieved chunks - includes FULL text
        """
        results = self.retrieve(query, k)
        
        if not results:
            return ""
        
        # Build context with ALL chunks - include FULL text
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('file_name', 'unknown')
            page = result['metadata'].get('page', 'N/A')
            chunk_text = result['text'].strip()
            score = result.get('similarity_score', 0)
            
            # Keep the full text - don't truncate
            context_part = f"""
=== SOURCE {i} ===
File: {source}
Page: {page}
Score: {score:.4f}

{chunk_text}

=== END SOURCE {i} ===
"""
            context_parts.append(context_part)
        
        full_context = "\n".join(context_parts)
        
        return full_context
    
    def get_citations(self, results: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Extract citations from retrieval results"""
        if results is None:
            results = self.last_results
        
        citations = []
        for result in results:
            metadata = result.get('metadata', {})
            citation = {
                'source': metadata.get('file_name', 'unknown'),
                'page': metadata.get('page', 'N/A'),
                'similarity_score': result.get('similarity_score', 0),
                'rank': result.get('rank', 0),
                'text_preview': result.get('text', '')[:400] + '...' if result.get('text') else ''
            }
            citations.append(citation)
        
        return citations
    
    def format_results(self, results: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format retrieval results for display"""
        if results is None:
            results = self.last_results
        
        if not results:
            return "No results found."
        
        output = f"\n📊 Found {len(results)} relevant chunks:\n"
        output += "=" * 60 + "\n"
        
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('file_name', 'unknown')
            page = result['metadata'].get('page', 'N/A')
            score = result.get('similarity_score', 0)
            text = result.get('text', '')[:150] + '...' if len(result.get('text', '')) > 150 else result.get('text', '')
            
            output += f"\n{i}. 📄 {source} (Page {page}) - Score: {score:.4f}\n"
            output += f"   {text}\n"
            output += "-" * 40 + "\n"
        
        return output