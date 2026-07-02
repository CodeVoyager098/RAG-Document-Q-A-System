"""
Text Processor Module
Handles text cleaning, normalization, and chunking
"""

import re
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    """Process and chunk text for RAG system"""
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        """
        Initialize text processor with chunking parameters
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Updated separators for better PDF and document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",           # Paragraph breaks (highest priority)
                "\n",             # Line breaks
                ". ",             # Sentences
                "? ",             # Questions
                "! ",             # Exclamations
                "; ",             # Semicolons
                ": ",             # Colons
                ", ",             # Commas
                " ",              # Words
                ""                # Characters (lowest priority)
            ],
            add_start_index=True,
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove page numbers and other common artifacts
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'===== Page \d+ =====', '', text, flags=re.IGNORECASE)
        text = re.sub(r'=\s*Page\s*\d+\s*=', '', text, flags=re.IGNORECASE)
        
        # Remove common header/footer patterns
        text = re.sub(r'^[A-Z][A-Z\s]+$', '', text, flags=re.MULTILINE)
        
        # Normalize quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"['']", "'", text)
        
        # Fix common issues
        text = re.sub(r'\.{2,}', '.', text)  # Remove extra dots
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\?\;\:\!\-\'\"\(\)]', ' ', text)
        
        return text.strip()
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document into smaller pieces
        
        Args:
            document: Document dictionary from DocumentLoader
            
        Returns:
            List of chunk dictionaries with metadata
        """
        full_text = document.get('full_text', '')
        metadata = document.get('metadata', {})
        
        if not full_text:
            return []
        
        # Clean the text first
        clean_text = self.clean_text(full_text)
        
        if not clean_text:
            return []
        
        # Split into chunks
        chunks = self.text_splitter.split_text(clean_text)
        
        # Create chunk dictionaries with metadata
        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            # Skip empty or very short chunks
            if len(chunk_text.strip()) < 20:
                continue
                
            chunk_doc = {
                'text': chunk_text,
                'metadata': {
                    'source': metadata.get('source', ''),
                    'file_name': metadata.get('file_name', ''),
                    'file_type': metadata.get('file_type', ''),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'source_type': 'document',
                }
            }
            
            # Add page numbers if available (from PDF)
            if 'text_by_page' in document and document['text_by_page']:
                # Try to determine which page this chunk belongs to
                for page_info in document['text_by_page']:
                    page_text = page_info.get('text', '')
                    # If chunk text appears in this page, assign this page
                    if chunk_text[:50] in page_text or page_text[:50] in chunk_text:
                        chunk_doc['metadata']['page'] = page_info.get('page', 1)
                        break
                else:
                    # Default to first page if not found
                    chunk_doc['metadata']['page'] = document['text_by_page'][0].get('page', 1)
            
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunk dictionaries with metadata
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"📝 Chunked: {doc['metadata'].get('file_name', 'unknown')} -> {len(chunks)} chunks")
        
        print(f"\n📊 Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_chars': 0
            }
        
        sizes = [len(chunk['text']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(sizes) / len(sizes),
            'min_chunk_size': min(sizes),
            'max_chunk_size': max(sizes),
            'total_chars': sum(sizes),
            'total_words': sum([chunk['metadata'].get('word_count', 0) for chunk in chunks])
        }
    
    def preview_chunks(self, chunks: List[Dict[str, Any]], num_chunks: int = 3) -> None:
        """
        Preview the first few chunks
        
        Args:
            chunks: List of chunk dictionaries
            num_chunks: Number of chunks to preview
        """
        print("\n" + "=" * 60)
        print("CHUNK PREVIEW")
        print("=" * 60)
        
        for i, chunk in enumerate(chunks[:num_chunks]):
            print(f"\n📄 Chunk {i + 1}:")
            print(f"   Source: {chunk['metadata'].get('file_name', 'unknown')}")
            print(f"   Characters: {chunk['metadata'].get('char_count', 0)}")
            print(f"   Words: {chunk['metadata'].get('word_count', 0)}")
            print(f"   Page: {chunk['metadata'].get('page', 'N/A')}")
            print(f"   Preview: {chunk['text'][:150]}...")
            print("-" * 40)
        
        if len(chunks) > num_chunks:
            print(f"\n... and {len(chunks) - num_chunks} more chunks")
        print("=" * 60)