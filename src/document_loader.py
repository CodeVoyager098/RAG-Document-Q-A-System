"""
Document Loader Module - Updated with better TXT/MD support
Handles loading and extracting text from PDF, TXT, and MD files
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2

class DocumentLoader:
    """Load and extract text from various document formats"""
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.txt', '.md', '.text'}
        
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load a single document and extract text with metadata
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with 'text', 'metadata', and 'pages' information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}. Supported: {self.supported_extensions}")
        
        # Route to appropriate loader
        if extension == '.pdf':
            return self._load_pdf(file_path)
        elif extension in ['.txt', '.text']:
            return self._load_txt(file_path)
        elif extension == '.md':
            return self._load_md(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def _load_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Load text from PDF file with page-by-page extraction"""
        text_by_page = []
        full_text = ""
        metadata = {
            'source': str(file_path),
            'file_name': file_path.name,
            'file_type': '.pdf',
            'total_pages': 0
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata['total_pages'] = len(pdf_reader.pages)
                
                # Extract metadata from PDF if available
                if pdf_reader.metadata:
                    metadata['pdf_metadata'] = {
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'producer': pdf_reader.metadata.get('/Producer', '')
                    }
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    
                    # Clean the page text
                    page_text = self._clean_text(page_text)
                    
                    text_by_page.append({
                        'page': page_num,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
                    
                    full_text += page_text + "\n"
        
        except Exception as e:
            raise Exception(f"Error reading PDF {file_path}: {str(e)}")
        
        # Clean the full text
        full_text = self._clean_text(full_text)
        
        return {
            'full_text': full_text,
            'text_by_page': text_by_page,
            'metadata': metadata,
            'char_count': len(full_text),
            'word_count': len(full_text.split())
        }
    
    def _load_txt(self, file_path: Path) -> Dict[str, Any]:
        """
        Load text from TXT file - ENHANCED VERSION
        """
        metadata = {
            'source': str(file_path),
            'file_name': file_path.name,
            'file_type': '.txt'
        }
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']
            full_text = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        full_text = file.read()
                        used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if full_text is None:
                raise Exception(f"Could not decode file with any encoding: {file_path}")
            
            # Clean the text
            full_text = self._clean_text(full_text)
            
            # Split into pages (treat as single page)
            text_by_page = [{
                'page': 1,
                'text': full_text,
                'char_count': len(full_text)
            }]
            
            metadata['total_pages'] = 1
            metadata['encoding'] = used_encoding
            
            return {
                'full_text': full_text,
                'text_by_page': text_by_page,
                'metadata': metadata,
                'char_count': len(full_text),
                'word_count': len(full_text.split())
            }
            
        except Exception as e:
            raise Exception(f"Error reading TXT file {file_path}: {str(e)}")
    
    def _load_md(self, file_path: Path) -> Dict[str, Any]:
        """
        Load text from Markdown file - ENHANCED VERSION
        """
        metadata = {
            'source': str(file_path),
            'file_name': file_path.name,
            'file_type': '.md'
        }
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            full_text = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        full_text = file.read()
                        used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if full_text is None:
                raise Exception(f"Could not decode file with any encoding: {file_path}")
            
            # Clean the text (markdown-specific cleaning)
            full_text = self._clean_markdown(full_text)
            full_text = self._clean_text(full_text)
            
            # Split into pages (treat as single page)
            text_by_page = [{
                'page': 1,
                'text': full_text,
                'char_count': len(full_text)
            }]
            
            metadata['total_pages'] = 1
            metadata['encoding'] = used_encoding
            
            return {
                'full_text': full_text,
                'text_by_page': text_by_page,
                'metadata': metadata,
                'char_count': len(full_text),
                'word_count': len(full_text.split())
            }
            
        except Exception as e:
            raise Exception(f"Error reading MD file {file_path}: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text - ENHANCED VERSION
        """
        if not text:
            return ""
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove extra spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Basic header/footer removal
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'===== Page \d+ =====', '', text, flags=re.IGNORECASE)
        text = re.sub(r'=\s*Page\s*\d+\s*=', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common PDF artifacts
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        
        return text.strip()
    
    def _clean_markdown(self, text: str) -> str:
        """
        Clean markdown-specific elements - ENHANCED VERSION
        """
        if not text:
            return ""
        
        # Remove markdown headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'___([^_]+)___', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove images
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
        
        # Remove horizontal rules
        text = re.sub(r'^---\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^___\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\*\*\*\s*$', '', text, flags=re.MULTILINE)
        
        # Remove list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        return text
    
    def load_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load multiple documents"""
        documents = []
        errors = []
        
        for file_path in file_paths:
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
                print(f"✅ Loaded: {Path(file_path).name}")
            except Exception as e:
                error_msg = f"❌ Error loading {Path(file_path).name}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
        
        print(f"\n📊 Summary: {len(documents)} loaded, {len(errors)} failed")
        
        return documents
    
    def load_directory(self, directory_path: str, extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load all supported documents from a directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        if extensions is None:
            extensions = self.supported_extensions
        
        file_paths = []
        for ext in extensions:
            file_paths.extend(directory.glob(f"*{ext}"))
        
        file_paths = [str(f) for f in file_paths]
        
        if not file_paths:
            print(f"⚠️  No supported files found in {directory_path}")
            print(f"   Supported extensions: {extensions}")
            return []
        
        print(f"📁 Found {len(file_paths)} files in {directory_path}")
        return self.load_multiple_documents(file_paths)