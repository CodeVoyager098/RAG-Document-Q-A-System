"""
LLM Responder Module - FINAL OPTIMIZED VERSION
Generates grounded responses using Ollama
"""

import json
import requests
from typing import List, Dict, Any, Optional

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    MAX_RESPONSE_TOKENS,
    OLLAMA_TIMEOUT
)

class LLMResponder:
    """Generate grounded responses using Ollama"""
    
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.1,
        max_tokens: int = 3000,
        timeout: int = 300
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.last_response = None
        self.last_prompt = None
    
    def generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """Generate grounded response using Ollama"""
        
        if not context or len(context.strip()) < 10:
            return {
                'success': True,
                'response': "I don't know - no context provided.",
                'query': query,
                'context_used': context,
                'model': self.model,
                'temperature': self.temperature
            }
        
        # Truncate context to prevent timeout
        max_context = 4000
        if len(context) > max_context:
            context = context[:max_context] + "\n...[context truncated for performance]"
        
        prompt = self._build_prompt(query, context)
        self.last_prompt = prompt
        
        try:
            response = self._call_ollama(prompt)
            self.last_response = response
            
            return {
                'success': True,
                'response': response,
                'query': query,
                'context_used': context,
                'model': self.model,
                'temperature': self.temperature
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': "I'm sorry, I encountered an error.",
                'error': str(e),
                'query': query
            }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for detailed answers - POLISHED VERSION"""
        
        prompt = f"""You are a detailed assistant. Answer the question using ONLY the context below.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Write a COHERENT, WELL-STRUCTURED answer in PARAGRAPH form.
2. Combine ALL information from ALL sources into ONE narrative.
3. DO NOT list sources within the answer.
4. Include ALL relevant facts from the context.
5. At the VERY END, cite all sources as: (Source: filename, Page X)
6. If the context doesn't contain the answer, say "I don't know."

ANSWER (write as a unified paragraph, cite sources at the end):"""
        
        return prompt

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with increased timeout"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status {response.status_code}")
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}")
        except requests.exceptions.Timeout:
            raise Exception(f"Ollama request timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def generate_strict_response(self, query: str, context: str) -> Dict[str, Any]:
        """Generate response with strict grounding - OPTIMIZED"""
        
        if not context or len(context.strip()) < 10:
            return {
                'success': True,
                'response': "I don't know - no context provided.",
                'query': query,
                'context_used': context,
                'model': self.model,
                'temperature': self.temperature,
                'strict_mode': True
            }
        
        # Truncate context for performance
        max_context = 4000
        if len(context) > max_context:
            context = context[:max_context] + "\n...[context truncated for performance]"
        
        prompt = f"""You are a strict assistant. Answer ONLY from the context below.

CONTEXT:
{context}

QUESTION: {query}

RULES:
1. Extract ALL relevant information from the context.
2. If it's a list, include ALL items.
3. If not in context, say "I don't know."
4. Cite source: (Source: filename, Page X)

ANSWER:"""
        
        try:
            response = self._call_ollama(prompt)
            self.last_response = response
            self.last_prompt = prompt
            
            return {
                'success': True,
                'response': response,
                'query': query,
                'context_used': context,
                'model': self.model,
                'temperature': self.temperature,
                'strict_mode': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': "I'm sorry, I encountered an error.",
                'error': str(e),
                'query': query
            }
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                return {
                    'status': 'running',
                    'models': model_names,
                    'model_available': self.model in model_names,
                    'base_url': self.base_url
                }
            else:
                return {
                    'status': 'error',
                    'message': f"Ollama returned status {response.status_code}",
                    'base_url': self.base_url
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'status': 'not_running',
                'message': f"Cannot connect to Ollama at {self.base_url}",
                'base_url': self.base_url
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'base_url': self.base_url
            }