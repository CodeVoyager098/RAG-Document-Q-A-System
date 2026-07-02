"""
Evaluation Module
Evaluates RAG system performance using RAGAS metrics
Measures answer faithfulness and relevance
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.retriever import Retriever
from src.llm_responder import LLMResponder
from src.config import DATA_DIR, VECTORSTORE_DIR

class RAGEvaluator:
    """
    Evaluate RAG system performance
    Measures faithfulness and relevance of answers
    """
    
    def __init__(self, retriever: Retriever, responder: LLMResponder):
        """
        Initialize evaluator
        
        Args:
            retriever: Retriever instance
            responder: LLMResponder instance
        """
        self.retriever = retriever
        self.responder = responder
        self.results = []
    
    def create_golden_qa_set(self, documents: List[Dict]) -> List[Dict]:
        """
        Create a golden QA set from documents
        This generates test questions and expected answers
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of QA pairs with expected answers
        """
        qa_pairs = []
        
        # For each document, extract key information
        for doc in documents:
            metadata = doc.get('metadata', {})
            text = doc.get('full_text', '')
            
            if not text:
                continue
            
            # Generate questions based on document content
            # This is a simple approach - you can manually create these
            lines = text.split('\n')
            questions = []
            answers = []
            
            # Extract first few sentences as potential questions
            for i, line in enumerate(lines[:20]):
                if len(line.strip()) > 20:
                    # Look for potential question triggers
                    if any(keyword in line.lower() for keyword in 
                           ['what', 'how', 'why', 'when', 'who', 'which', 'define']):
                        questions.append(line.strip())
            
            # For demo purposes, let's create some sample QA pairs
            # In production, you should manually create these
            sample_qa = [
                {
                    "question": f"What is the main topic of {metadata.get('file_name', 'this document')}?",
                    "answer": text[:200] if len(text) > 200 else text,
                    "source": metadata.get('file_name', 'unknown'),
                    "page": 1
                },
                {
                    "question": "What are the key points discussed?",
                    "answer": text[:300] if len(text) > 300 else text,
                    "source": metadata.get('file_name', 'unknown'),
                    "page": 1
                }
            ]
            
            qa_pairs.extend(sample_qa)
        
        return qa_pairs
    
    def create_manual_qa_set(self) -> List[Dict]:
        """
        Create a manual QA set based on your documents
        You should customize this with your own questions
        """
        # TODO: Replace with your own questions based on your documents
        qa_set = [
            {
                "question": "What is the Constitution of 1956?",
                "answer": "The Constitution of 1956 was Pakistan's first constitution...",
                "source": "Constitutional-Development-and-Political-Struggle-in-Pakistan.pdf",
                "page": 1
            },
            {
                "question": "What were the key features of the 1956 Constitution?",
                "answer": "The key features included Islamic character, parliamentary system, federal structure...",
                "source": "Constitutional-Development-and-Political-Struggle-in-Pakistan.pdf",
                "page": 1
            },
            {
                "question": "What was the War of Independence 1857?",
                "answer": "The War of Independence 1857 was a major uprising against British rule...",
                "source": "Historic Struggle for Pakistan 1857-1947.pdf",
                "page": 4
            },
            {
                "question": "What is an economy?",
                "answer": "An economy is a complex structure of interdependent producers and consumers...",
                "source": "economy.pdf",
                "page": 1
            },
            {
                "question": "What are the types of economies?",
                "answer": "The types of economies include Free Market Economy, Command Economy, and Mixed Economy...",
                "source": "economy.pdf",
                "page": 2
            }
        ]
        return qa_set
    
    def evaluate_qa_pair(self, qa_pair: Dict) -> Dict:
        """
        Evaluate a single QA pair
        
        Args:
            qa_pair: Dictionary with question and expected answer
            
        Returns:
            Evaluation results dictionary
        """
        question = qa_pair.get('question', '')
        expected_answer = qa_pair.get('answer', '')
        expected_source = qa_pair.get('source', 'unknown')
        expected_page = qa_pair.get('page', 1)
        
        # Retrieve context
        context = self.retriever.get_context(question, k=5)
        results = self.retriever.last_results
        citations = self.retriever.get_citations()
        
        # Generate response
        response_data = self.responder.generate_strict_response(question, context)
        generated_answer = response_data.get('response', '')
        
        # Calculate faithfulness score
        faithfulness = self._calculate_faithfulness(generated_answer, context)
        
        # Calculate relevance score
        relevance = self._calculate_relevance(generated_answer, question)
        
        # Check if source is correct
        source_correct = False
        for citation in citations:
            if expected_source in citation.get('source', ''):
                source_correct = True
                break
        
        return {
            'question': question,
            'expected_answer': expected_answer,
            'generated_answer': generated_answer,
            'context': context,
            'citations': citations,
            'faithfulness_score': faithfulness,
            'relevance_score': relevance,
            'source_correct': source_correct,
            'expected_source': expected_source,
            'found_source': [c.get('source', '') for c in citations[:3]],
            'page_correct': expected_page in [c.get('page', '') for c in citations] if citations else False
        }
    
    def _calculate_faithfulness(self, answer: str, context: str) -> float:
        """
        Calculate faithfulness score (does answer appear in context?)
        
        Returns:
            Score between 0 and 1
        """
        if not answer or not context:
            return 0.0
        
        # Check if answer content appears in context
        answer_sentences = answer.split('.')
        context_lower = context.lower()
        
        matched_sentences = 0
        for sentence in answer_sentences:
            sentence = sentence.strip().lower()
            if len(sentence) > 10 and sentence in context_lower:
                matched_sentences += 1
        
        if len(answer_sentences) > 0:
            score = matched_sentences / len(answer_sentences)
        else:
            score = 0.0
        
        # Check if answer says "I don't know" when it should
        if "don't know" in answer.lower() and context and len(context) > 100:
            score = min(score, 0.3)  # Penalize "don't know" when context exists
        
        return min(score, 1.0)
    
    def _calculate_relevance(self, answer: str, question: str) -> float:
        """
        Calculate relevance score (is answer relevant to question?)
        
        Returns:
            Score between 0 and 1
        """
        if not answer or not question:
            return 0.0
        
        # Check if answer contains key terms from question
        question_words = set(question.lower().split())
        answer_lower = answer.lower()
        
        # Remove common stopwords
        stopwords = {'what', 'how', 'why', 'when', 'who', 'which', 'is', 'are', 'was', 'were', 
                     'the', 'a', 'an', 'of', 'to', 'for', 'with', 'on', 'at', 'from', 'by'}
        question_words = question_words - stopwords
        
        if not question_words:
            return 1.0
        
        matched_words = sum(1 for word in question_words if word in answer_lower)
        score = matched_words / len(question_words) if question_words else 0.5
        
        return min(score, 1.0)
    
    def evaluate_all(self, qa_set: List[Dict]) -> pd.DataFrame:
        """
        Evaluate all QA pairs
        
        Args:
            qa_set: List of QA pairs
            
        Returns:
            DataFrame with evaluation results
        """
        print("\n" + "=" * 60)
        print("📊 EVALUATING QA PAIRS")
        print("=" * 60)
        
        results = []
        for i, qa_pair in enumerate(qa_set, 1):
            print(f"\n📝 Evaluating {i}/{len(qa_set)}: {qa_pair.get('question', '')[:50]}...")
            result = self.evaluate_qa_pair(qa_pair)
            results.append(result)
            
            print(f"   Faithfulness: {result['faithfulness_score']:.2f}")
            print(f"   Relevance: {result['relevance_score']:.2f}")
            print(f"   Source Correct: {result['source_correct']}")
        
        self.results = results
        df = pd.DataFrame(results)
        return df
    
    def get_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics from evaluation results
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            Summary statistics dictionary
        """
        if df.empty:
            return {
                'total_qa_pairs': 0,
                'avg_faithfulness': 0,
                'avg_relevance': 0,
                'source_accuracy': 0,
                'success_rate': 0
            }
        
        total = len(df)
        avg_faithfulness = df['faithfulness_score'].mean()
        avg_relevance = df['relevance_score'].mean()
        source_accuracy = df['source_correct'].mean() * 100
        
        # Calculate success rate (faithfulness > 0.5 and relevance > 0.5)
        success = ((df['faithfulness_score'] > 0.5) & (df['relevance_score'] > 0.5)).sum()
        success_rate = (success / total) * 100 if total > 0 else 0
        
        return {
            'total_qa_pairs': total,
            'avg_faithfulness': avg_faithfulness,
            'avg_relevance': avg_relevance,
            'source_accuracy': source_accuracy,
            'success_rate': success_rate,
            'min_faithfulness': df['faithfulness_score'].min(),
            'max_faithfulness': df['faithfulness_score'].max(),
            'min_relevance': df['relevance_score'].min(),
            'max_relevance': df['relevance_score'].max()
        }
    
    def print_report(self, df: pd.DataFrame) -> None:
        """
        Print detailed evaluation report
        
        Args:
            df: DataFrame with evaluation results
        """
        summary = self.get_summary(df)
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\n📈 Overall Metrics:")
        print(f"   Total QA Pairs: {summary['total_qa_pairs']}")
        print(f"   Average Faithfulness: {summary['avg_faithfulness']:.3f}")
        print(f"   Average Relevance: {summary['avg_relevance']:.3f}")
        print(f"   Source Accuracy: {summary['source_accuracy']:.1f}%")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        
        print(f"\n📊 Score Ranges:")
        print(f"   Faithfulness: {summary['min_faithfulness']:.2f} - {summary['max_faithfulness']:.2f}")
        print(f"   Relevance: {summary['min_relevance']:.2f} - {summary['max_relevance']:.2f}")
        
        print("\n📝 Detailed Results:")
        print("-" * 60)
        for i, row in df.iterrows():
            print(f"\n{i+1}. Question: {row['question']}")
            print(f"   Generated: {row['generated_answer'][:100]}...")
            print(f"   Faithfulness: {row['faithfulness_score']:.3f}")
            print(f"   Relevance: {row['relevance_score']:.3f}")
            print(f"   Source Correct: {row['source_correct']}")
            print(f"   Sources Found: {', '.join(row['found_source'])}")
    
    def save_results(self, df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Save evaluation results to file
        
        Args:
            df: DataFrame with evaluation results
            output_path: Path to save results
        """
        if output_path is None:
            output_path = Path(VECTORSTORE_DIR) / "evaluation_results.csv"
        else:
            output_path = Path(output_path)
        
        df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Also save summary
        summary = self.get_summary(df)
        summary_path = output_path.parent / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Summary saved to: {summary_path}")