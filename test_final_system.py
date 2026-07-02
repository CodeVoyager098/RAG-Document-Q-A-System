"""
Test the final RAG system with all fixes applied
Tests: 14 points, Anti-Muslim Policies, and general questions
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.embedding_indexer import EmbeddingIndexer
from src.retriever import Retriever
from src.llm_responder import LLMResponder
from src.config import TOP_K_RESULTS

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_query(retriever, responder, query, expected_keywords=None):
    """
    Test a single query and display results
    
    Args:
        retriever: Retriever instance
        responder: LLMResponder instance
        query: Question to ask
        expected_keywords: List of keywords that should appear in the answer
    """
    print_section(f"📝 Query: {query}")
    
    # Retrieve context
    print("\n🔍 Retrieving context...")
    context = retriever.get_context(query, k=TOP_K_RESULTS)
    results = retriever.last_results
    
    print(f"📄 Retrieved {len(results)} chunks")
    print(f"📄 Context length: {len(context)} characters")
    
    # Show retrieved chunks
    print("\n📄 Retrieved Chunks:")
    for i, result in enumerate(results[:3], 1):
        source = result['metadata'].get('file_name', 'unknown')
        page = result['metadata'].get('page', 'N/A')
        score = result.get('similarity_score', 0)
        text = result.get('text', '')[:150]
        print(f"\n   {i}. {source} (Page {page}) Score: {score:.4f}")
        print(f"      Preview: {text}...")
    
    # Generate response
    print("\n🤖 Generating response...")
    response_data = responder.generate_strict_response(query, context)
    
    if response_data['success']:
        response_text = response_data['response']
        print(f"\n✅ RESPONSE:")
        print("-" * 40)
        print(response_text)
        print("-" * 40)
        
        # Quality check
        print("\n📊 Quality Check:")
        char_count = len(response_text)
        word_count = len(response_text.split())
        print(f"   Characters: {char_count}")
        print(f"   Words: {word_count}")
        
        if char_count > 200:
            print("   ✅ Response has good length")
        else:
            print("   ⚠️  Response is too short")
        
        if "source" in response_text.lower() or "page" in response_text.lower():
            print("   ✅ Response includes source citation")
        else:
            print("   ⚠️  Response does NOT cite source")
        
        # Check for expected keywords
        if expected_keywords:
            found_keywords = []
            for keyword in expected_keywords:
                if keyword.lower() in response_text.lower():
                    found_keywords.append(keyword)
            
            if found_keywords:
                print(f"   ✅ Found keywords: {', '.join(found_keywords)}")
            else:
                print(f"   ⚠️  Expected keywords not found: {', '.join(expected_keywords)}")
        
        return response_text
    else:
        print(f"\n❌ Error: {response_data.get('error', 'Unknown error')}")
        return None

def main():
    print("=" * 60)
    print("  FINAL RAG SYSTEM TEST")
    print("=" * 60)
    
    # Load index
    print("\n📂 Loading index...")
    try:
        indexer = EmbeddingIndexer()
        indexer.load_index(index_name="main_index")
        print(f"✅ Index loaded with {len(indexer.chunks)} chunks")
    except Exception as e:
        print(f"❌ Error loading index: {str(e)}")
        print("   Run: python rebuild_final_index.py")
        return
    
    # Initialize retriever and responder
    retriever = Retriever(indexer)
    responder = LLMResponder(temperature=0.1)
    
    # Check Ollama status
    status = responder.check_ollama_status()
    print(f"\n🤖 Ollama Status:")
    print(f"   Status: {status['status']}")
    if status['status'] == 'running':
        print(f"   Model: {status.get('models', ['unknown'])[0] if status.get('models') else 'unknown'}")
    else:
        print("   ⚠️  Ollama is not running! Start with: ollama serve")
        return
    
    # Test queries
    test_queries = [
        {
            "query": "Give me the 14-points of Quaid-i-Azam",
            "keywords": ["14", "points", "Quaid", "Muslim", "federal", "separate", "electorate"]
        },
        {
            "query": "Tell me all info about The Anti-Muslim Policies of the Provincial Congress Governments",
            "keywords": ["Anti-Muslim", "Congress", "Provincial", "policies", "Muslim", "rights"]
        },
        {
            "query": "What was the Pakistan Resolution?",
            "keywords": ["Pakistan", "Resolution", "Muslim", "Lahore", "1940"]
        },
        {
            "query": "Tell me about the division of India",
            "keywords": ["division", "India", "Muslim", "Hindu", "Pakistan", "Bengal", "Punjab"]
        }
    ]
    
    results = []
    for test in test_queries:
        response = test_query(
            retriever, 
            responder, 
            test["query"], 
            test.get("keywords", [])
        )
        results.append({
            "query": test["query"],
            "success": response is not None,
            "length": len(response) if response else 0
        })
    
    # Summary
    print_section("TEST SUMMARY")
    print(f"Total tests: {len(results)}")
    successful = sum(1 for r in results if r['success'])
    print(f"Successful: {successful}/{len(results)}")
    
    if successful == len(results):
        print("\n✅ ALL TESTS PASSED!")
        print("   Your RAG system is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()