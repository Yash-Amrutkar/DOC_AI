#!/usr/bin/env python3
"""
Document Search Script
Search through PDF embeddings using semantic similarity
"""

import os
import sys
from dotenv import load_dotenv
from create_embeddings import PDFEmbeddingProcessor

# Load environment variables
load_dotenv()

def search_documents():
    """Interactive search function"""
    try:
        # Initialize the processor
        processor = PDFEmbeddingProcessor()
        
        print("=" * 60)
        print("PDF Document Search System")
        print("=" * 60)
        print("Search through your PDF documents using semantic similarity")
        print("Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            # Get search query from user
            query = input("\n🔍 Enter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                print("❌ Please enter a search query")
                continue
            
            # Get number of results
            try:
                top_k = input("📊 Number of results (default 10): ").strip()
                top_k = int(top_k) if top_k else 10
            except ValueError:
                top_k = 10
            
            print(f"\n🔍 Searching for: '{query}'")
            print(f"📊 Returning top {top_k} results...")
            print("-" * 60)
            
            # Perform search
            results = processor.search_similar(query, top_k=top_k)
            
            if not results:
                print("❌ No results found")
                continue
            
            # Display results
            for i, result in enumerate(results, 1):
                print(f"\n{i}. 📄 {result['filename']}")
                print(f"   📊 Similarity Score: {result['score']:.3f}")
                print(f"   📝 Type: {result['type']}")
                if result['chunk_index'] >= 0:
                    print(f"   🔢 Chunk: {result['chunk_index'] + 1}")
                print(f"   📄 Pages: {result['total_pages']}")
                print(f"   💬 Preview: {result['text_preview']}...")
                print("-" * 40)
    
    except KeyboardInterrupt:
        print("\n👋 Search interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Error during search: {e}")

def demo_searches():
    """Run some demo searches"""
    try:
        processor = PDFEmbeddingProcessor()
        
        print("=" * 60)
        print("Demo Searches")
        print("=" * 60)
        
        demo_queries = [
            "environmental report",
            "budget allocation",
            "water supply department",
            "tree plantation",
            "RTI information",
            "municipal commissioner",
            "tax collection",
            "public notice",
            "ward office",
            "fire brigade"
        ]
        
        for query in demo_queries:
            print(f"\n🔍 Searching for: '{query}'")
            print("-" * 40)
            
            results = processor.search_similar(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['filename']} (Score: {result['score']:.3f})")
                    print(f"   Preview: {result['text_preview']}...")
            else:
                print("No results found")
            
            print("-" * 40)
    
    except Exception as e:
        print(f"❌ Error during demo searches: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_searches()
    else:
        search_documents()

if __name__ == "__main__":
    main()
