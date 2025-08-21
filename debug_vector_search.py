#!/usr/bin/env python3
"""
Debug Script to Show Vector Database Results
Shows exactly what content is being retrieved from Qdrant for any query
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class VectorSearchDebugger:
    def __init__(self):
        """Initialize the debugger"""
        self.storage_path = "./qdrant_storage"
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'pdf_embeddings')
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.top_k_results = 8
        
        # Initialize components
        self._init_vector_db()
        self._init_embedding_model()
        
    def _init_vector_db(self):
        """Initialize Qdrant vector database"""
        try:
            self.qdrant_client = QdrantClient(path=self.storage_path)
            print(f"✅ Connected to Qdrant database at {self.storage_path}")
            
            # Check collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                raise Exception(f"Collection '{self.collection_name}' not found!")
                
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"📊 Database contains {collection_info.points_count} embeddings")
            
        except Exception as e:
            print(f"❌ Error initializing vector database: {e}")
            raise
    
    def _init_embedding_model(self):
        """Initialize sentence transformer model"""
        try:
            print(f"📚 Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    
    def search_and_show_results(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant documents and show detailed results"""
        if top_k is None:
            top_k = self.top_k_results
            
        print(f"\n🔍 Searching for: '{query}'")
        print("="*80)
        
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Show detailed results
            documents = []
            for i, result in enumerate(search_results, 1):
                doc_info = {
                    'rank': i,
                    'score': result.score,
                    'filename': result.payload.get('filename', 'Unknown'),
                    'text': result.payload.get('text', ''),
                    'type': result.payload.get('type', 'Unknown'),
                    'chunk_index': result.payload.get('chunk_index', -1),
                    'total_pages': result.payload.get('total_pages', 0)
                }
                documents.append(doc_info)
                
                # Display result
                print(f"\n📄 RESULT #{i}")
                print(f"   📁 File: {doc_info['filename']}")
                print(f"   🎯 Score: {doc_info['score']:.4f}")
                print(f"   📋 Type: {doc_info['type']}")
                if doc_info['chunk_index'] != -1:
                    print(f"   🔢 Chunk: {doc_info['chunk_index']}")
                print(f"   📄 Pages: {doc_info['total_pages']}")
                print(f"   📝 Content Preview:")
                print("   " + "-"*60)
                
                # Show text content (truncated for readability)
                text = doc_info['text'].strip()
                if len(text) > 500:
                    print(f"   {text[:500]}...")
                    print(f"   [Content truncated - {len(text)} characters total]")
                else:
                    print(f"   {text}")
                print("   " + "-"*60)
            
            return documents
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def show_context_sent_to_llm(self, query: str, documents: List[Dict]) -> str:
        """Show exactly what context is being sent to the LLM"""
        print(f"\n🤖 CONTEXT BEING SENT TO LLM")
        print("="*80)
        
        if not documents:
            print("❌ No documents found!")
            return "No relevant documents found."
        
        context_parts = []
        total_length = 0
        max_context_length = 32000 * 0.8  # 80% of Gemini's context limit
        
        for i, doc in enumerate(documents, 1):
            # Format document information
            doc_info = f"Document {i}: {doc['filename']}"
            if doc['type'] == 'chunk':
                doc_info += f" (Chunk {doc['chunk_index']})"
            doc_info += f" (Relevance: {doc['score']:.3f})"
            
            # Add document text
            doc_text = doc['text'].strip()
            
            # Check if adding this document would exceed context limit
            estimated_length = len(doc_info) + len(doc_text) + total_length
            if estimated_length > max_context_length:
                print(f"⚠️  Document {i} would exceed context limit - stopping here")
                break
                
            context_parts.append(f"{doc_info}\n{doc_text}\n")
            total_length = estimated_length
            
            print(f"📄 Including Document {i}: {doc['filename']}")
            print(f"   Length: {len(doc_text)} characters")
            print(f"   Running total: {total_length} characters")
        
        final_context = "\n".join(context_parts)
        
        print(f"\n📊 FINAL CONTEXT SUMMARY:")
        print(f"   Total documents included: {len(context_parts)}")
        print(f"   Total context length: {len(final_context)} characters")
        print(f"   Context limit: {max_context_length:.0f} characters")
        
        print(f"\n📝 FULL CONTEXT BEING SENT TO LLM:")
        print("="*80)
        print(final_context)
        print("="*80)
        
        return final_context

def main():
    """Main function"""
    try:
        # Initialize debugger
        debugger = VectorSearchDebugger()
        
        print("\n🎯 Vector Database Debug Tool")
        print("="*60)
        print("This tool shows exactly what the vector database returns")
        print("and what context is sent to the LLM for any query.")
        
        while True:
            try:
                query = input("\n❓ Enter your query (or 'quit' to exit): ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Search and show results
                documents = debugger.search_and_show_results(query)
                
                # Show context being sent to LLM
                context = debugger.show_context_sent_to_llm(query, documents)
                
                print(f"\n✅ Analysis complete for query: '{query}'")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()
