#!/usr/bin/env python3
"""
Quality RAG QnA System - Uses improved embeddings with better semantic matching
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from dotenv import load_dotenv

# Vector database
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Gemini API
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QualityRAGQnASystem:
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize the quality RAG QnA system"""
        self.model_name = model_name
        self.storage_path = "./qdrant_storage_quality"
        self.collection_name = "pdf_embeddings_quality"
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.max_context_length = 32000
        self.top_k_results = 10  # Increased for better coverage
        self.min_score_threshold = 0.5  # Lower threshold for quality embeddings
        self.min_text_length = 50
        
        # Initialize components
        self._init_vector_db()
        self._init_embedding_model()
        self._init_gemini_api()
        
    def _init_vector_db(self):
        """Initialize Qdrant vector database"""
        try:
            self.qdrant_client = QdrantClient(path=self.storage_path)
            logger.info(f"✅ Connected to quality Qdrant database at {self.storage_path}")
            
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                raise Exception(f"Quality collection '{self.collection_name}' not found! Run create_quality_embeddings.py first.")
                
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"📊 Quality database contains {collection_info.points_count} embeddings")
            
        except Exception as e:
            logger.error(f"❌ Error initializing vector database: {e}")
            raise
    
    def _init_embedding_model(self):
        """Initialize sentence transformer model"""
        try:
            logger.info(f"📚 Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}")
            raise
    
    def _init_gemini_api(self):
        """Initialize Gemini API"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise Exception("GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            logger.info(f"🔑 Gemini API configured with model: {self.model_name}")
            
            # Test API connection
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content("Hello, test connection.")
            logger.info("✅ Gemini API connection successful")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Gemini API: {e}")
            raise
    
    def search_relevant_documents(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant documents with quality filtering"""
        if top_k is None:
            top_k = self.top_k_results
            
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Filter and format results
            documents = []
            for result in search_results:
                # Skip low relevance scores
                if result.score < self.min_score_threshold:
                    continue
                
                text = result.payload.get('text', '').strip()
                
                # Skip low-quality content
                if len(text) < self.min_text_length:
                    continue
                
                documents.append({
                    'score': result.score,
                    'filename': result.payload.get('filename', 'Unknown'),
                    'text': text,
                    'type': result.payload.get('type', 'Unknown'),
                    'chunk_index': result.payload.get('chunk_index', -1),
                    'page_number': result.payload.get('page_number', 0)
                })
            
            # Sort by score and take top 8
            documents.sort(key=lambda x: x['score'], reverse=True)
            documents = documents[:8]
            
            logger.info(f"📊 Found {len(documents)} quality documents from {len(search_results)} candidates")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error searching documents: {e}")
            return []
    
    def create_context_from_documents(self, documents: List[Dict]) -> str:
        """Create context string from relevant documents with better formatting"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents, 1):
            # Format document information
            doc_info = f"Document {i}: {doc['filename']}"
            if doc['type'] == 'smart_chunk':
                doc_info += f" (Chunk {doc['chunk_index']})"
            if doc['page_number']:
                doc_info += f" (Page {doc['page_number']})"
            doc_info += f" (Relevance: {doc['score']:.3f})"
            
            # Clean and format document text
            doc_text = doc['text'].strip()
            doc_text = re.sub(r'\s+', ' ', doc_text)  # Normalize whitespace
            
            # Check if adding this document would exceed context limit
            estimated_length = len(doc_info) + len(doc_text) + total_length
            if estimated_length > self.max_context_length * 0.8:
                break
                
            context_parts.append(f"{doc_info}\n{doc_text}\n")
            total_length = estimated_length
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """Generate answer using Gemini API with improved prompt"""
        try:
            # Create improved prompt
            prompt = f"""You are a helpful AI assistant with access to a knowledge base of government documents from Pune Municipal Corporation. 

Use the following context to answer the user's question. Be helpful and informative, even if the information is not complete.

Context:
{context}

Question: {question}

Instructions:
1. Answer based on the provided context
2. Be accurate and factual
3. If the context contains relevant information, provide it clearly
4. If the context doesn't contain specific information, provide related information from the context
5. If the question is in Marathi, respond in Marathi
6. If the question is in English, respond in English
7. Be helpful and provide as much useful information as possible from the context
8. If you find specific details like addresses, phone numbers, or procedures, include them

Answer:"""
            
            # Generate response using Gemini
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            
            return {
                'answer': response.text,
                'model_used': self.model_name,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            return {
                'answer': f"Sorry, I encountered an error while generating the answer: {str(e)}",
                'model_used': self.model_name,
                'success': False,
                'error': str(e)
            }
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Main method to ask a question and get answer"""
        try:
            logger.info(f"🔍 Processing question: {question}")
            
            # Step 1: Search for relevant documents
            start_time = time.time()
            relevant_docs = self.search_relevant_documents(question)
            search_time = time.time() - start_time
            
            if not relevant_docs:
                return {
                    'answer': "I couldn't find any relevant documents in the knowledge base to answer your question.",
                    'documents_used': [],
                    'search_time': search_time,
                    'total_time': time.time() - start_time,
                    'model_used': self.model_name
                }
            
            # Step 2: Create context from documents
            context = self.create_context_from_documents(relevant_docs)
            
            # Step 3: Generate answer
            generation_start = time.time()
            answer_result = self.generate_answer(question, context)
            generation_time = time.time() - generation_start
            
            # Step 4: Prepare response
            response = {
                'answer': answer_result['answer'],
                'documents_used': [
                    {
                        'filename': doc['filename'],
                        'relevance_score': doc['score'],
                        'type': doc['type'],
                        'chunk_index': doc['chunk_index'],
                        'page_number': doc['page_number']
                    }
                    for doc in relevant_docs
                ],
                'search_time': search_time,
                'generation_time': generation_time,
                'total_time': time.time() - start_time,
                'model_used': self.model_name,
                'context_length': len(context),
                'success': answer_result.get('success', False)
            }
            
            logger.info(f"✅ Answer generated in {response['total_time']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in ask_question: {e}")
            return {
                'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
                'documents_used': [],
                'search_time': 0,
                'generation_time': 0,
                'total_time': 0,
                'model_used': self.model_name,
                'success': False,
                'error': str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                'model_name': self.model_name,
                'embedding_model': self.embedding_model_name,
                'total_embeddings': collection_info.points_count,
                'vector_dimension': collection_info.config.params.vectors.size,
                'storage_path': self.storage_path,
                'max_context_length': self.max_context_length,
                'top_k_results': self.top_k_results,
                'min_score_threshold': self.min_score_threshold,
                'min_text_length': self.min_text_length
            }
        except Exception as e:
            logger.error(f"❌ Error getting system info: {e}")
            return {}

def select_model() -> str:
    """Interactive model selection"""
    print("\n" + "="*60)
    print("🤖 Quality RAG QnA System - Model Selection")
    print("="*60)
    print("Available Gemini models:")
    print("1. gemini-2.0-flash-exp (Fast, recommended)")
    print("2. gemini-1.5-pro (More capable, slower)")
    print("3. gemini-1.5-flash (Balanced)")
    print("4. gemini-1.0-pro (Legacy)")
    
    while True:
        try:
            choice = input("\nSelect model (1-4) or press Enter for default (1): ").strip()
            if not choice:
                choice = "1"
            
            model_map = {
                "1": "gemini-2.0-flash-exp",
                "2": "gemini-1.5-pro",
                "3": "gemini-1.5-flash",
                "4": "gemini-1.0-pro"
            }
            
            if choice in model_map:
                selected_model = model_map[choice]
                print(f"✅ Selected model: {selected_model}")
                return selected_model
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function for command-line usage"""
    try:
        # Select model
        model_name = select_model()
        
        # Initialize system
        print(f"\n🚀 Initializing Quality RAG QnA System with {model_name}...")
        rag_system = QualityRAGQnASystem(model_name)
        
        # Show system info
        system_info = rag_system.get_system_info()
        print(f"\n📊 System Information:")
        print(f"   Model: {system_info['model_name']}")
        print(f"   Embeddings: {system_info['total_embeddings']}")
        print(f"   Vector Dimension: {system_info['vector_dimension']}")
        print(f"   Min Score Threshold: {system_info['min_score_threshold']}")
        
        # Interactive QnA
        print(f"\n🎯 Ready to answer questions! (Type 'quit' to exit)")
        print("="*60)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Get answer
                print("🤔 Thinking...")
                result = rag_system.ask_question(question)
                
                # Display answer
                print(f"\n💡 Answer:")
                print(f"{result['answer']}")
                
                # Display metadata
                print(f"\n📊 Metadata:")
                print(f"   Time: {result['total_time']:.2f}s")
                print(f"   Documents used: {len(result['documents_used'])}")
                print(f"   Context length: {result['context_length']} chars")
                
                if result['documents_used']:
                    print(f"   Top document: {result['documents_used'][0]['filename']} (Score: {result['documents_used'][0]['relevance_score']:.3f})")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()
