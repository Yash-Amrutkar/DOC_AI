#!/usr/bin/env python3
"""
RAG-based QnA System with Gemini 2.5 Pro API
Uses Qdrant vector database as external knowledge base
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from dotenv import load_dotenv

# Vector database
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Gemini API
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Web interface
from flask import Flask, render_template, request, jsonify, session
import threading
import webbrowser

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGQnASystem:
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize the RAG QnA system"""
        self.model_name = model_name
        self.storage_path = "./qdrant_storage"
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'pdf_embeddings')
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.max_context_length = 32000  # Gemini 2.5 Pro context limit
        self.top_k_results = 8  # Increased from 5 to get more context
        
        # Initialize components
        self._init_vector_db()
        self._init_embedding_model()
        self._init_gemini_api()
        
    def _init_vector_db(self):
        """Initialize Qdrant vector database"""
        try:
            self.qdrant_client = QdrantClient(path=self.storage_path)
            logger.info(f"✅ Connected to Qdrant database at {self.storage_path}")
            
            # Check collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                raise Exception(f"Collection '{self.collection_name}' not found!")
                
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"📊 Database contains {collection_info.points_count} embeddings")
            
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
        """Search for relevant documents in vector database"""
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
            
            # Format results
            documents = []
            for result in search_results:
                documents.append({
                    'score': result.score,
                    'filename': result.payload.get('filename', 'Unknown'),
                    'text': result.payload.get('text', ''),
                    'type': result.payload.get('type', 'Unknown'),
                    'chunk_index': result.payload.get('chunk_index', -1),
                    'total_pages': result.payload.get('total_pages', 0)
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error searching documents: {e}")
            return []
    
    def create_context_from_documents(self, documents: List[Dict]) -> str:
        """Create context string from relevant documents"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents, 1):
            # Format document information
            doc_info = f"Document {i}: {doc['filename']}"
            if doc['type'] == 'chunk':
                doc_info += f" (Chunk {doc['chunk_index']})"
            doc_info += f" (Relevance: {doc['score']:.3f})"
            
            # Add document text
            doc_text = doc['text'].strip()
            
            # Clean and format the text better
            doc_text = doc_text.replace('\n\n', '\n').replace('\n\n\n', '\n').strip()
            
            # Check if adding this document would exceed context limit
            estimated_length = len(doc_info) + len(doc_text) + total_length
            if estimated_length > self.max_context_length * 0.8:  # Leave 20% for prompt
                break
                
            context_parts.append(f"{doc_info}\n{doc_text}\n")
            total_length = estimated_length
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """Generate answer using Gemini API"""
        try:
            # Create prompt
            prompt = f"""You are a helpful AI assistant with access to a knowledge base of government documents from Pune Municipal Corporation. 

Use the following context to answer the user's question. Be helpful and provide the best possible answer based on the available information.

Context:
{context}

Question: {question}

Instructions:
1. Answer based on the provided context
2. Be helpful and informative - provide any relevant information you can find
3. If the context contains related information, use it to provide a helpful response
4. If the context doesn't contain the specific information requested, acknowledge this but provide any related information that might be useful
5. Be confident in sharing what you do know from the documents
6. If the question is in Marathi, respond in Marathi
7. If the question is in English, respond in English
8. Always try to be helpful rather than saying you don't have enough information

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
                        'chunk_index': doc['chunk_index']
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
                'top_k_results': self.top_k_results
            }
        except Exception as e:
            logger.error(f"❌ Error getting system info: {e}")
            return {}

def select_model() -> str:
    """Interactive model selection"""
    print("\n" + "="*60)
    print("🤖 RAG QnA System - Model Selection")
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
        print(f"\n🚀 Initializing RAG QnA System with {model_name}...")
        rag_system = RAGQnASystem(model_name)
        
        # Show system info
        system_info = rag_system.get_system_info()
        print(f"\n📊 System Information:")
        print(f"   Model: {system_info['model_name']}")
        print(f"   Embeddings: {system_info['total_embeddings']}")
        print(f"   Vector Dimension: {system_info['vector_dimension']}")
        
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
