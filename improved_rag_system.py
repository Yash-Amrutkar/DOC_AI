#!/usr/bin/env python3
"""
Improved RAG QnA System with Better Content Filtering
Filters out low-quality content and focuses on meaningful information
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
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedRAGQnASystem:
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize the improved RAG QnA system"""
        self.model_name = model_name
        self.storage_path = "./qdrant_storage"
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'pdf_embeddings')
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.max_context_length = 32000
        self.top_k_results = 15  # Increased to get more candidates
        self.min_score_threshold = 0.6  # Minimum relevance score
        self.min_text_length = 50  # Minimum text length to consider
        
        # Initialize components
        self._init_vector_db()
        self._init_embedding_model()
        self._init_gemini_api()
        
    def _init_vector_db(self):
        """Initialize Qdrant vector database"""
        try:
            self.qdrant_client = QdrantClient(path=self.storage_path)
            logger.info(f"✅ Connected to Qdrant database at {self.storage_path}")
            
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
    
    def is_quality_content(self, text: str) -> bool:
        """Check if the text content is of good quality"""
        if not text or len(text.strip()) < self.min_text_length:
            return False
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Check for common low-quality patterns
        low_quality_patterns = [
            r'^[^\w]*$',  # Only special characters
            r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$',  # Just email
            r'^https?://',  # Just URL
            r'^[0-9\s\-\(\)\+]+$',  # Just phone numbers
            r'^[A-Z\s]+$',  # Just uppercase letters
            r'ISO\s+\d+',  # ISO certification text
            r'www\.',  # Website references
            r'@[^\s]+',  # Email addresses
            r'^\s*[|]\s*$',  # Just separators
            r'^\s*[-_]\s*$',  # Just dashes/underscores
        ]
        
        for pattern in low_quality_patterns:
            if re.match(pattern, text):
                return False
        
        # Check for meaningful content indicators
        meaningful_words = [
            'कार्यालय', 'विभाग', 'कर्मचारी', 'अधिकारी', 'नियुक्ती', 'पद', 'कार्य',
            'office', 'department', 'employee', 'officer', 'appointment', 'position', 'work',
            'महापालिका', 'नगरपालिका', 'कॉर्पोरेशन', 'municipal', 'corporation',
            'कायदा', 'नियम', 'परिपत्रक', 'सूचना', 'law', 'rule', 'circular', 'notice',
            'अर्ज', 'प्रमाणपत्र', 'परवाना', 'application', 'certificate', 'license'
        ]
        
        text_lower = text.lower()
        meaningful_count = sum(1 for word in meaningful_words if word in text_lower)
        
        # Must have at least 2 meaningful words
        return meaningful_count >= 2
    
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
                if not self.is_quality_content(text):
                    continue
                
                documents.append({
                    'score': result.score,
                    'filename': result.payload.get('filename', 'Unknown'),
                    'text': text,
                    'type': result.payload.get('type', 'Unknown'),
                    'chunk_index': result.payload.get('chunk_index', -1),
                    'total_pages': result.payload.get('total_pages', 0)
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
            if doc['type'] == 'chunk':
                doc_info += f" (Chunk {doc['chunk_index']})"
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
                'top_k_results': self.top_k_results,
                'min_score_threshold': self.min_score_threshold,
                'min_text_length': self.min_text_length
            }
        except Exception as e:
            logger.error(f"❌ Error getting system info: {e}")
            return {}

def main():
    """Main function for testing the improved system"""
    try:
        # Initialize system
        print(f"\n🚀 Initializing Improved RAG QnA System...")
        rag_system = ImprovedRAGQnASystem()
        
        # Show system info
        system_info = rag_system.get_system_info()
        print(f"\n📊 System Information:")
        print(f"   Model: {system_info['model_name']}")
        print(f"   Embeddings: {system_info['total_embeddings']}")
        print(f"   Min Score Threshold: {system_info['min_score_threshold']}")
        print(f"   Min Text Length: {system_info['min_text_length']}")
        
        # Test with a question
        test_question = "धनकवडी-सहकारनगर कार्यालयाचा पत्ता काय आहे?"
        print(f"\n🧪 Testing with: {test_question}")
        
        result = rag_system.ask_question(test_question)
        
        print(f"\n💡 Answer:")
        print(f"{result['answer']}")
        
        print(f"\n📊 Metadata:")
        print(f"   Time: {result['total_time']:.2f}s")
        print(f"   Documents used: {len(result['documents_used'])}")
        print(f"   Context length: {result['context_length']} chars")
        
        if result['documents_used']:
            print(f"   Top document: {result['documents_used'][0]['filename']} (Score: {result['documents_used'][0]['relevance_score']:.3f})")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()
