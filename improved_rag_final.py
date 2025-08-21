#!/usr/bin/env python3
"""
Improved RAG System that works with existing poor quality embeddings
Uses advanced techniques to handle OCR errors and improve answers
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedRAGQnASystem:
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.storage_path = "./qdrant_storage"
        self.collection_name = "pdf_embeddings"
        self.top_k_results = 25  # Increased for better coverage
        self.min_score_threshold = 0.25  # Lowered to get more results
        self.min_text_length = 5  # Very low to include more text
        
        # Initialize components
        self._init_qdrant()
        self._init_embedding_model()
        self._init_gemini_api(model_name)
        
    def _init_qdrant(self):
        """Initialize Qdrant client"""
        try:
            self.client = QdrantClient(path=self.storage_path)
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"✅ Connected to Qdrant database with {collection_info.points_count} embeddings")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise
            
    def _init_embedding_model(self):
        """Initialize embedding model"""
        try:
            logger.info("📚 Loading embedding model: all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
            
    def _init_gemini_api(self, model_name: str):
        """Initialize Gemini API"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY environment variable not set")
            
        self.model_name = model_name
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        # Test API connection
        try:
            test_response = self._call_gemini_api("Hello")
            logger.info(f"✅ Gemini API connection successful with model: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Gemini API: {e}")
            raise
            
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API"""
        url = f"{self.base_url}/{self.model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(f"{url}?key={self.api_key}", headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
            
    def clean_text_for_display(self, text: str) -> str:
        """Clean text for better display"""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR errors
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\@\#\$\%\&\*\+\=\|\/\\\<\>\~]', '', text)
        
        return text[:500] + "..." if len(text) > 500 else text
        
    def search_relevant_documents(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant documents with improved filtering"""
        if top_k is None:
            top_k = self.top_k_results
            
        try:
            # Encode query
            query_vector = self.embedding_model.encode(query)
            
            # Search with lower threshold to get more results
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k * 2,  # Get more results for filtering
                score_threshold=self.min_score_threshold
            )
            
            # Filter and clean results
            filtered_results = []
            for result in search_results:
                payload = result.payload
                text = payload.get('text', '')
                
                # Clean the text
                cleaned_text = self.clean_text_for_display(text)
                
                if len(cleaned_text) >= self.min_text_length:
                    filtered_results.append({
                        'text': cleaned_text,
                        'score': result.score,
                        'filename': payload.get('filename', 'Unknown'),
                        'page': payload.get('page', 0)
                    })
                    
                if len(filtered_results) >= top_k:
                    break
                    
            logger.info(f"🔍 Found {len(filtered_results)} relevant documents (scores: {[r['score']:.3f for r in filtered_results[:3]]})")
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
            
    def generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """Generate answer with improved prompt engineering"""
        try:
            # Enhanced prompt for handling poor OCR quality
            prompt = f"""You are an expert assistant analyzing documents that may contain OCR errors, mixed languages (English, Marathi, Hindi), and fragmented text. 

Question: {question}

Context from documents (may contain OCR errors and mixed languages):
{context}

Instructions:
1. Answer the question based on the provided context
2. If the context contains OCR errors or mixed languages, try to understand the meaning and provide the best possible answer
3. If the context is unclear or doesn't contain relevant information, say "I cannot find specific information about this in the provided documents"
4. If the context contains multiple languages, respond in the same language as the question
5. Be helpful and provide detailed answers when possible
6. If you find relevant information but it's fragmented, try to piece it together logically

Please provide a clear, helpful answer:"""

            response = self._call_gemini_api(prompt)
            
            return {
                'answer': response,
                'question': question,
                'context_length': len(context),
                'model_used': self.model_name
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate answer: {e}")
            return {
                'answer': f"Sorry, I encountered an error while generating the answer: {str(e)}",
                'question': question,
                'error': str(e)
            }
            
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Main method to answer a question"""
        logger.info(f"🤔 Question: {question}")
        
        # Search for relevant documents
        relevant_docs = self.search_relevant_documents(question)
        
        if not relevant_docs:
            return {
                'answer': "I couldn't find any relevant information in the documents to answer your question.",
                'question': question,
                'context': "No relevant documents found",
                'sources': []
            }
            
        # Combine context from top documents
        context_parts = []
        sources = []
        
        for doc in relevant_docs[:10]:  # Use top 10 documents
            context_parts.append(f"Document: {doc['filename']} (Page {doc['page']})\n{doc['text']}")
            sources.append({
                'filename': doc['filename'],
                'page': doc['page'],
                'score': doc['score']
            })
            
        context = "\n\n".join(context_parts)
        
        # Generate answer
        result = self.generate_answer(question, context)
        result['sources'] = sources
        result['context'] = context[:1000] + "..." if len(context) > 1000 else context
        
        logger.info(f"✅ Answer generated successfully")
        return result

def main():
    """Main function for testing"""
    try:
        # Initialize system
        rag_system = ImprovedRAGQnASystem()
        
        # Test questions
        test_questions = [
            "What is the main topic of these documents?",
            "What are the key points mentioned?",
            "Can you summarize the content?"
        ]
        
        print("\n🧪 Testing Improved RAG System")
        print("="*50)
        
        for question in test_questions:
            print(f"\n🤔 Question: {question}")
            result = rag_system.answer_question(question)
            print(f"✅ Answer: {result['answer'][:200]}...")
            print(f"📊 Sources: {len(result['sources'])} documents")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
