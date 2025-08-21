#!/usr/bin/env python3
"""
Final Working RAG System - Handles all issues and works properly
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalWorkingRAG:
    def __init__(self):
        self.storage_path = "./qdrant_storage"
        self.collection_name = "pdf_embeddings"
        # Set your actual API key here
        self.api_key = "AIzaSyBqXqXqXqXqXqXqXqXqXqXqXqXqXqXqXqX"
        os.environ['GEMINI_API_KEY'] = self.api_key
        self._init_components()
        
    def _init_components(self):
        """Initialize all components"""
        try:
            self.client = QdrantClient(path=self.storage_path)
            logger.info("✅ Connected to Qdrant database")
            
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            raise
        
    def search_documents(self, query: str, top_k: int = 15):
        """Search for relevant documents with improved parameters"""
        try:
            query_vector = self.embedding_model.encode(query)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=0.15  # Very low threshold to get more results
            )
            
            documents = []
            for r in results:
                text = r.payload.get("text", "")
                if len(text.strip()) > 10:  # Only include meaningful text
                    documents.append({
                        "text": text[:500],  # Limit text length
                        "score": r.score,
                        "filename": r.payload.get("filename", "Unknown"),
                        "page": r.payload.get("page", 0)
                    })
            
            logger.info(f"🔍 Found {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
        
    def generate_answer(self, question: str, context: str):
        """Generate answer with better error handling"""
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
            headers = {"Content-Type": "application/json"}
            
            # Improved prompt for handling OCR errors
            prompt = f"""You are a helpful assistant analyzing documents. The context may contain OCR errors, mixed languages (English, Marathi, Hindi), and fragmented text. 

Question: {question}

Context from documents:
{context}

Instructions:
1. Answer the question based on the provided context
2. If the context contains OCR errors, try to understand the meaning
3. If the context is unclear, say "I cannot find specific information about this"
4. Be helpful and provide detailed answers when possible
5. If you find relevant information but it's fragmented, piece it together logically

Please provide a clear, helpful answer:"""

            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(f"{url}?key={self.api_key}", headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "I couldn't generate a proper answer from the available information."
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return f"Sorry, I encountered an error (Status: {response.status_code}). Please try again."
                
        except requests.exceptions.Timeout:
            return "Sorry, the request timed out. Please try again."
        except Exception as e:
            logger.error(f"❌ Answer generation error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
            
    def answer_question(self, question: str):
        """Main method to answer questions"""
        logger.info(f"🤔 Question: {question}")
        
        # Search for relevant documents
        docs = self.search_documents(question)
        
        if not docs:
            return "I couldn't find any relevant information in the documents to answer your question."
        
        # Combine context from top documents
        context_parts = []
        for i, doc in enumerate(docs[:8]):  # Use top 8 documents
            context_parts.append(f"Document {i+1}: {doc['filename']} (Page {doc['page']})\n{doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        logger.info(f"✅ Answer generated successfully")
        return answer

def main():
    """Main function for testing"""
    try:
        print("🚀 Initializing Final Working RAG System...")
        rag = FinalWorkingRAG()
        
        print("\n✅ System ready! You can now ask questions.")
        print("💡 Example questions:")
        print("   - What is this document about?")
        print("   - What are the main topics?")
        print("   - Can you summarize the content?")
        print("   - Type 'quit' to exit\n")
        
        while True:
            question = input("🤔 Ask a question: ")
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if question.strip():
                print("🤔 Thinking...")
                answer = rag.answer_question(question)
                print(f"\n✅ Answer: {answer}\n")
            else:
                print("Please enter a question.\n")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
