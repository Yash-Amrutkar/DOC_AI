#!/usr/bin/env python3
import os
import json
import logging
import requests
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingRAGSystem:
    def __init__(self):
        self.storage_path = "./qdrant_storage"
        self.collection_name = "pdf_embeddings"
        self.api_key = os.getenv('GEMINI_API_KEY')
        self._init_components()
        
    def _init_components(self):
        self.client = QdrantClient(path=self.storage_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def search_documents(self, query: str, top_k: int = 20):
        query_vector = self.embedding_model.encode(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=0.2  # Lower threshold for more results
        )
        return [{"text": r.payload.get("text", ""), "score": r.score, "filename": r.payload.get("filename", "")} for r in results]
        
    def generate_answer(self, question: str, context: str):
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        headers = {"Content-Type": "application/json"}
        
        prompt = f"""Answer this question based on the context provided. The context may contain OCR errors and mixed languages (English, Marathi, Hindi). Do your best to understand and provide a helpful answer.

Question: {question}

Context: {context}

Please provide a clear, helpful answer:"""

        data = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(f"{url}?key={self.api_key}", headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Error: {response.status_code}"
            
    def answer_question(self, question: str):
        docs = self.search_documents(question)
        if not docs:
            return "No relevant documents found."
            
        context = "\n\n".join([f"Document: {d['filename']}\n{d['text']}" for d in docs[:5]])
        return self.generate_answer(question, context)

if __name__ == "__main__":
    rag = WorkingRAGSystem()
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        answer = rag.answer_question(question)
        print(f"\nAnswer: {answer}")
