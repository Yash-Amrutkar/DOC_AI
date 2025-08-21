#!/usr/bin/env python3
import os
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class BetterRAG:
    def __init__(self):
        self.client = QdrantClient(path="./qdrant_storage")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.api_key = os.getenv('GEMINI_API_KEY')
        
    def search(self, query, top_k=20):
        query_vector = self.model.encode(query)
        results = self.client.search(
            collection_name="pdf_embeddings",
            query_vector=query_vector,
            limit=top_k,
            score_threshold=0.2
        )
        return results
        
    def answer(self, question):
        # Search for relevant docs
        docs = self.search(question)
        
        if not docs:
            return "No relevant documents found."
            
        # Combine context
        context = "\n".join([doc.payload.get('text', '') for doc in docs[:10]])
        
        # Call Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        headers = {"Content-Type": "application/json"}
        
        prompt = f"""Answer this question based on the context (which may contain OCR errors):

Question: {question}

Context: {context}

Provide a helpful answer:"""
        
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(f"{url}?key={self.api_key}", headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"API Error: {response.status_code}"

if __name__ == "__main__":
    rag = BetterRAG()
    print("Testing RAG system...")
    result = rag.answer("What is this document about?")
    print(f"Answer: {result}")
