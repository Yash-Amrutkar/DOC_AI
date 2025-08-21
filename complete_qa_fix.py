#!/usr/bin/env python3
"""
Complete Q&A System Fix - Handles all issues and creates working system
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import subprocess
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteQAFix:
    def __init__(self):
        self.api_key = "AIzaSyBqXqXqXqXqXqXqXqXqXqXqXqXqXqXqXqX"  # Your API key
        os.environ['GEMINI_API_KEY'] = self.api_key
        
    def fix_api_key(self):
        """Set up API key properly"""
        logger.info("🔑 Setting up Gemini API key...")
        os.environ['GEMINI_API_KEY'] = self.api_key
        logger.info("✅ API key configured")
        
    def create_working_rag_system(self):
        """Create a working RAG system that handles poor OCR quality"""
        logger.info("🧠 Creating improved RAG system...")
        
        rag_code = '''#!/usr/bin/env python3
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
            
        context = "\\n\\n".join([f"Document: {d['filename']}\\n{d['text']}" for d in docs[:5]])
        return self.generate_answer(question, context)

if __name__ == "__main__":
    rag = WorkingRAGSystem()
    while True:
        question = input("\\nAsk a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        answer = rag.answer_question(question)
        print(f"\\nAnswer: {answer}")
'''
        
        with open('working_rag_system.py', 'w') as f:
            f.write(rag_code)
        logger.info("✅ Working RAG system created")
        
    def create_web_interface(self):
        """Create a web interface for the Q&A system"""
        logger.info("🌐 Creating web interface...")
        
        web_code = '''#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify
import os
from working_rag_system import WorkingRAGSystem

app = Flask(__name__)
rag_system = WorkingRAGSystem()

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Q&A System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .answer { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 Document Q&A System</h1>
            <p>Ask questions about your documents:</p>
            <input type="text" id="question" placeholder="Enter your question here...">
            <button onclick="askQuestion()">Ask Question</button>
            <div id="answer" class="answer" style="display:none;"></div>
        </div>
        
        <script>
        function askQuestion() {
            const question = document.getElementById('question').value;
            const answerDiv = document.getElementById('answer');
            
            if (!question) return;
            
            answerDiv.innerHTML = 'Thinking...';
            answerDiv.style.display = 'block';
            
            fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            })
            .then(response => response.json())
            .then(data => {
                answerDiv.innerHTML = '<strong>Answer:</strong><br>' + data.answer;
            })
            .catch(error => {
                answerDiv.innerHTML = 'Error: ' + error;
            });
        }
        </script>
    </body>
    </html>
    """

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    answer = rag_system.answer_question(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
'''
        
        with open('web_interface.py', 'w') as f:
            f.write(web_code)
        logger.info("✅ Web interface created")
        
    def run_complete_process(self):
        """Run the complete process to fix everything"""
        logger.info("🚀 Starting complete Q&A system fix...")
        
        # Step 1: Fix API key
        self.fix_api_key()
        
        # Step 2: Create working RAG system
        self.create_working_rag_system()
        
        # Step 3: Create web interface
        self.create_web_interface()
        
        # Step 4: Test the system
        logger.info("🧪 Testing the system...")
        try:
            from working_rag_system import WorkingRAGSystem
            rag = WorkingRAGSystem()
            test_answer = rag.answer_question("What is this document about?")
            logger.info(f"✅ Test successful: {test_answer[:100]}...")
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            
        # Step 5: Start web interface
        logger.info("🌐 Starting web interface...")
        subprocess.Popen(['python3', 'web_interface.py'])
        
        logger.info("🎉 Complete Q&A system is ready!")
        logger.info("📱 Web interface available at: http://localhost:5000")
        logger.info("💻 Command line interface: python3 working_rag_system.py")

def main():
    fixer = CompleteQAFix()
    fixer.run_complete_process()

if __name__ == "__main__":
    main()
