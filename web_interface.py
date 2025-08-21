#!/usr/bin/env python3
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
