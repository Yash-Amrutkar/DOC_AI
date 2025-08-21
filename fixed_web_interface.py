#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify
import os
from fixed_rag_system import FixedRAGSystem

app = Flask(__name__)

# Initialize RAG system
try:
    rag_system = FixedRAGSystem()
    system_ready = True
    print("✅ RAG System initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize RAG system: {e}")
    system_ready = False

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Q&A System</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                max-width: 900px; 
                margin: 0 auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                background: white; 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 { 
                color: #333; 
                text-align: center; 
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .status {
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
                text-align: center;
                font-weight: bold;
            }
            .status.ready { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            input[type="text"] { 
                width: 100%; 
                padding: 15px; 
                margin: 15px 0; 
                border: 2px solid #ddd; 
                border-radius: 10px; 
                font-size: 16px;
                box-sizing: border-box;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
            }
            button { 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 10px; 
                cursor: pointer; 
                font-size: 16px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .answer { 
                background: #f8f9fa; 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 10px; 
                border-left: 5px solid #667eea;
                display: none;
                animation: fadeIn 0.5s;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .loading {
                text-align: center;
                color: #667eea;
                font-style: italic;
            }
            .error {
                color: #dc3545;
                background: #f8d7da;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #f5c6cb;
            }
            .success {
                color: #155724;
                background: #d4edda;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #c3e6cb;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 Document Q&A System</h1>
            
            <div class="status """ + ("ready" if system_ready else "error") + """">
                """ + ("✅ System Ready - Ask questions about your documents!" if system_ready else "❌ System Error - Please check the setup") + """
            </div>
            
            <p style="text-align: center; color: #666; margin-bottom: 30px;">
                Ask questions about your PDF documents and get intelligent answers powered by AI.
            </p>
            
            <input type="text" id="question" placeholder="Enter your question here... (e.g., What is this document about?)" 
                   """ + ("disabled" if not system_ready else "") + """>
            <button onclick="askQuestion()" """ + ("disabled" if not system_ready else "") + """>
                🤔 Ask Question
            </button>
            
            <div id="answer" class="answer"></div>
        </div>
        
        <script>
        function askQuestion() {
            const question = document.getElementById('question').value;
            const answerDiv = document.getElementById('answer');
            const button = document.querySelector('button');
            
            if (!question.trim()) {
                alert('Please enter a question.');
                return;
            }
            
            // Show loading state
            answerDiv.innerHTML = '<div class="loading">🤔 Thinking and searching through documents...</div>';
            answerDiv.style.display = 'block';
            button.disabled = true;
            button.textContent = '⏳ Processing...';
            
            // Make API call
            fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    answerDiv.innerHTML = '<div class="error">❌ Error: ' + data.error + '</div>';
                } else {
                    answerDiv.innerHTML = '<div class="success"><strong>✅ Answer:</strong></div><br>' + data.answer.replace(/\\n/g, '<br>');
                }
                button.disabled = false;
                button.textContent = '🤔 Ask Question';
            })
            .catch(error => {
                answerDiv.innerHTML = '<div class="error">❌ Network Error: ' + error.message + '</div>';
                button.disabled = false;
                button.textContent = '🤔 Ask Question';
            });
        }
        
        // Allow Enter key to submit
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
        </script>
    </body>
    </html>
    """

@app.route('/ask', methods=['POST'])
def ask():
    if not system_ready:
        return jsonify({'error': 'System not initialized properly'})
    
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({'error': 'Please provide a question'})
        
        answer = rag_system.answer_question(question)
        return jsonify({'answer': answer})
        
    except Exception as e:
        return jsonify({'error': f'Error processing question: {str(e)}'})

@app.route('/status')
def status():
    return jsonify({
        'system_ready': system_ready,
        'message': 'System is ready' if system_ready else 'System initialization failed'
    })

if __name__ == '__main__':
    print("🚀 Starting Fixed Document Q&A Web Interface...")
    print("📱 Web interface will be available at: http://localhost:5000")
    print("💻 You can now lock your screen - the system will continue running!")
    app.run(host='0.0.0.0', port=5000, debug=False)
