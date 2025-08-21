#!/usr/bin/env python3
"""
Enhanced Web Interface for RAG System
"""

from flask import Flask, render_template, request, jsonify
from enhanced_rag_final import EnhancedRAGFinal
import logging

app = Flask(__name__)

# Initialize the RAG system (prefers High Quality OCR storage)
try:
    rag_system = EnhancedRAGFinal()
    system_ready = True
    system_status = f"✅ Connected to {rag_system.storage_name}"
except Exception as e:
    system_ready = False
    system_status = f"❌ Error: {str(e)}"

@app.route('/')
def index():
    return render_template('enhanced_interface.html', system_ready=system_ready, system_status=system_status)

@app.route('/ask', methods=['POST'])
def ask_question():
    if not system_ready:
        return jsonify({'error': 'System not ready'})
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a question'})
        
        # Get answer from RAG system
        answer = rag_system.answer_question(question)
        
        return jsonify({
            'answer': answer,
            'question': question,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing question: {str(e)}'})

@app.route('/status')
def status():
    return jsonify({
        'system_ready': system_ready,
        'system_status': system_status
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
