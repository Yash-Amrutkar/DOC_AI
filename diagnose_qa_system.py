#!/usr/bin/env python3
"""
Comprehensive Q&A System Diagnostic Tool
Checks each component to identify where the problem is occurring
"""

import os
import json
import logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import requests
from sentence_transformers import SentenceTransformer
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QASystemDiagnostic:
    def __init__(self):
        self.results = {}
        
    def check_ocr_quality(self):
        """Check OCR quality of existing extracted data"""
        logger.info("🔍 Checking OCR Quality...")
        
        json_dir = Path("extracted PDF data")
        if not json_dir.exists():
            self.results['ocr_quality'] = "❌ No extracted data found"
            return
            
        json_files = list(json_dir.glob("*.json"))
        if not json_files:
            self.results['ocr_quality'] = "❌ No JSON files found"
            return
            
        # Sample a few files to check quality
        sample_files = json_files[:3]
        total_chars = 0
        broken_chars = 0
        mixed_languages = 0
        
        for file_path in sample_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for page in data.get('pages', []):
                    text = page.get('text', '')
                    total_chars += len(text)
                    
                    # Check for broken characters
                    broken_chars += len([c for c in text if ord(c) > 127 and not c.isprintable()])
                    
                    # Check for mixed languages (basic check)
                    if any(ord(c) > 127 for c in text) and any(c.isascii() for c in text):
                        mixed_languages += 1
                        
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                
        quality_score = 1 - (broken_chars / max(total_chars, 1))
        self.results['ocr_quality'] = {
            'score': quality_score,
            'total_files': len(json_files),
            'broken_chars_ratio': broken_chars / max(total_chars, 1),
            'mixed_languages_detected': mixed_languages > 0,
            'status': "✅ Good" if quality_score > 0.8 else "⚠️ Poor" if quality_score > 0.5 else "❌ Very Poor"
        }
        
    def check_embeddings_database(self):
        """Check if embeddings exist and are properly stored"""
        logger.info("🔍 Checking Embeddings Database...")
        
        # Check existing storage
        storage_paths = ["./qdrant_storage", "./qdrant_storage_quality", "./qdrant_storage_improved"]
        
        for storage_path in storage_paths:
            if os.path.exists(storage_path):
                try:
                    client = QdrantClient(path=storage_path)
                    collections = client.get_collections()
                    
                    for collection in collections.collections:
                        collection_info = client.get_collection(collection.name)
                        count = collection_info.points_count
                        
                        self.results[f'embeddings_{storage_path.replace("./", "").replace("/", "_")}'] = {
                            'collection': collection.name,
                            'points_count': count,
                            'status': "✅ Good" if count > 0 else "❌ Empty",
                            'storage_path': storage_path
                        }
                        
                except Exception as e:
                    self.results[f'embeddings_{storage_path.replace("./", "").replace("/", "_")}'] = {
                        'error': str(e),
                        'status': "❌ Error",
                        'storage_path': storage_path
                    }
            else:
                self.results[f'embeddings_{storage_path.replace("./", "").replace("/", "_")}'] = {
                    'status': "❌ Not Found",
                    'storage_path': storage_path
                }
                
    def test_semantic_search(self):
        """Test semantic search functionality"""
        logger.info("🔍 Testing Semantic Search...")
        
        # Find the best available storage
        best_storage = None
        best_count = 0
        
        for storage_path in ["./qdrant_storage", "./qdrant_storage_quality", "./qdrant_storage_improved"]:
            if os.path.exists(storage_path):
                try:
                    client = QdrantClient(path=storage_path)
                    collections = client.get_collections()
                    
                    for collection in collections.collections:
                        collection_info = client.get_collection(collection.name)
                        if collection_info.points_count > best_count:
                            best_count = collection_info.points_count
                            best_storage = (storage_path, collection.name)
                            
                except Exception:
                    continue
                    
        if not best_storage:
            self.results['semantic_search'] = "❌ No embeddings found for testing"
            return
            
        try:
            # Test semantic search
            client = QdrantClient(path=best_storage[0])
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Test query
            test_query = "What is the main topic of this document?"
            query_vector = model.encode(test_query)
            
            search_results = client.search(
                collection_name=best_storage[1],
                query_vector=query_vector,
                limit=5
            )
            
            if search_results:
                avg_score = sum(r.score for r in search_results) / len(search_results)
                self.results['semantic_search'] = {
                    'storage_used': best_storage[0],
                    'collection': best_storage[1],
                    'results_found': len(search_results),
                    'average_score': avg_score,
                    'status': "✅ Good" if avg_score > 0.6 else "⚠️ Poor" if avg_score > 0.3 else "❌ Very Poor"
                }
            else:
                self.results['semantic_search'] = "❌ No search results found"
                
        except Exception as e:
            self.results['semantic_search'] = f"❌ Error: {str(e)}"
            
    def test_gemini_api(self):
        """Test Gemini API connectivity and Marathi support"""
        logger.info("🔍 Testing Gemini API...")
        
        # Check if API key is available
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            self.results['gemini_api'] = "❌ No API key found"
            return
            
        try:
            # Test basic API call
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
            headers = {
                "Content-Type": "application/json",
            }
            
            # Test with English
            data = {
                "contents": [{
                    "parts": [{"text": "Hello, how are you?"}]
                }]
            }
            
            response = requests.post(f"{url}?key={api_key}", headers=headers, json=data)
            
            if response.status_code == 200:
                # Test with Marathi
                marathi_data = {
                    "contents": [{
                        "parts": [{"text": "नमस्कार, तुम्ही कसे आहात?"}]
                    }]
                }
                
                marathi_response = requests.post(f"{url}?key={api_key}", headers=headers, json=marathi_data)
                
                self.results['gemini_api'] = {
                    'english_support': "✅ Working",
                    'marathi_support': "✅ Working" if marathi_response.status_code == 200 else "❌ Failed",
                    'status': "✅ Good"
                }
            else:
                self.results['gemini_api'] = f"❌ API Error: {response.status_code}"
                
        except Exception as e:
            self.results['gemini_api'] = f"❌ Connection Error: {str(e)}"
            
    def test_rag_system(self):
        """Test the complete RAG system"""
        logger.info("🔍 Testing RAG System...")
        
        try:
            # Import and test the RAG system
            from final_rag_system import FinalRAGQnASystem
            
            rag_system = FinalRAGQnASystem()
            
            # Test question
            test_question = "What is this document about?"
            result = rag_system.answer_question(test_question)
            
            if result and result.get('answer'):
                answer_length = len(result['answer'])
                self.results['rag_system'] = {
                    'answer_generated': True,
                    'answer_length': answer_length,
                    'status': "✅ Working" if answer_length > 10 else "⚠️ Short Answer"
                }
            else:
                self.results['rag_system'] = "❌ No answer generated"
                
        except Exception as e:
            self.results['rag_system'] = f"❌ Error: {str(e)}"
            
    def run_diagnostic(self):
        """Run all diagnostic checks"""
        logger.info("🚀 Starting Q&A System Diagnostic...")
        
        self.check_ocr_quality()
        self.check_embeddings_database()
        self.test_semantic_search()
        self.test_gemini_api()
        self.test_rag_system()
        
        # Generate summary
        self.generate_summary()
        
    def generate_summary(self):
        """Generate diagnostic summary"""
        logger.info("📊 Generating Diagnostic Summary...")
        
        print("\n" + "="*80)
        print("🔍 Q&A SYSTEM DIAGNOSTIC REPORT")
        print("="*80)
        
        # OCR Quality
        print("\n📄 OCR QUALITY:")
        if 'ocr_quality' in self.results:
            if isinstance(self.results['ocr_quality'], dict):
                print(f"   Status: {self.results['ocr_quality']['status']}")
                print(f"   Quality Score: {self.results['ocr_quality']['score']:.2f}")
                print(f"   Files Found: {self.results['ocr_quality']['total_files']}")
            else:
                print(f"   {self.results['ocr_quality']}")
                
        # Embeddings
        print("\n💾 EMBEDDINGS DATABASE:")
        for key, value in self.results.items():
            if key.startswith('embeddings_'):
                if isinstance(value, dict):
                    print(f"   {key}: {value['status']} ({value.get('points_count', 0)} points)")
                else:
                    print(f"   {key}: {value}")
                    
        # Semantic Search
        print("\n🔍 SEMANTIC SEARCH:")
        if 'semantic_search' in self.results:
            if isinstance(self.results['semantic_search'], dict):
                print(f"   Status: {self.results['semantic_search']['status']}")
                print(f"   Average Score: {self.results['semantic_search'].get('average_score', 0):.3f}")
            else:
                print(f"   {self.results['semantic_search']}")
                
        # Gemini API
        print("\n🤖 GEMINI API:")
        if 'gemini_api' in self.results:
            if isinstance(self.results['gemini_api'], dict):
                print(f"   Status: {self.results['gemini_api']['status']}")
                print(f"   English: {self.results['gemini_api']['english_support']}")
                print(f"   Marathi: {self.results['gemini_api']['marathi_support']}")
            else:
                print(f"   {self.results['gemini_api']}")
                
        # RAG System
        print("\n🧠 RAG SYSTEM:")
        if 'rag_system' in self.results:
            if isinstance(self.results['rag_system'], dict):
                print(f"   Status: {self.results['rag_system']['status']}")
                print(f"   Answer Generated: {self.results['rag_system']['answer_generated']}")
            else:
                print(f"   {self.results['rag_system']}")
                
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        self.generate_recommendations()
        
        print("\n" + "="*80)
        
    def generate_recommendations(self):
        """Generate recommendations based on diagnostic results"""
        recommendations = []
        
        # Check OCR quality
        if 'ocr_quality' in self.results and isinstance(self.results['ocr_quality'], dict):
            if self.results['ocr_quality']['score'] < 0.7:
                recommendations.append("🔧 Improve OCR quality - current quality is poor")
                
        # Check embeddings
        embeddings_found = False
        for key, value in self.results.items():
            if key.startswith('embeddings_') and isinstance(value, dict) and value.get('points_count', 0) > 0:
                embeddings_found = True
                break
                
        if not embeddings_found:
            recommendations.append("💾 Create embeddings - no embeddings found")
            
        # Check semantic search
        if 'semantic_search' in self.results and isinstance(self.results['semantic_search'], dict):
            if self.results['semantic_search'].get('average_score', 0) < 0.5:
                recommendations.append("🔍 Improve semantic search - low matching scores")
                
        # Check Gemini API
        if 'gemini_api' in self.results and isinstance(self.results['gemini_api'], dict):
            if self.results['gemini_api']['marathi_support'] == "❌ Failed":
                recommendations.append("🌐 Check Marathi language support in Gemini API")
                
        # Check RAG system
        if 'rag_system' in self.results and isinstance(self.results['rag_system'], dict):
            if not self.results['rag_system'].get('answer_generated', False):
                recommendations.append("🧠 Fix RAG system - not generating answers")
                
        if not recommendations:
            recommendations.append("✅ All components appear to be working correctly")
            
        for rec in recommendations:
            print(f"   {rec}")

if __name__ == "__main__":
    diagnostic = QASystemDiagnostic()
    diagnostic.run_diagnostic()
