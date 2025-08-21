#!/usr/bin/env python3
import os
import json
from pathlib import Path

def check_components():
    print("🔍 Q&A SYSTEM DIAGNOSTIC")
    print("="*50)
    
    # 1. Check OCR Quality
    print("\n📄 OCR QUALITY:")
    json_dir = Path("extracted PDF data")
    if json_dir.exists():
        json_files = list(json_dir.glob("*.json"))
        print(f"   ✅ Found {len(json_files)} JSON files")
        
        if json_files:
            try:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                text = data.get('pages', [{}])[0].get('text', '')
                print(f"   📝 Sample text length: {len(text)} characters")
                print(f"   🔤 Text quality: {'Good' if len(text) > 100 else 'Poor'}")
            except:
                print("   ❌ Error reading sample file")
    else:
        print("   ❌ No extracted data found")
    
    # 2. Check Embeddings
    print("\n💾 EMBEDDINGS:")
    storage_paths = ["./qdrant_storage", "./qdrant_storage_quality", "./qdrant_storage_improved"]
    embeddings_found = False
    
    for path in storage_paths:
        if os.path.exists(path):
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(path=path)
                collections = client.get_collections()
                for collection in collections.collections:
                    info = client.get_collection(collection.name)
                    print(f"   ✅ {path}: {info.points_count} embeddings")
                    embeddings_found = True
            except Exception as e:
                print(f"   ❌ {path}: Error - {e}")
    
    if not embeddings_found:
        print("   ❌ No embeddings found")
    
    # 3. Check Gemini API
    print("\n🤖 GEMINI API:")
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print("   ✅ API key found")
    else:
        print("   ❌ No API key found")
    
    # 4. Check RAG System
    print("\n🧠 RAG SYSTEM:")
    try:
        from final_rag_system import FinalRAGQnASystem
        rag = FinalRAGQnASystem()
        print("   ✅ RAG system can be imported")
    except Exception as e:
        print(f"   ❌ RAG system error: {e}")

if __name__ == "__main__":
    check_components()
