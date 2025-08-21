#!/usr/bin/env python3
"""
Simple script to check the persistent embeddings database
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def check_database():
    """Check the persistent embeddings database"""
    storage_path = "./qdrant_storage"
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'pdf_embeddings')
    
    print("=" * 50)
    print("📊 EMBEDDINGS DATABASE STATUS")
    print("=" * 50)
    
    # Check if storage directory exists
    if not Path(storage_path).exists():
        print("❌ No database found!")
        print(f"💡 Run 'python3 create_embeddings_persistent.py' to create embeddings")
        return
    
    try:
        # Connect to persistent storage
        client = QdrantClient(path=storage_path)
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        
        print(f"✅ Database found at: {storage_path}")
        print(f"📁 Collection: {collection_name}")
        print(f"📊 Total embeddings: {collection_info.points_count}")
        print(f"🔢 Vector dimension: {collection_info.config.params.vectors.size}")
        print(f"📏 Distance metric: {collection_info.config.params.vectors.distance}")
        
        # Get sample data
        sample_points = client.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False
        )[0]
        
        print(f"\n📄 Sample documents:")
        for i, point in enumerate(sample_points, 1):
            filename = point.payload.get('filename', 'Unknown')
            doc_type = point.payload.get('type', 'Unknown')
            print(f"   {i}. {filename} ({doc_type})")
        
        print(f"\n🎯 Database is ready for searching!")
        print(f"🔍 Use 'python3 search_documents.py' to search documents")
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")

if __name__ == "__main__":
    check_database()
