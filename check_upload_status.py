#!/usr/bin/env python3
"""
Check Upload Status - Verify all documents are uploaded to vector database
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

def check_upload_status():
    """Check if all documents are properly uploaded to the vector database"""
    
    print("="*80)
    print("📊 DOCUMENT UPLOAD STATUS CHECK")
    print("="*80)
    
    # Get file counts
    json_dir = Path("extraceted PDF data")
    pdf_dir = Path("downloads/documents")
    storage_path = "./qdrant_storage"
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'pdf_embeddings')
    
    # Count source files
    json_files = list(json_dir.glob("*.json"))
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    print(f"\n📁 SOURCE FILES:")
    print(f"   📄 PDF files: {len(pdf_files)}")
    print(f"   📄 JSON files: {len(json_files)}")
    print(f"   📊 Missing JSON files: {len(pdf_files) - len(json_files)}")
    
    # Check database
    try:
        client = QdrantClient(path=storage_path)
        collection_info = client.get_collection(collection_name)
        total_embeddings = collection_info.points_count
        
        print(f"\n💾 VECTOR DATABASE:")
        print(f"   📊 Total embeddings: {total_embeddings}")
        print(f"   🔢 Vector dimension: {collection_info.config.params.vectors.size}")
        
        # Get unique filenames from database
        all_points = client.scroll(
            collection_name=collection_name,
            limit=total_embeddings,
            with_payload=True,
            with_vectors=False
        )[0]
        
        db_filenames = set()
        chunk_counts = {}
        
        for point in all_points:
            filename = point.payload.get('filename', 'Unknown')
            doc_type = point.payload.get('type', 'Unknown')
            db_filenames.add(filename)
            
            if filename not in chunk_counts:
                chunk_counts[filename] = {'full_document': 0, 'chunk': 0}
            chunk_counts[filename][doc_type] += 1
        
        print(f"   📄 Unique files in DB: {len(db_filenames)}")
        
        # Compare with source files
        json_filenames = {f.stem for f in json_files}
        
        print(f"\n🔍 COMPARISON:")
        print(f"   📄 JSON files: {len(json_filenames)}")
        print(f"   📄 Files in DB: {len(db_filenames)}")
        
        # Find missing files
        missing_in_db = json_filenames - db_filenames
        extra_in_db = db_filenames - json_filenames
        
        if missing_in_db:
            print(f"\n❌ MISSING IN DATABASE ({len(missing_in_db)} files):")
            for filename in sorted(missing_in_db)[:10]:  # Show first 10
                print(f"   - {filename}")
            if len(missing_in_db) > 10:
                print(f"   ... and {len(missing_in_db) - 10} more")
        
        if extra_in_db:
            print(f"\n⚠️  EXTRA IN DATABASE ({len(extra_in_db)} files):")
            for filename in sorted(extra_in_db)[:10]:  # Show first 10
                print(f"   - {filename}")
            if len(extra_in_db) > 10:
                print(f"   ... and {len(extra_in_db) - 10} more")
        
        # Show chunk distribution
        print(f"\n📊 CHUNK DISTRIBUTION (Sample):")
        sample_files = list(chunk_counts.items())[:10]
        for filename, counts in sample_files:
            total_chunks = counts['full_document'] + counts['chunk']
            print(f"   📄 {filename}: {total_chunks} chunks ({counts['full_document']} full, {counts['chunk']} chunks)")
        
        # Calculate average chunks per file
        if chunk_counts:
            total_chunks = sum(sum(counts.values()) for counts in chunk_counts.values())
            avg_chunks = total_chunks / len(chunk_counts)
            print(f"\n📈 STATISTICS:")
            print(f"   📊 Average chunks per file: {avg_chunks:.1f}")
            print(f"   📊 Total chunks: {total_chunks}")
            print(f"   📊 Files with chunks: {len(chunk_counts)}")
        
        # Overall status
        if not missing_in_db and not extra_in_db:
            print(f"\n✅ STATUS: All {len(json_filenames)} JSON files are uploaded to the database!")
        else:
            print(f"\n⚠️  STATUS: Upload incomplete!")
            print(f"   ✅ Uploaded: {len(json_filenames) - len(missing_in_db)}")
            print(f"   ❌ Missing: {len(missing_in_db)}")
            print(f"   ⚠️  Extra: {len(extra_in_db)}")
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")

def check_sample_document():
    """Check a sample document to see its content and chunks"""
    
    print(f"\n" + "="*80)
    print("🔍 SAMPLE DOCUMENT ANALYSIS")
    print("="*80)
    
    try:
        client = QdrantClient(path="./qdrant_storage")
        collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'pdf_embeddings')
        
        # Get a sample document
        sample_points = client.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False
        )[0]
        
        for i, point in enumerate(sample_points, 1):
            filename = point.payload.get('filename', 'Unknown')
            doc_type = point.payload.get('type', 'Unknown')
            chunk_index = point.payload.get('chunk_index', -1)
            text = point.payload.get('text', '')[:200]  # First 200 chars
            
            print(f"\n📄 Sample {i}: {filename}")
            print(f"   Type: {doc_type}")
            print(f"   Chunk: {chunk_index}")
            print(f"   Text preview: {text}...")
            
    except Exception as e:
        print(f"❌ Error analyzing sample: {e}")

if __name__ == "__main__":
    check_upload_status()
    check_sample_document()
