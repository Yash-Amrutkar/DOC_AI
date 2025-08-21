#!/usr/bin/env python3
"""
Final Comprehensive Embedding Creator
This script will create high-quality embeddings from all available data sources
"""

import os
import json
import logging
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalEmbeddingCreator:
    def __init__(self):
        self.input_dir = "extraceted PDF data"
        self.improved_dir = "improved_ocr_data_v2"
        self.embeddings_dir = "qdrant_storage_v2"
        self.collection_name = "pdf_embeddings_v2"
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove very short texts
        if len(text) < 10:
            return ""
            
        return text
    
    def generate_uuid(self, text: str) -> str:
        """Generate a UUID-like string from text"""
        hash_object = hashlib.md5(text.encode())
        hash_hex = hash_object.hexdigest()
        # Format as UUID: 8-4-4-4-12
        return f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    
    def create_chunks(self, text: str, filename: str, page: str, chunk_size: int = 800) -> List[Dict]:
        """Create text chunks for embedding"""
        chunks = []
        
        # Clean the text first
        text = self.clean_text(text)
        if not text:
            return chunks
        
        # Split by sentences first
        sentences = text.split('. ')
        
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunk_id_str = f"{filename}_{page}_chunk{chunk_id}"
                    chunks.append({
                        'id': self.generate_uuid(chunk_id_str),
                        'text': current_chunk.strip(),
                        'filename': filename,
                        'page': page,
                        'chunk_id': chunk_id
                    })
                    chunk_id += 1
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_id_str = f"{filename}_{page}_chunk{chunk_id}"
            chunks.append({
                'id': self.generate_uuid(chunk_id_str),
                'text': current_chunk.strip(),
                'filename': filename,
                'page': page,
                'chunk_id': chunk_id
            })
        
        return chunks
    
    def process_original_json(self, json_path: Path) -> List[Dict]:
        """Process original JSON files"""
        chunks = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = json_path.stem
            
            # Handle the correct JSON structure
            pages = data.get('pages', {})
            
            for page_key, page_data in pages.items():
                if isinstance(page_data, dict):
                    # Get original text and translations
                    original_text = page_data.get('original_text', '')
                    translations = page_data.get('translations', {})
                    english_text = translations.get('english', '')
                    
                    # Combine original and English text for better coverage
                    combined_text = original_text
                    if english_text and english_text != original_text:
                        combined_text += "\n\nEnglish Translation:\n" + english_text
                    
                    # Create chunks
                    page_chunks = self.create_chunks(combined_text, filename, page_key)
                    chunks.extend(page_chunks)
            
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
        
        return chunks
    
    def process_improved_json(self, json_path: Path) -> List[Dict]:
        """Process improved JSON files"""
        chunks = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = data.get('filename', json_path.stem)
            
            # Process pages
            for page in data.get('pages', []):
                text = page.get('text', '')
                page_number = page.get('page_number', 'unknown')
                
                # Create chunks
                page_chunks = self.create_chunks(text, filename, page_number)
                chunks.extend(page_chunks)
            
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> None:
        """Create embeddings and store in Qdrant"""
        try:
            # Initialize Qdrant client
            client = QdrantClient(path=self.embeddings_dir)
            
            # Delete existing collection if it exists
            try:
                client.delete_collection(self.collection_name)
                logger.info(f"🗑️ Deleted existing collection: {self.collection_name}")
            except:
                pass
            
            # Create new collection
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"✅ Created new collection: {self.collection_name}")
            
            # Process chunks in batches
            batch_size = 50  # Smaller batch size for better memory management
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            logger.info(f"📊 Processing {len(chunks)} chunks in {total_batches} batches...")
            
            for i in tqdm(range(0, len(chunks), batch_size), desc="Creating embeddings"):
                batch = chunks[i:i + batch_size]
                
                # Create embeddings
                texts = [chunk['text'] for chunk in batch]
                embeddings = self.embedding_model.encode(texts)
                
                # Prepare points for Qdrant - using the correct format
                points = []
                for j, chunk in enumerate(batch):
                    points.append(PointStruct(
                        id=chunk['id'],
                        vector=embeddings[j].tolist(),
                        payload={
                            'text': chunk['text'],
                            'filename': chunk['filename'],
                            'page': chunk['page'],
                            'chunk_id': chunk['chunk_id']
                        }
                    ))
                
                # Upsert to Qdrant using the correct method
                client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            
            # Get final count
            collection_info = client.get_collection(self.collection_name)
            logger.info(f"✅ Created embeddings for {len(chunks)} chunks")
            logger.info(f"✅ Total embeddings in collection: {collection_info.points_count}")
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings: {e}")
            raise
    
    def run_complete_pipeline(self):
        """Run the complete embedding creation pipeline"""
        logger.info("🚀 Starting Final Embedding Creation Pipeline")
        
        all_chunks = []
        
        # Step 1: Process original JSON files
        logger.info("📁 Processing original JSON files...")
        original_files = list(Path(self.input_dir).glob("*.json"))
        logger.info(f"Found {len(original_files)} original JSON files")
        
        for json_file in tqdm(original_files, desc="Processing original files"):
            chunks = self.process_original_json(json_file)
            all_chunks.extend(chunks)
        
        logger.info(f"📊 Created {len(all_chunks)} chunks from original files")
        
        # Step 2: Process improved JSON files if they exist
        if os.path.exists(self.improved_dir):
            logger.info("📁 Processing improved JSON files...")
            improved_files = list(Path(self.improved_dir).glob("*_improved.json"))
            logger.info(f"Found {len(improved_files)} improved JSON files")
            
            improved_chunks = []
            for json_file in tqdm(improved_files, desc="Processing improved files"):
                chunks = self.process_improved_json(json_file)
                improved_chunks.extend(chunks)
            
            logger.info(f"📊 Created {len(improved_chunks)} chunks from improved files")
            
            # Combine chunks (prefer improved chunks if they exist for the same file)
            # Create a map of existing chunks by filename
            chunk_map = {}
            for chunk in all_chunks:
                key = f"{chunk['filename']}_{chunk['page']}"
                chunk_map[key] = chunk
            
            # Add improved chunks, replacing existing ones
            for chunk in improved_chunks:
                key = f"{chunk['filename']}_{chunk['page']}"
                chunk_map[key] = chunk
            
            all_chunks = list(chunk_map.values())
        
        logger.info(f"📊 Total chunks to process: {len(all_chunks)}")
        
        # Step 3: Create embeddings
        if all_chunks:
            logger.info("🧠 Creating embeddings...")
            self.create_embeddings(all_chunks)
            logger.info("🎉 Pipeline completed successfully!")
            
            # Final verification
            client = QdrantClient(path=self.embeddings_dir)
            collection_info = client.get_collection(self.collection_name)
            logger.info(f"✅ Final verification: {collection_info.points_count} embeddings in collection")
        else:
            logger.error("❌ No chunks created, pipeline failed")

def main():
    """Main function"""
    try:
        creator = FinalEmbeddingCreator()
        creator.run_complete_pipeline()
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
