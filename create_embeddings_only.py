#!/usr/bin/env python3
"""
Create embeddings from improved OCR data
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingCreator:
    def __init__(self):
        self.input_dir = "improved_ocr_data_v2"
        self.embeddings_dir = "qdrant_storage_v2"
        self.collection_name = "pdf_embeddings_v2"
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_chunks(self, text: str, filename: str, page: str, chunk_size: int = 1000) -> List[Dict]:
        """Create text chunks for embedding"""
        chunks = []
        
        # Split by sentences first
        sentences = text.split('. ')
        
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append({
                        'id': f"{filename}_{page}_chunk{chunk_id}",
                        'text': current_chunk.strip(),
                        'filename': filename,
                        'page': page,
                        'chunk_id': chunk_id
                    })
                    chunk_id += 1
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'id': f"{filename}_{page}_chunk{chunk_id}",
                'text': current_chunk.strip(),
                'filename': filename,
                'page': page,
                'chunk_id': chunk_id
            })
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> None:
        """Create embeddings and store in Qdrant"""
        try:
            # Initialize Qdrant client
            client = QdrantClient(path=self.embeddings_dir)
            
            # Create collection if it doesn't exist
            try:
                client.get_collection(self.collection_name)
                logger.info(f"✅ Using existing collection: {self.collection_name}")
            except:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.info(f"✅ Created new collection: {self.collection_name}")
            
            # Process chunks in batches
            batch_size = 100
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(chunks), batch_size), desc="Creating embeddings"):
                batch = chunks[i:i + batch_size]
                
                # Create embeddings
                texts = [chunk['text'] for chunk in batch]
                embeddings = self.embedding_model.encode(texts)
                
                # Prepare points for Qdrant
                points = []
                for j, chunk in enumerate(batch):
                    points.append({
                        'id': chunk['id'],
                        'vector': embeddings[j].tolist(),
                        'payload': {
                            'text': chunk['text'],
                            'filename': chunk['filename'],
                            'page': chunk['page'],
                            'chunk_id': chunk['chunk_id']
                        }
                    })
                
                # Upsert to Qdrant
                client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            # Get final count
            collection_info = client.get_collection(self.collection_name)
            logger.info(f"✅ Created embeddings for {len(chunks)} chunks")
            logger.info(f"✅ Total embeddings in collection: {collection_info.points_count}")
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings: {e}")
            raise
    
    def run_pipeline(self):
        """Run the embedding creation pipeline"""
        logger.info("🚀 Starting Embedding Creation Pipeline")
        
        # Find all improved JSON files
        json_files = list(Path(self.input_dir).glob("*_improved.json"))
        logger.info(f"📁 Found {len(json_files)} improved JSON files to process")
        
        all_chunks = []
        processed_count = 0
        
        for json_file in tqdm(json_files, desc="Processing files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Create chunks for each page
                for page in data.get('pages', []):
                    text = page.get('text', '')
                    if text.strip():
                        chunks = self.create_chunks(
                            text,
                            data['filename'],
                            page['page_number']
                        )
                        all_chunks.extend(chunks)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {json_file}: {e}")
                continue
        
        logger.info(f"📊 Created {len(all_chunks)} chunks from {processed_count} files")
        
        # Create embeddings
        if all_chunks:
            logger.info("🧠 Creating embeddings...")
            self.create_embeddings(all_chunks)
            logger.info("🎉 Pipeline completed successfully!")
        else:
            logger.error("❌ No chunks created, pipeline failed")

def main():
    """Main function"""
    try:
        creator = EmbeddingCreator()
        creator.run_pipeline()
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()
