#!/usr/bin/env python3
"""
PDF Embedding Creation Script - Persistent Version
Uses Qdrant with file storage (embeddings persist on disk)
"""

import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import time
from dotenv import load_dotenv

# Import required libraries
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'embedding_process.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFEmbeddingProcessor:
    def __init__(self, storage_path: str = "./qdrant_storage"):
        """Initialize the embedding processor with persistent storage"""
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'pdf_embeddings')
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION', 384))
        self.batch_size = int(os.getenv('BATCH_SIZE', 32))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
        self.overlap_size = int(os.getenv('OVERLAP_SIZE', 200))
        self.storage_path = storage_path
        
        # Create storage directory
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize Qdrant client with file storage
        logger.info(f"Initializing Qdrant with persistent storage at: {self.storage_path}")
        self.client = QdrantClient(path=self.storage_path)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(self.embedding_model_name)
        
        # Create collection
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection"""
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap_size
            
            if start >= len(text):
                break
        
        return chunks
    
    def _extract_text_from_json(self, json_file_path: str) -> Dict[str, Any]:
        """Extract and process text from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract filename without extension
            filename = Path(json_file_path).stem
            
            # Combine all text from all pages
            all_text = ""
            page_texts = []
            
            for page_num, page_data in data.get('pages', {}).items():
                original_text = page_data.get('original_text', '')
                translations = page_data.get('translations', {})
                
                # Combine original text and translations
                page_text = original_text
                if translations:
                    for lang, translated_text in translations.items():
                        page_text += f"\n{lang.upper()}: {translated_text}"
                
                page_texts.append(page_text)
                all_text += page_text + "\n\n"
            
            return {
                'filename': filename,
                'full_text': all_text.strip(),
                'page_texts': page_texts,
                'total_pages': len(data.get('pages', {}))
            }
            
        except Exception as e:
            logger.error(f"Error processing {json_file_path}: {e}")
            return None
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts"""
        try:
            embeddings = self.model.encode(texts, batch_size=self.batch_size)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def process_json_file(self, json_file_path: str) -> int:
        """Process a single JSON file and store embeddings in Qdrant"""
        logger.info(f"Processing: {json_file_path}")
        
        # Extract text from JSON
        extracted_data = self._extract_text_from_json(json_file_path)
        if not extracted_data:
            return 0
        
        filename = extracted_data['filename']
        full_text = extracted_data['full_text']
        
        # Create chunks from full text
        text_chunks = self._chunk_text(full_text)
        
        # Create embeddings for chunks
        chunk_embeddings = self._create_embeddings(text_chunks)
        
        # Prepare points for Qdrant
        points = []
        point_id = int(time.time() * 1000)  # Use timestamp as base ID
        
        # Add full document embedding
        full_doc_embedding = self._create_embeddings([full_text])[0]
        points.append(PointStruct(
            id=point_id,
            vector=full_doc_embedding,
            payload={
                'filename': filename,
                'text': full_text[:1000],  # First 1000 chars for preview
                'type': 'full_document',
                'total_pages': extracted_data['total_pages'],
                'chunk_index': -1
            }
        ))
        point_id += 1
        
        # Add chunk embeddings
        for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'filename': filename,
                    'text': chunk[:1000],  # First 1000 chars for preview
                    'type': 'chunk',
                    'chunk_index': i,
                    'total_chunks': len(text_chunks),
                    'total_pages': extracted_data['total_pages']
                }
            ))
            point_id += 1
        
        # Store in Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} embeddings for {filename}")
            return len(points)
        except Exception as e:
            logger.error(f"Error storing embeddings for {filename}: {e}")
            return 0
    
    def process_all_files(self, json_data_dir: str) -> Dict[str, int]:
        """Process all JSON files in the directory"""
        json_data_path = Path(json_data_dir)
        if not json_data_path.exists():
            logger.error(f"Directory not found: {json_data_dir}")
            return {}
        
        json_files = list(json_data_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        results = {
            'total_files': len(json_files),
            'processed_files': 0,
            'total_embeddings': 0,
            'failed_files': 0
        }
        
        for i, json_file in enumerate(json_files, 1):
            try:
                logger.info(f"Processing file {i}/{len(json_files)}: {json_file.name}")
                embeddings_count = self.process_json_file(str(json_file))
                
                if embeddings_count > 0:
                    results['processed_files'] += 1
                    results['total_embeddings'] += embeddings_count
                else:
                    results['failed_files'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {json_file.name}: {e}")
                results['failed_files'] += 1
        
        return results
    
    def search_similar(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for similar documents"""
        try:
            # Create embedding for query
            query_embedding = self.model.encode([query])[0].tolist()
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'score': result.score,
                    'filename': result.payload.get('filename'),
                    'text_preview': result.payload.get('text', '')[:200],
                    'type': result.payload.get('type'),
                    'chunk_index': result.payload.get('chunk_index'),
                    'total_pages': result.payload.get('total_pages')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'storage_path': self.storage_path,
                'collection_name': self.collection_name,
                'total_points': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return {}

def main():
    """Main function to run the embedding process"""
    try:
        # Initialize processor with persistent storage
        storage_path = "./qdrant_storage"
        processor = PDFEmbeddingProcessor(storage_path)
        
        # Get JSON data directory from environment
        json_data_dir = os.getenv('JSON_DATA_DIR', './extraceted PDF data')
        
        # Check if embeddings already exist
        storage_info = processor.get_storage_info()
        if storage_info.get('total_points', 0) > 0:
            logger.info(f"Found existing embeddings: {storage_info['total_points']} points")
            logger.info("Processing all files to ensure complete coverage...")
            
            # Clear existing collection to start fresh
            try:
                client = QdrantClient(path=storage_path)
                client.delete_collection(collection_name=processor.collection_name)
                logger.info("Cleared existing collection to start fresh")
            except Exception as e:
                logger.warning(f"Could not clear collection: {e}")
        
        # Process all files
        logger.info("Starting PDF embedding process...")
        results = processor.process_all_files(json_data_dir)
        
        # Print results
        logger.info("=" * 50)
        logger.info("EMBEDDING PROCESS COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Total files found: {results['total_files']}")
        logger.info(f"Successfully processed: {results['processed_files']}")
        logger.info(f"Failed files: {results['failed_files']}")
        logger.info(f"Total embeddings created: {results['total_embeddings']}")
        logger.info(f"Storage location: {storage_path}")
        logger.info("=" * 50)
        
        # Test search functionality
        logger.info("Testing search functionality...")
        test_query = "environmental report"
        search_results = processor.search_similar(test_query, top_k=5)
        
        logger.info(f"Search results for '{test_query}':")
        for i, result in enumerate(search_results, 1):
            logger.info(f"{i}. {result['filename']} (Score: {result['score']:.3f})")
            logger.info(f"   Preview: {result['text_preview']}...")
        
        logger.info(f"\n✅ Embeddings saved to: {storage_path}")
        logger.info("💾 Data is persistent and will be available in future sessions.")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
