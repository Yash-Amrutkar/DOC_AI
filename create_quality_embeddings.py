#!/usr/bin/env python3
"""
Create Quality Embeddings - Fix OCR and Chunking Issues
Recreates embeddings with better text processing and chunking
"""

import os
import json
import re
import logging
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QualityEmbeddingCreator:
    def __init__(self):
        """Initialize the quality embedding creator"""
        self.storage_path = "./qdrant_storage_quality"
        self.collection_name = "pdf_embeddings_quality"
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.chunk_size = 2000  # Larger chunks for better context
        self.overlap_size = 200
        self.min_chunk_length = 100  # Minimum chunk length
        
        # Initialize components
        self._init_embedding_model()
        self._init_vector_db()
        
    def _init_embedding_model(self):
        """Initialize sentence transformer model"""
        try:
            logger.info(f"📚 Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}")
            raise
    
    def _init_vector_db(self):
        """Initialize Qdrant vector database"""
        try:
            self.qdrant_client = QdrantClient(path=self.storage_path)
            
            # Delete existing collection if it exists
            try:
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                logger.info("🗑️ Deleted existing collection")
            except:
                pass
            
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "size": self.embedding_model.get_sentence_embedding_dimension(),
                    "distance": "Cosine"
                }
            )
            logger.info(f"✅ Created new collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing vector database: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\@\#\$\%\&\*\+\=\|\/\\\<\>\~`\^\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF\u0E00-\u0E7F\u0E80-\u0EFF\u0F00-\u0FFF]', '', text)
        
        # Fix common OCR errors
        text = re.sub(r'[|]{2,}', ' ', text)  # Multiple pipes
        text = re.sub(r'[-]{2,}', '-', text)  # Multiple dashes
        text = re.sub(r'[=]{2,}', '=', text)  # Multiple equals
        text = re.sub(r'[~]{2,}', '~', text)  # Multiple tildes
        
        # Remove isolated characters (likely OCR errors)
        text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Single letters
        text = re.sub(r'\b[0-9]\b', '', text)     # Single digits
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def is_meaningful_text(self, text: str) -> bool:
        """Check if text contains meaningful content"""
        if len(text) < self.min_chunk_length:
            return False
        
        # Remove common low-quality patterns
        low_quality_patterns = [
            r'^[^\w]*$',  # Only special characters
            r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$',  # Just email
            r'^https?://',  # Just URL
            r'^[0-9\s\-\(\)\+]+$',  # Just phone numbers
            r'^[A-Z\s]+$',  # Just uppercase letters
            r'ISO\s+\d+',  # ISO certification text
            r'www\.',  # Website references
            r'@[^\s]+',  # Email addresses
            r'^\s*[|]\s*$',  # Just separators
            r'^\s*[-_]\s*$',  # Just dashes/underscores
        ]
        
        for pattern in low_quality_patterns:
            if re.match(pattern, text):
                return False
        
        # Check for meaningful content indicators
        meaningful_words = [
            'कार्यालय', 'विभाग', 'कर्मचारी', 'अधिकारी', 'नियुक्ती', 'पद', 'कार्य',
            'office', 'department', 'employee', 'officer', 'appointment', 'position', 'work',
            'महापालिका', 'नगरपालिका', 'कॉर्पोरेशन', 'municipal', 'corporation',
            'कायदा', 'नियम', 'परिपत्रक', 'सूचना', 'law', 'rule', 'circular', 'notice',
            'अर्ज', 'प्रमाणपत्र', 'परवाना', 'application', 'certificate', 'license',
            'पत्ता', 'संपर्क', 'फोन', 'ईमेल', 'address', 'contact', 'phone', 'email'
        ]
        
        text_lower = text.lower()
        meaningful_count = sum(1 for word in meaningful_words if word in text_lower)
        
        # Must have at least 3 meaningful words for longer chunks
        return meaningful_count >= 3
    
    def create_smart_chunks(self, text: str, filename: str) -> List[Dict]:
        """Create smart chunks with better text processing"""
        chunks = []
        
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text or len(cleaned_text) < self.min_chunk_length:
            return chunks
        
        # Split by sentences first (better semantic boundaries)
        sentences = re.split(r'[.!?।]+', cleaned_text)
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    # Save current chunk if it's meaningful
                    if self.is_meaningful_text(current_chunk):
                        chunks.append({
                            'text': current_chunk.strip(),
                            'chunk_index': chunk_index,
                            'filename': filename,
                            'type': 'smart_chunk'
                        })
                        chunk_index += 1
                
                # Start new chunk with current sentence
                current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it exists and is meaningful
        if current_chunk and self.is_meaningful_text(current_chunk):
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'filename': filename,
                'type': 'smart_chunk'
            })
        
        return chunks
    
    def process_json_file(self, json_path: Path) -> List[Dict]:
        """Process a single JSON file and create quality chunks"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = json_path.stem
            chunks = []
            
            # Extract text from JSON structure
            if isinstance(data, dict):
                # Handle different JSON structures
                if 'pages' in data:
                    # Multi-page structure
                    for page_num, page_data in enumerate(data['pages']):
                        if isinstance(page_data, dict) and 'text' in page_data:
                            page_text = page_data['text']
                            page_chunks = self.create_smart_chunks(page_text, filename)
                            for chunk in page_chunks:
                                chunk['page_number'] = page_num + 1
                            chunks.extend(page_chunks)
                
                elif 'text' in data:
                    # Single text structure
                    text = data['text']
                    chunks = self.create_smart_chunks(text, filename)
                
                elif 'content' in data:
                    # Content structure
                    content = data['content']
                    if isinstance(content, str):
                        chunks = self.create_smart_chunks(content, filename)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                item_chunks = self.create_smart_chunks(item['text'], filename)
                                chunks.extend(item_chunks)
            
            elif isinstance(data, list):
                # List structure
                for item in data:
                    if isinstance(item, dict) and 'text' in item:
                        item_chunks = self.create_smart_chunks(item['text'], filename)
                        chunks.extend(item_chunks)
            
            logger.info(f"📄 {filename}: Created {len(chunks)} quality chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error processing {json_path}: {e}")
            return []
    
    def create_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Create embeddings for chunks"""
        if not chunks:
            return []
        
        # Extract texts for batch processing
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings in batches
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            
            for j, embedding in enumerate(batch_embeddings):
                chunk_index = i + j
                embeddings.append({
                    'vector': embedding.tolist(),
                    'payload': chunks[chunk_index]
                })
        
        return embeddings
    
    def upload_to_qdrant(self, embeddings: List[Dict]):
        """Upload embeddings to Qdrant"""
        if not embeddings:
            return
        
        # Prepare points for upload
        points = []
        for i, embedding_data in enumerate(embeddings):
            points.append({
                'id': i,
                'vector': embedding_data['vector'],
                'payload': embedding_data['payload']
            })
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"✅ Uploaded {len(points)} embeddings to Qdrant")
    
    def process_all_files(self):
        """Process all JSON files and create quality embeddings"""
        json_dir = Path("extraceted PDF data")
        json_files = list(json_dir.glob("*.json"))
        
        logger.info(f"🚀 Starting quality embedding creation for {len(json_files)} files")
        
        total_chunks = 0
        total_embeddings = 0
        
        for json_file in tqdm(json_files, desc="Processing files"):
            # Process file
            chunks = self.process_json_file(json_file)
            
            if chunks:
                # Create embeddings
                embeddings = self.create_embeddings(chunks)
                
                # Upload to Qdrant
                self.upload_to_qdrant(embeddings)
                
                total_chunks += len(chunks)
                total_embeddings += len(embeddings)
        
        logger.info(f"🎉 Completed! Created {total_chunks} chunks and {total_embeddings} embeddings")
        
        # Show final statistics
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        logger.info(f"📊 Final database: {collection_info.points_count} embeddings")

def main():
    """Main function"""
    try:
        creator = QualityEmbeddingCreator()
        creator.process_all_files()
        
        print(f"\n✅ Quality embeddings created successfully!")
        print(f"📁 Database location: {creator.storage_path}")
        print(f"📊 Collection name: {creator.collection_name}")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()
