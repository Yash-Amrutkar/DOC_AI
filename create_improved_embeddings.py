#!/usr/bin/env python3
"""
Create Improved Embeddings - Work with existing poor OCR quality
Creates better embeddings from existing JSON files with improved processing
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

class ImprovedEmbeddingCreator:
    def __init__(self):
        """Initialize the improved embedding creator"""
        self.storage_path = "./qdrant_storage_improved"
        self.collection_name = "pdf_embeddings_improved"
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.chunk_size = 1500  # Smaller chunks for better handling
        self.overlap_size = 100
        self.min_chunk_length = 30  # Much lower threshold
        
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
    
    def clean_text_basic(self, text: str) -> str:
        """Basic text cleaning that preserves more content"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove only the most problematic OCR artifacts
        text = re.sub(r'[|]{3,}', ' ', text)  # Multiple pipes
        text = re.sub(r'[-]{3,}', '-', text)  # Multiple dashes
        text = re.sub(r'[=]{3,}', '=', text)  # Multiple equals
        
        # Remove isolated single characters (but keep meaningful ones)
        text = re.sub(r'\b[a-zA-Z]\b(?!\w)', '', text)  # Single letters not followed by word chars
        text = re.sub(r'\b[0-9]\b(?!\w)', '', text)     # Single digits not followed by word chars
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def is_acceptable_text(self, text: str) -> bool:
        """More lenient text quality check"""
        if len(text) < self.min_chunk_length:
            return False
        
        # Remove only the worst patterns
        worst_patterns = [
            r'^[^\w]*$',  # Only special characters
            r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$',  # Just email
            r'^https?://',  # Just URL
            r'^\s*[|]\s*$',  # Just separators
            r'^\s*[-_]\s*$',  # Just dashes/underscores
        ]
        
        for pattern in worst_patterns:
            if re.match(pattern, text):
                return False
        
        # Check for any meaningful content (very lenient)
        meaningful_indicators = [
            'कार्यालय', 'विभाग', 'कर्मचारी', 'अधिकारी', 'नियुक्ती', 'पद', 'कार्य',
            'office', 'department', 'employee', 'officer', 'appointment', 'position', 'work',
            'महापालिका', 'नगरपालिका', 'कॉर्पोरेशन', 'municipal', 'corporation',
            'कायदा', 'नियम', 'परिपत्रक', 'सूचना', 'law', 'rule', 'circular', 'notice',
            'अर्ज', 'प्रमाणपत्र', 'परवाना', 'application', 'certificate', 'license',
            'पत्ता', 'संपर्क', 'फोन', 'ईमेल', 'address', 'contact', 'phone', 'email',
            'पुणे', 'pune', 'महाराष्ट्र', 'maharashtra', 'भारत', 'india'
        ]
        
        text_lower = text.lower()
        meaningful_count = sum(1 for word in meaningful_indicators if word in text_lower)
        
        # Must have at least 1 meaningful word (very lenient)
        return meaningful_count >= 1
    
    def create_simple_chunks(self, text: str, filename: str) -> List[Dict]:
        """Create simple chunks with basic text processing"""
        chunks = []
        
        # Clean the text first
        cleaned_text = self.clean_text_basic(text)
        
        if not cleaned_text or len(cleaned_text) < self.min_chunk_length:
            return chunks
        
        # Simple character-based chunking
        chunk_index = 0
        start = 0
        
        while start < len(cleaned_text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(cleaned_text):
                # Look for sentence endings
                for i in range(end, max(start + self.chunk_size - 200, start), -1):
                    if cleaned_text[i] in '.!?।':
                        end = i + 1
                        break
            
            chunk_text = cleaned_text[start:end].strip()
            
            if chunk_text and self.is_acceptable_text(chunk_text):
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'filename': filename,
                    'type': 'improved_chunk'
                })
                chunk_index += 1
            
            start = end
            
            # Prevent infinite loop
            if start >= len(cleaned_text):
                break
        
        return chunks
    
    def process_json_file(self, json_path: Path) -> List[Dict]:
        """Process a single JSON file and create improved chunks"""
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
                            page_chunks = self.create_simple_chunks(page_text, filename)
                            for chunk in page_chunks:
                                chunk['page_number'] = page_num + 1
                            chunks.extend(page_chunks)
                
                elif 'text' in data:
                    # Single text structure
                    text = data['text']
                    chunks = self.create_simple_chunks(text, filename)
                
                elif 'content' in data:
                    # Content structure
                    content = data['content']
                    if isinstance(content, str):
                        chunks = self.create_simple_chunks(content, filename)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                item_chunks = self.create_simple_chunks(item['text'], filename)
                                chunks.extend(item_chunks)
            
            elif isinstance(data, list):
                # List structure
                for item in data:
                    if isinstance(item, dict) and 'text' in item:
                        item_chunks = self.create_simple_chunks(item['text'], filename)
                        chunks.extend(item_chunks)
            
            logger.info(f"📄 {filename}: Created {len(chunks)} improved chunks")
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
        """Process all JSON files and create improved embeddings"""
        json_dir = Path("extraceted PDF data")
        json_files = list(json_dir.glob("*.json"))
        
        logger.info(f"🚀 Starting improved embedding creation for {len(json_files)} files")
        
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
        creator = ImprovedEmbeddingCreator()
        creator.process_all_files()
        
        print(f"\n✅ Improved embeddings created successfully!")
        print(f"📁 Database location: {creator.storage_path}")
        print(f"📊 Collection name: {creator.collection_name}")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()
