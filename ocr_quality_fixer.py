#!/usr/bin/env python3
"""
OCR Quality Fixer - Comprehensive solution for poor OCR quality
"""

import os
import json
import logging
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pathlib import Path
from typing import List, Dict, Any
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRQualityFixer:
    def __init__(self):
        self.input_dir = "extraceted PDF data"
        self.output_dir = "high_quality_ocr_data"
        self.embeddings_dir = "qdrant_storage_high_quality"
        self.collection_name = "pdf_embeddings_high_quality"
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def generate_uuid(self, text: str) -> str:
        """Generate a UUID-like string from text"""
        hash_object = hashlib.md5(text.encode())
        hash_hex = hash_object.hexdigest()
        return f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    
    def preprocess_image_for_ocr(self, image: Image.Image) -> List[Image.Image]:
        """Apply multiple preprocessing techniques to improve OCR"""
        processed_images = []
        
        # Original image
        processed_images.append(image)
        
        # 1. High contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        contrast_img = enhancer.enhance(3.0)  # Increased contrast
        processed_images.append(contrast_img)
        
        # 2. Sharpening
        sharpened = image.filter(ImageFilter.SHARPEN)
        processed_images.append(sharpened)
        
        # 3. Grayscale with CLAHE
        gray = image.convert('L')
        gray_array = np.array(gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_array)
        enhanced_img = Image.fromarray(enhanced)
        processed_images.append(enhanced_img)
        
        # 4. Adaptive binarization
        gray_array = np.array(gray)
        binary = cv2.adaptiveThreshold(gray_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        binary_img = Image.fromarray(binary)
        processed_images.append(binary_img)
        
        # 5. Noise reduction with bilateral filter
        denoised = cv2.bilateralFilter(np.array(gray), 9, 75, 75)
        denoised_img = Image.fromarray(denoised)
        processed_images.append(denoised_img)
        
        # 6. Morphological operations
        kernel = np.ones((1,1), np.uint8)
        morphed = cv2.morphologyEx(np.array(gray), cv2.MORPH_CLOSE, kernel)
        morphed_img = Image.fromarray(morphed)
        processed_images.append(morphed_img)
        
        return processed_images
    
    def extract_text_with_multiple_configs(self, image: Image.Image) -> str:
        """Extract text using multiple OCR configurations"""
        texts = []
        
        # Different Tesseract configurations for better accuracy
        configs = [
            '--oem 3 --psm 6',  # Assume uniform block of text
            '--oem 3 --psm 3',  # Fully automatic page segmentation
            '--oem 3 --psm 4',  # Assume single column of text
            '--oem 3 --psm 8',  # Single word
            '--oem 3 --psm 13', # Raw line
            '--oem 1 --psm 6',  # Legacy engine
        ]
        
        # Try different languages with better combinations
        languages = ['eng', 'mar+eng', 'hin+eng', 'mar+hin+eng', 'eng+mar+hin']
        
        for lang in languages:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(
                        image, 
                        lang=lang, 
                        config=config
                    )
                    if text.strip():
                        texts.append(text.strip())
                except Exception as e:
                    logger.debug(f"OCR config failed: {lang} {config} - {e}")
                    continue
        
        return self.combine_texts(texts)
    
    def combine_texts(self, texts: List[str]) -> str:
        """Intelligently combine multiple OCR results"""
        if not texts:
            return ""
        
        # Remove duplicates
        unique_texts = list(set(texts))
        
        # Score texts by quality
        scored_texts = []
        for text in unique_texts:
            score = 0
            # Prefer longer texts
            score += len(text) * 0.1
            # Prefer texts with more alphabetic characters
            alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
            score += alpha_ratio * 200  # Increased weight
            # Penalize texts with too many special characters
            special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
            score -= special_ratio * 100  # Increased penalty
            # Bonus for readable text
            if ' ' in text and len(text.split()) > 3:
                score += 50
            
            scored_texts.append((score, text))
        
        # Sort by score and take the best
        scored_texts.sort(reverse=True)
        best_text = scored_texts[0][1]
        
        # If we have multiple good texts, try to merge them
        if len(scored_texts) > 1 and scored_texts[1][0] > scored_texts[0][0] * 0.7:
            merged = best_text
            for score, text in scored_texts[1:]:
                if score > scored_texts[0][0] * 0.6:
                    # Add unique parts
                    words1 = set(best_text.split())
                    words2 = set(text.split())
                    unique_words = words2 - words1
                    if unique_words:
                        merged += " " + " ".join(unique_words)
            return merged
        
        return best_text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')  # Be careful with this
        text = text.replace('1', 'I')  # Be careful with this
        text = text.replace('l', 'I')  # Common confusion
        text = text.replace('rn', 'm')  # Common confusion
        
        # Remove lines that are too short (likely noise)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 5:  # Increased minimum length
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def process_json_file(self, json_path: Path) -> Dict[str, Any]:
        """Process a single JSON file and improve its OCR"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            improved_data = {
                'filename': json_path.stem,
                'pages': []
            }
            
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
                    
                    # Clean the text
                    cleaned_text = self.clean_text(combined_text)
                    
                    improved_data['pages'].append({
                        'page_number': page_key,
                        'text': cleaned_text,
                        'original_text': original_text,
                        'english_translation': english_text
                    })
            
            return improved_data
            
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            return None
    
    def create_chunks(self, text: str, filename: str, page: str, chunk_size: int = 600) -> List[Dict]:
        """Create text chunks for embedding with better quality"""
        chunks = []
        
        # Clean the text first
        text = self.clean_text(text)
        if not text or len(text.strip()) < 20:
            return chunks
        
        # Split by sentences first
        sentences = text.split('. ')
        
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip() and len(current_chunk.strip()) > 30:
                    chunks.append({
                        'id': self.generate_uuid(f"{filename}_{page}_chunk{chunk_id}"),
                        'text': current_chunk.strip(),
                        'filename': filename,
                        'page': page,
                        'chunk_id': chunk_id
                    })
                    chunk_id += 1
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 30:
            chunks.append({
                'id': self.generate_uuid(f"{filename}_{page}_chunk{chunk_id}"),
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
            batch_size = 50
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            logger.info(f"📊 Processing {len(chunks)} chunks in {total_batches} batches...")
            
            for i in tqdm(range(0, len(chunks), batch_size), desc="Creating embeddings"):
                batch = chunks[i:i + batch_size]
                
                # Create embeddings
                texts = [chunk['text'] for chunk in batch]
                embeddings = self.embedding_model.encode(texts)
                
                # Prepare points for Qdrant
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
                
                # Upsert to Qdrant
                client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                time.sleep(0.1)
            
            # Get final count
            collection_info = client.get_collection(self.collection_name)
            logger.info(f"✅ Created embeddings for {len(chunks)} chunks")
            logger.info(f"✅ Total embeddings in collection: {collection_info.points_count}")
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings: {e}")
            raise
    
    def run_complete_pipeline(self):
        """Run the complete OCR quality improvement and embedding pipeline"""
        logger.info("🚀 Starting OCR Quality Improvement Pipeline")
        
        # Find all JSON files
        json_files = list(Path(self.input_dir).glob("*.json"))
        logger.info(f"📁 Found {len(json_files)} JSON files to process")
        
        all_chunks = []
        processed_count = 0
        
        for json_file in tqdm(json_files, desc="Processing files"):
            try:
                # Process the JSON file
                improved_data = self.process_json_file(json_file)
                
                if improved_data:
                    # Save improved data
                    output_path = Path(self.output_dir) / f"{json_file.stem}_improved.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(improved_data, f, ensure_ascii=False, indent=2)
                    
                    # Create chunks
                    for page in improved_data['pages']:
                        chunks = self.create_chunks(
                            page['text'],
                            improved_data['filename'],
                            page['page_number']
                        )
                        all_chunks.extend(chunks)
                    
                    processed_count += 1
                    logger.info(f"✅ Processed {json_file.name} ({processed_count}/{len(json_files)})")
                
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
        fixer = OCRQualityFixer()
        fixer.run_complete_pipeline()
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
