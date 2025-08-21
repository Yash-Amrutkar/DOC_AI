#!/usr/bin/env python3
"""
Comprehensive OCR Improvement and Embedding Creation
This script will improve OCR quality and create new embeddings for better Q&A performance
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
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRImprover:
    def __init__(self):
        self.input_dir = "extraceted PDF data"
        self.output_dir = "improved_ocr_data_v2"
        self.embeddings_dir = "qdrant_storage_v2"
        self.collection_name = "pdf_embeddings_v2"
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def preprocess_image_for_ocr(self, image: Image.Image) -> List[Image.Image]:
        """Apply multiple preprocessing techniques to improve OCR"""
        processed_images = []
        
        # Original image
        processed_images.append(image)
        
        # 1. Contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        contrast_img = enhancer.enhance(2.0)
        processed_images.append(contrast_img)
        
        # 2. Sharpening
        sharpened = image.filter(ImageFilter.SHARPEN)
        processed_images.append(sharpened)
        
        # 3. Grayscale with high contrast
        gray = image.convert('L')
        # Apply histogram equalization
        gray_array = np.array(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_array)
        enhanced_img = Image.fromarray(enhanced)
        processed_images.append(enhanced_img)
        
        # 4. Binarization
        gray_array = np.array(gray)
        _, binary = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_img = Image.fromarray(binary)
        processed_images.append(binary_img)
        
        # 5. Noise reduction
        denoised = cv2.fastNlMeansDenoising(np.array(gray))
        denoised_img = Image.fromarray(denoised)
        processed_images.append(denoised_img)
        
        return processed_images
    
    def extract_text_with_multiple_configs(self, image: Image.Image) -> str:
        """Extract text using multiple OCR configurations"""
        texts = []
        
        # Different Tesseract configurations
        configs = [
            '--oem 3 --psm 6',  # Assume uniform block of text
            '--oem 3 --psm 3',  # Fully automatic page segmentation
            '--oem 3 --psm 4',  # Assume single column of text
            '--oem 3 --psm 8',  # Single word
            '--oem 3 --psm 13', # Raw line
        ]
        
        # Try different languages
        languages = ['eng', 'mar+eng', 'hin+eng', 'mar+hin+eng']
        
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
        
        # Score texts by length and character quality
        scored_texts = []
        for text in unique_texts:
            score = 0
            # Prefer longer texts
            score += len(text) * 0.1
            # Prefer texts with more alphabetic characters
            alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
            score += alpha_ratio * 100
            # Penalize texts with too many special characters
            special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
            score -= special_ratio * 50
            
            scored_texts.append((score, text))
        
        # Sort by score and take the best
        scored_texts.sort(reverse=True)
        best_text = scored_texts[0][1]
        
        # If we have multiple good texts, try to merge them
        if len(scored_texts) > 1 and scored_texts[1][0] > scored_texts[0][0] * 0.8:
            # Merge texts that are similar
            merged = best_text
            for score, text in scored_texts[1:]:
                if score > scored_texts[0][0] * 0.7:
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
        
        # Remove lines that are too short (likely noise)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3:  # Keep lines with more than 3 characters
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
            except:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
            
            # Process chunks in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
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
                
                logger.info(f"✅ Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            logger.info(f"✅ Created embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings: {e}")
            raise
    
    def run_complete_pipeline(self):
        """Run the complete OCR improvement and embedding pipeline"""
        logger.info("🚀 Starting OCR Improvement and Embedding Pipeline")
        
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
        improver = OCRImprover()
        improver.run_complete_pipeline()
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()
