#!/usr/bin/env python3
"""
Simple OCR Quality Improvement - Focus on most effective techniques
Improves OCR quality using key preprocessing and multiple engines
"""

import os
import json
import re
import logging
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleOCRImprover:
    def __init__(self):
        """Initialize the simple OCR improver"""
        self.output_dir = "improved_ocr_data"
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Initialize OCR configuration
        self._init_ocr_config()
        
    def _init_ocr_config(self):
        """Initialize OCR configuration"""
        try:
            # Tesseract with multiple language support
            self.tesseract_langs = "mar+eng+hin"
            
            # Test Tesseract
            version = pytesseract.get_tesseract_version()
            logger.info(f"✅ Tesseract version: {version}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing OCR: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> List[Image.Image]:
        """Apply key preprocessing techniques"""
        processed_images = []
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # 1. Basic preprocessing (grayscale + resize)
        basic = self._basic_preprocessing(img_array)
        processed_images.append(basic)
        
        # 2. Contrast enhancement
        enhanced = self._enhance_contrast(img_array)
        processed_images.append(enhanced)
        
        # 3. Noise reduction
        denoised = self._reduce_noise(img_array)
        processed_images.append(denoised)
        
        # 4. Binarization (Otsu)
        binary = self._binarize_image(img_array)
        processed_images.append(binary)
        
        return processed_images
    
    def _basic_preprocessing(self, img_array: np.ndarray) -> Image.Image:
        """Basic image preprocessing"""
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Resize for better OCR (max width 2000px)
        height, width = gray.shape
        if width > 2000:
            scale = 2000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return Image.fromarray(gray)
    
    def _enhance_contrast(self, img_array: np.ndarray) -> Image.Image:
        """Enhance image contrast using CLAHE"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return Image.fromarray(enhanced)
    
    def _reduce_noise(self, img_array: np.ndarray) -> Image.Image:
        """Reduce image noise using bilateral filter"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply bilateral filter
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return Image.fromarray(denoised)
    
    def _binarize_image(self, img_array: np.ndarray) -> Image.Image:
        """Apply Otsu's binarization"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return Image.fromarray(binary)
    
    def extract_text_with_multiple_configs(self, image: Image.Image) -> str:
        """Extract text using multiple Tesseract configurations"""
        texts = []
        
        # Different PSM modes
        psm_modes = [3, 4, 6, 8]
        
        for psm in psm_modes:
            try:
                config = f'--psm {psm} --oem 3'
                text = pytesseract.image_to_string(image, lang=self.tesseract_langs, config=config)
                if text.strip():
                    texts.append(text.strip())
            except Exception as e:
                logger.warning(f"PSM {psm} error: {e}")
        
        # Enhanced preprocessing with PIL
        try:
            pil_image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(2.0)
            text_enhanced = pytesseract.image_to_string(enhanced, lang=self.tesseract_langs)
            if text_enhanced.strip():
                texts.append(text_enhanced.strip())
            
            # Apply sharpening
            sharpened = pil_image.filter(ImageFilter.SHARPEN)
            text_sharpened = pytesseract.image_to_string(sharpened, lang=self.tesseract_langs)
            if text_sharpened.strip():
                texts.append(text_sharpened.strip())
                
        except Exception as e:
            logger.warning(f"Enhanced processing error: {e}")
        
        return self._combine_texts(texts)
    
    def _combine_texts(self, texts: List[str]) -> str:
        """Combine multiple OCR results intelligently"""
        if not texts:
            return ""
        
        # Remove duplicates and empty texts
        unique_texts = list(set([text.strip() for text in texts if text.strip()]))
        
        if not unique_texts:
            return ""
        
        if len(unique_texts) == 1:
            return unique_texts[0]
        
        # Score each text based on quality
        scored_texts = []
        for text in unique_texts:
            score = self._score_text_quality(text)
            scored_texts.append((score, text))
        
        # Sort by score (highest first)
        scored_texts.sort(reverse=True)
        
        # Use the best text
        return scored_texts[0][1]
    
    def _score_text_quality(self, text: str) -> float:
        """Score text quality based on various indicators"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length score (prefer longer texts)
        score += min(len(text) / 100, 1.0) * 0.3
        
        # Word count score
        words = text.split()
        score += min(len(words) / 20, 1.0) * 0.3
        
        # Character variety score
        unique_chars = len(set(text))
        score += min(unique_chars / 50, 1.0) * 0.2
        
        # Meaningful content score
        meaningful_words = [
            'कार्यालय', 'विभाग', 'कर्मचारी', 'अधिकारी', 'नियुक्ती', 'पद', 'कार्य',
            'office', 'department', 'employee', 'officer', 'appointment', 'position', 'work',
            'महापालिका', 'नगरपालिका', 'कॉर्पोरेशन', 'municipal', 'corporation',
            'कायदा', 'नियम', 'परिपत्रक', 'सूचना', 'law', 'rule', 'circular', 'notice',
            'पुणे', 'pune', 'महाराष्ट्र', 'maharashtra'
        ]
        
        meaningful_count = sum(1 for word in meaningful_words if word.lower() in text.lower())
        score += min(meaningful_count / 5, 1.0) * 0.2
        
        return score
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common OCR artifacts
        text = re.sub(r'[|]{2,}', ' ', text)  # Multiple pipes
        text = re.sub(r'[-]{2,}', '-', text)  # Multiple dashes
        text = re.sub(r'[=]{2,}', '=', text)  # Multiple equals
        
        # Remove isolated characters (likely OCR errors)
        text = re.sub(r'\b[a-zA-Z]\b(?!\w)', '', text)  # Single letters
        text = re.sub(r'\b[0-9]\b(?!\w)', '', text)     # Single digits
        
        # Fix common OCR errors
        text = re.sub(r'(\w)\1{3,}', r'\1\1', text)  # Remove excessive repetition
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def process_pdf_with_improved_ocr(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF with improved OCR"""
        try:
            filename = pdf_path.stem
            logger.info(f"📄 Processing {filename} with improved OCR...")
            
            # Convert PDF to images with higher DPI
            images = convert_from_path(pdf_path, dpi=300)
            
            all_pages = []
            total_text = ""
            
            for page_num, image in enumerate(images, 1):
                logger.info(f"  📄 Processing page {page_num}/{len(images)}")
                
                # Apply preprocessing techniques
                processed_images = self.preprocess_image(image)
                
                # Extract text from each processed image
                page_texts = []
                for i, processed_image in enumerate(processed_images):
                    try:
                        text = self.extract_text_with_multiple_configs(processed_image)
                        if text.strip():
                            page_texts.append(text)
                    except Exception as e:
                        logger.warning(f"    Error processing variant {i+1}: {e}")
                
                # Combine and clean page text
                if page_texts:
                    page_text = self._combine_texts(page_texts)
                    page_text = self.clean_extracted_text(page_text)
                    
                    if page_text:
                        all_pages.append({
                            'page_number': page_num,
                            'text': page_text,
                            'text_length': len(page_text)
                        })
                        total_text += page_text + "\n"
                
                # Clean up processed images
                for processed_image in processed_images:
                    processed_image.close()
            
            # Create result
            result = {
                'filename': filename,
                'total_pages': len(images),
                'processed_pages': len(all_pages),
                'total_text_length': len(total_text),
                'pages': all_pages,
                'text': total_text.strip()
            }
            
            logger.info(f"✅ {filename}: Extracted {len(total_text)} characters from {len(all_pages)} pages")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path}: {e}")
            return None
    
    def process_all_pdfs(self):
        """Process all PDFs with improved OCR"""
        pdf_dir = Path("downloads/documents")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        logger.info(f"🚀 Starting improved OCR processing for {len(pdf_files)} PDFs")
        
        successful = 0
        failed = 0
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            result = self.process_pdf_with_improved_ocr(pdf_file)
            
            if result:
                # Save improved result
                output_file = Path(self.output_dir) / f"{result['filename']}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                successful += 1
            else:
                failed += 1
        
        logger.info(f"🎉 Completed! {successful} successful, {failed} failed")
        logger.info(f"📁 Improved OCR data saved to: {self.output_dir}")

def main():
    """Main function"""
    try:
        improver = SimpleOCRImprover()
        improver.process_all_pdfs()
        
        print(f"\n✅ Improved OCR processing completed!")
        print(f"📁 Output directory: {improver.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()
