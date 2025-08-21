#!/usr/bin/env python3
"""
Improve OCR Quality - Comprehensive solution for better text extraction
Uses multiple techniques to enhance OCR results before embedding
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
import easyocr
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCRQualityImprover:
    def __init__(self):
        """Initialize the OCR quality improver"""
        self.output_dir = "improved_ocr_data"
        self.temp_dir = "temp_images"
        
        # Create directories
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.temp_dir).mkdir(exist_ok=True)
        
        # Initialize OCR engines
        self._init_ocr_engines()
        
    def _init_ocr_engines(self):
        """Initialize multiple OCR engines for better results"""
        try:
            # Tesseract with multiple language support
            self.tesseract_langs = "mar+eng+hin+dev"
            
            # EasyOCR for better accuracy
            self.easyocr_reader = easyocr.Reader(['mr', 'en', 'hi'], gpu=False)
            
            logger.info("✅ OCR engines initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing OCR engines: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> List[Image.Image]:
        """Apply multiple preprocessing techniques to improve OCR"""
        processed_images = []
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # 1. Basic preprocessing
        processed_images.append(self._basic_preprocessing(img_array))
        
        # 2. Contrast enhancement
        processed_images.append(self._enhance_contrast(img_array))
        
        # 3. Noise reduction
        processed_images.append(self._reduce_noise(img_array))
        
        # 4. Sharpening
        processed_images.append(self._sharpen_image(img_array))
        
        # 5. Binarization with different thresholds
        processed_images.extend(self._binarize_image(img_array))
        
        # 6. Deskewing
        processed_images.append(self._deskew_image(img_array))
        
        return processed_images
    
    def _basic_preprocessing(self, img_array: np.ndarray) -> Image.Image:
        """Basic image preprocessing"""
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Resize for better OCR
        height, width = gray.shape
        if width > 2000:
            scale = 2000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return Image.fromarray(gray)
    
    def _enhance_contrast(self, img_array: np.ndarray) -> Image.Image:
        """Enhance image contrast"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return Image.fromarray(enhanced)
    
    def _reduce_noise(self, img_array: np.ndarray) -> Image.Image:
        """Reduce image noise"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return Image.fromarray(denoised)
    
    def _sharpen_image(self, img_array: np.ndarray) -> Image.Image:
        """Sharpen image for better text recognition"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply unsharp mask
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        return Image.fromarray(sharpened)
    
    def _binarize_image(self, img_array: np.ndarray) -> List[Image.Image]:
        """Apply different binarization techniques"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        binarized_images = []
        
        # 1. Otsu's thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binarized_images.append(Image.fromarray(otsu))
        
        # 2. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        binarized_images.append(Image.fromarray(adaptive))
        
        # 3. Simple thresholding with different values
        _, simple1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binarized_images.append(Image.fromarray(simple1))
        
        _, simple2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        binarized_images.append(Image.fromarray(simple2))
        
        return binarized_images
    
    def _deskew_image(self, img_array: np.ndarray) -> Image.Image:
        """Deskew image to correct rotation"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Find text lines and calculate skew angle
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        # Rotate image
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return Image.fromarray(rotated)
    
    def extract_text_with_multiple_engines(self, image: Image.Image) -> str:
        """Extract text using multiple OCR engines and combine results"""
        texts = []
        
        # 1. Tesseract with different configurations
        try:
            # Basic Tesseract
            text1 = pytesseract.image_to_string(image, lang=self.tesseract_langs, config='--psm 6')
            texts.append(text1)
            
            # Tesseract with different PSM modes
            text2 = pytesseract.image_to_string(image, lang=self.tesseract_langs, config='--psm 3')
            texts.append(text2)
            
            text3 = pytesseract.image_to_string(image, lang=self.tesseract_langs, config='--psm 4')
            texts.append(text3)
            
        except Exception as e:
            logger.warning(f"Tesseract error: {e}")
        
        # 2. EasyOCR
        try:
            results = self.easyocr_reader.readtext(np.array(image))
            easyocr_text = ' '.join([result[1] for result in results])
            texts.append(easyocr_text)
        except Exception as e:
            logger.warning(f"EasyOCR error: {e}")
        
        # 3. Tesseract with different preprocessing
        try:
            # Convert to PIL and apply filters
            pil_image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(2.0)
            text_enhanced = pytesseract.image_to_string(enhanced, lang=self.tesseract_langs)
            texts.append(text_enhanced)
            
            # Apply sharpening
            sharpened = pil_image.filter(ImageFilter.SHARPEN)
            text_sharpened = pytesseract.image_to_string(sharpened, lang=self.tesseract_langs)
            texts.append(text_sharpened)
            
        except Exception as e:
            logger.warning(f"Enhanced Tesseract error: {e}")
        
        return self._combine_texts(texts)
    
    def _combine_texts(self, texts: List[str]) -> str:
        """Combine multiple OCR results intelligently"""
        if not texts:
            return ""
        
        # Remove empty texts
        texts = [text.strip() for text in texts if text.strip()]
        
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # Score each text based on quality indicators
        scored_texts = []
        for text in texts:
            score = self._score_text_quality(text)
            scored_texts.append((score, text))
        
        # Sort by score (highest first)
        scored_texts.sort(reverse=True)
        
        # Use the best text as base
        best_text = scored_texts[0][1]
        
        # Try to improve by combining with other texts
        improved_text = self._merge_texts([text for _, text in scored_texts[:3]])
        
        return improved_text if improved_text else best_text
    
    def _score_text_quality(self, text: str) -> float:
        """Score text quality based on various indicators"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length score (prefer longer texts)
        score += min(len(text) / 100, 1.0) * 0.2
        
        # Word count score
        words = text.split()
        score += min(len(words) / 20, 1.0) * 0.2
        
        # Character variety score (avoid repetitive characters)
        unique_chars = len(set(text))
        score += min(unique_chars / 50, 1.0) * 0.2
        
        # Meaningful content score
        meaningful_words = ['कार्यालय', 'विभाग', 'कर्मचारी', 'अधिकारी', 'office', 'department', 'employee', 'officer']
        meaningful_count = sum(1 for word in meaningful_words if word.lower() in text.lower())
        score += min(meaningful_count / 5, 1.0) * 0.3
        
        # Punctuation score (good texts have proper punctuation)
        punct_count = len(re.findall(r'[.!?।,;:]', text))
        score += min(punct_count / 10, 1.0) * 0.1
        
        return score
    
    def _merge_texts(self, texts: List[str]) -> str:
        """Merge multiple texts intelligently"""
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # Find the longest text as base
        base_text = max(texts, key=len)
        
        # Try to fill gaps from other texts
        merged_text = base_text
        
        for other_text in texts:
            if other_text == base_text:
                continue
            
            # Find missing words from other text
            base_words = set(base_text.lower().split())
            other_words = set(other_text.lower().split())
            
            missing_words = other_words - base_words
            
            if missing_words:
                # Add missing words at appropriate positions
                # This is a simplified approach - in practice, you'd want more sophisticated merging
                merged_text += " " + " ".join(missing_words)
        
        return merged_text
    
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
        text = re.sub(r'(\w)\1{3,}', r'\1\1', text)  # Remove excessive character repetition
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def process_pdf_with_improved_ocr(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF with improved OCR"""
        try:
            filename = pdf_path.stem
            logger.info(f"📄 Processing {filename} with improved OCR...")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)  # Higher DPI for better quality
            
            all_pages = []
            total_text = ""
            
            for page_num, image in enumerate(images, 1):
                logger.info(f"  📄 Processing page {page_num}/{len(images)}")
                
                # Apply multiple preprocessing techniques
                processed_images = self.preprocess_image(image)
                
                # Extract text from each processed image
                page_texts = []
                for i, processed_image in enumerate(processed_images):
                    try:
                        text = self.extract_text_with_multiple_engines(processed_image)
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
        improver = OCRQualityImprover()
        improver.process_all_pdfs()
        
        print(f"\n✅ Improved OCR processing completed!")
        print(f"📁 Output directory: {improver.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()
