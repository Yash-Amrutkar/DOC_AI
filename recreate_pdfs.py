#!/usr/bin/env python3
"""
Recreate PDF files from JSON data
"""

import os
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFRecreator:
    def __init__(self):
        self.input_dir = "extraceted PDF data"
        self.output_dir = "recreated_pdfs"
        self.styles = getSampleStyleSheet()
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Setup styles for different languages
        self.setup_styles()
    
    def setup_styles(self):
        """Setup paragraph styles for different languages"""
        # Default style
        self.default_style = ParagraphStyle(
            'Default',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT
        )
        
        # Title style
        self.title_style = ParagraphStyle(
            'Title',
            parent=self.styles['Heading1'],
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=20
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=self.styles['Heading2'],
            fontSize=12,
            leading=16,
            alignment=TA_LEFT,
            spaceAfter=10
        )
    
    def extract_text_from_json(self, json_file_path):
        """Extract text content from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            extracted_text = []
            
            # Extract filename as title
            filename = Path(json_file_path).stem
            extracted_text.append(("title", filename))
            
            # Process pages
            if 'pages' in data:
                pages = data['pages']
                if isinstance(pages, dict):
                    for page_key, page_data in pages.items():
                        if isinstance(page_data, dict):
                            # Extract original text
                            if 'original_text' in page_data:
                                text = page_data['original_text']
                                if text and text.strip():
                                    extracted_text.append(("page", f"Page {page_key}"))
                                    extracted_text.append(("content", text))
                            
                            # Extract English translation if available
                            if 'translations' in page_data and 'english' in page_data['translations']:
                                eng_text = page_data['translations']['english']
                                if eng_text and eng_text.strip():
                                    extracted_text.append(("translation", f"English Translation - {page_key}"))
                                    extracted_text.append(("content", eng_text))
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error processing {json_file_path}: {e}")
            return []
    
    def create_pdf_from_text(self, filename, text_content):
        """Create PDF file from extracted text"""
        pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
        
        try:
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            story = []
            
            for content_type, text in text_content:
                if content_type == "title":
                    story.append(Paragraph(text, self.title_style))
                    story.append(Spacer(1, 20))
                
                elif content_type == "page":
                    story.append(Paragraph(text, self.subtitle_style))
                    story.append(Spacer(1, 10))
                
                elif content_type == "translation":
                    story.append(Paragraph(text, self.subtitle_style))
                    story.append(Spacer(1, 10))
                
                elif content_type == "content":
                    # Clean and format the text
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:
                        # Split into paragraphs
                        paragraphs = cleaned_text.split('\n\n')
                        for para in paragraphs:
                            if para.strip():
                                story.append(Paragraph(para.strip(), self.default_style))
                                story.append(Spacer(1, 6))
            
            doc.build(story)
            logger.info(f"✅ Created PDF: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating PDF {pdf_path}: {e}")
            return False
    
    def clean_text(self, text):
        """Clean and format text for PDF"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Replace common OCR artifacts
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')  # Be careful with this one
        text = text.replace('1', 'l')  # Be careful with this one
        
        return text
    
    def process_all_files(self):
        """Process all JSON files and create PDFs"""
        json_files = list(Path(self.input_dir).glob("*.json"))
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        success_count = 0
        total_count = len(json_files)
        
        for i, json_file in enumerate(json_files, 1):
            logger.info(f"Processing {i}/{total_count}: {json_file.name}")
            
            # Extract text from JSON
            text_content = self.extract_text_from_json(json_file)
            
            if text_content:
                # Create PDF
                filename = json_file.stem
                if self.create_pdf_from_text(filename, text_content):
                    success_count += 1
            
            # Progress update
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total_count} files processed")
        
        logger.info(f"✅ PDF recreation completed!")
        logger.info(f"📊 Successfully created: {success_count}/{total_count} PDF files")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        return success_count, total_count

def main():
    """Main function"""
    logger.info("🚀 Starting PDF recreation process...")
    
    recreator = PDFRecreator()
    success_count, total_count = recreator.process_all_files()
    
    if success_count > 0:
        logger.info(f"🎉 Successfully recreated {success_count} PDF files!")
        logger.info(f"📂 Check the '{recreator.output_dir}' directory for your PDF files")
    else:
        logger.error("❌ No PDF files were created successfully")

if __name__ == "__main__":
    main()
