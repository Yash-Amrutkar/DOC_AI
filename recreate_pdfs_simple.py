#!/usr/bin/env python3
"""
Simple PDF recreation from JSON data
"""

import os
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_json(json_file_path):
    """Extract text from JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text_content = []
        
        # Add filename as title
        filename = Path(json_file_path).stem
        text_content.append(f"DOCUMENT: {filename}")
        text_content.append("=" * 50)
        text_content.append("")
        
        # Process pages
        if 'pages' in data:
            pages = data['pages']
            if isinstance(pages, dict):
                for page_key, page_data in pages.items():
                    if isinstance(page_data, dict):
                        # Original text
                        if 'original_text' in page_data:
                            text = page_data['original_text']
                            if text and text.strip():
                                text_content.append(f"PAGE: {page_key}")
                                text_content.append("-" * 30)
                                text_content.append(text.strip())
                                text_content.append("")
                        
                        # English translation
                        if 'translations' in page_data and 'english' in page_data['translations']:
                            eng_text = page_data['translations']['english']
                            if eng_text and eng_text.strip():
                                text_content.append(f"ENGLISH TRANSLATION - {page_key}")
                                text_content.append("-" * 30)
                                text_content.append(eng_text.strip())
                                text_content.append("")
        
        return text_content
        
    except Exception as e:
        logger.error(f"Error processing {json_file_path}: {e}")
        return []

def create_simple_pdf(filename, text_content, output_dir):
    """Create simple PDF from text content"""
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    
    try:
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        
        y_position = height - 50
        line_height = 15
        
        for line in text_content:
            if y_position < 50:  # New page if near bottom
                c.showPage()
                y_position = height - 50
            
            # Handle long lines
            if len(line) > 80:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) < 80:
                        current_line += " " + word if current_line else word
                    else:
                        c.drawString(50, y_position, current_line)
                        y_position -= line_height
                        current_line = word
                if current_line:
                    c.drawString(50, y_position, current_line)
                    y_position -= line_height
            else:
                c.drawString(50, y_position, line)
                y_position -= line_height
        
        c.save()
        logger.info(f"✅ Created: {pdf_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating {pdf_path}: {e}")
        return False

def main():
    """Main function"""
    input_dir = "extraceted PDF data"
    output_dir = "recreated_pdfs"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    json_files = list(Path(input_dir).glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files")
    
    success_count = 0
    
    for i, json_file in enumerate(json_files, 1):
        logger.info(f"Processing {i}/{len(json_files)}: {json_file.name}")
        
        # Extract text
        text_content = extract_text_from_json(json_file)
        
        if text_content:
            # Create PDF
            filename = json_file.stem
            if create_simple_pdf(filename, text_content, output_dir):
                success_count += 1
    
    logger.info(f"✅ Completed! Created {success_count} PDF files in '{output_dir}' directory")

if __name__ == "__main__":
    main()
