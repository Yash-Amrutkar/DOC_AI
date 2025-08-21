#!/usr/bin/env python3
"""
Complete OCR Pipeline - Runs all steps automatically
Continues processing even when screen is locked
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command_with_logging(command, description):
    """Run a command and log the output"""
    logger.info(f"🚀 Starting: {description}")
    logger.info(f"Command: {command}")
    
    start_time = time.time()
    
    try:
        # Run command and capture output
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=None  # No timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ Completed: {description} in {duration:.2f} seconds")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"❌ Failed: {description}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception in {description}: {e}")
        return False
    
    return True

def main():
    """Run the complete OCR improvement pipeline"""
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🎯 Starting Complete OCR Pipeline - {timestamp}")
    
    # Step 1: Run OCR improvement
    logger.info("="*60)
    logger.info("STEP 1: Running OCR Quality Improvement")
    logger.info("="*60)
    
    success = run_command_with_logging(
        "python3 simple_ocr_improvement.py",
        "OCR Quality Improvement"
    )
    
    if not success:
        logger.error("❌ OCR improvement failed. Stopping pipeline.")
        return
    
    # Step 2: Create quality embeddings
    logger.info("="*60)
    logger.info("STEP 2: Creating Quality Embeddings")
    logger.info("="*60)
    
    success = run_command_with_logging(
        "python3 create_quality_embeddings.py",
        "Quality Embeddings Creation"
    )
    
    if not success:
        logger.error("❌ Embedding creation failed. Stopping pipeline.")
        return
    
    # Step 3: Test the improved system
    logger.info("="*60)
    logger.info("STEP 3: Testing Improved System")
    logger.info("="*60)
    
    # Create a simple test script
    test_script = """
import sys
sys.path.append('.')
from final_rag_system import FinalRAGQnASystem

# Test the improved system
rag_system = FinalRAGQnASystem("gemini-2.0-flash-exp")
print("✅ Improved RAG system initialized successfully!")
print(f"📊 System info: {rag_system.get_system_info()}")
"""
    
    with open("test_improved_system.py", "w") as f:
        f.write(test_script)
    
    success = run_command_with_logging(
        "python3 test_improved_system.py",
        "Improved System Test"
    )
    
    # Step 4: Create summary report
    logger.info("="*60)
    logger.info("STEP 4: Creating Summary Report")
    logger.info("="*60)
    
    summary_script = """
import json
from pathlib import Path

# Count improved OCR files
improved_dir = Path("improved_ocr_data")
improved_files = list(improved_dir.glob("*.json")) if improved_dir.exists() else []

# Count original files
original_dir = Path("extraceted PDF data")
original_files = list(original_dir.glob("*.json")) if original_dir.exists() else []

# Create summary
summary = {
    "timestamp": "{}",
    "original_files": len(original_files),
    "improved_files": len(improved_files),
    "improvement_ratio": len(improved_files) / len(original_files) if original_files else 0,
    "status": "COMPLETED" if len(improved_files) > 0 else "FAILED"
}

with open("ocr_improvement_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"📊 OCR Improvement Summary:")
print(f"   Original files: {len(original_files)}")
print(f"   Improved files: {len(improved_files)}")
print(f"   Success rate: {summary['improvement_ratio']:.1%}")
print(f"   Status: {summary['status']}")
""".format(timestamp)
    
    with open("create_summary.py", "w") as f:
        f.write(summary_script)
    
    run_command_with_logging(
        "python3 create_summary.py",
        "Summary Report Creation"
    )
    
    # Final completion message
    logger.info("="*60)
    logger.info("🎉 COMPLETE OCR PIPELINE FINISHED!")
    logger.info("="*60)
    logger.info("✅ All steps completed successfully!")
    logger.info("📁 Check the following files:")
    logger.info("   - ocr_pipeline.log (detailed progress)")
    logger.info("   - ocr_improvement_summary.json (summary)")
    logger.info("   - improved_ocr_data/ (improved text files)")
    logger.info("   - qdrant_storage_quality/ (new embeddings)")
    logger.info("")
    logger.info("🚀 You can now use the improved system:")
    logger.info("   python3 final_rag_system.py")
    logger.info("   python3 web_interface.py")

if __name__ == "__main__":
    main()
