#!/usr/bin/env python3
"""
Check OCR Progress - Monitor the OCR improvement process
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

def check_progress():
    """Check the current progress of OCR improvement"""
    
    # Count files
    pdf_dir = Path("downloads/documents")
    improved_dir = Path("improved_ocr_data")
    
    total_pdfs = len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else 0
    improved_files = len(list(improved_dir.glob("*.json"))) if improved_dir.exists() else 0
    
    # Calculate progress
    progress = (improved_files / total_pdfs * 100) if total_pdfs > 0 else 0
    
    # Estimate time remaining (based on 2 minutes per PDF)
    processed_time = improved_files * 2  # minutes
    remaining_files = total_pdfs - improved_files
    remaining_time = remaining_files * 2  # minutes
    
    # Convert to hours and minutes
    remaining_hours = remaining_time // 60
    remaining_minutes = remaining_time % 60
    
    # Check if process is still running
    import subprocess
    result = subprocess.run("ps aux | grep 'simple_ocr_improvement.py' | grep -v grep", shell=True, capture_output=True, text=True)
    is_running = result.returncode == 0
    
    # Print status
    print("="*60)
    print("📊 OCR Improvement Progress")
    print("="*60)
    print(f"🕐 Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Total PDFs: {total_pdfs}")
    print(f"✅ Processed: {improved_files}")
    print(f"📈 Progress: {progress:.1f}%")
    print(f"⏱️  Estimated time remaining: {remaining_hours}h {remaining_minutes}m")
    print(f"🔄 Status: {'🟢 RUNNING' if is_running else '🔴 STOPPED'}")
    
    if improved_files > 0:
        print(f"\n📂 Latest processed files:")
        json_files = list(improved_dir.glob("*.json"))
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        for i, file in enumerate(json_files[:5]):
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            print(f"   {i+1}. {file.stem} ({mtime.strftime('%H:%M:%S')})")
    
    print("\n📋 To check detailed logs:")
    print("   tail -f ocr_pipeline.log")
    print("   tail -f ocr_pipeline_output.log")
    
    print("\n🎯 When complete, you can use:")
    print("   python3 final_rag_system.py")
    print("   python3 web_interface.py")

if __name__ == "__main__":
    check_progress()
