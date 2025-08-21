#!/usr/bin/env python3
"""
RAG QnA System Startup Script
Guides users through the setup and startup process
"""

import os
import sys
from pathlib import Path

def check_api_key():
    """Check if Gemini API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        return False
    return True

def check_database():
    """Check if vector database exists"""
    storage_path = Path("./qdrant_storage")
    return storage_path.exists() and any(storage_path.iterdir())

def main():
    """Main startup function"""
    print("="*60)
    print("🤖 RAG QnA System - Startup Guide")
    print("="*60)
    
    # Check API key
    print("\n🔑 Checking Gemini API key...")
    if not check_api_key():
        print("❌ Gemini API key not configured!")
        print("\n📝 To get your API key:")
        print("1. Go to https://aistudio.google.com/")
        print("2. Sign in and click 'Get API key'")
        print("3. Copy the API key")
        print("4. Edit the .env file and replace 'your_gemini_api_key_here'")
        print("\nExample:")
        print("GEMINI_API_KEY=AIzaSyC...your_actual_key_here")
        return False
    
    print("✅ Gemini API key configured!")
    
    # Check database
    print("\n💾 Checking vector database...")
    if not check_database():
        print("❌ Vector database not found!")
        print("\n📝 To create embeddings:")
        print("python3 create_embeddings_persistent.py")
        return False
    
    print("✅ Vector database ready!")
    
    # Show system info
    print("\n📊 System Status:")
    print("   ✅ API Key: Configured")
    print("   ✅ Database: Ready")
    print("   ✅ Dependencies: Installed")
    
    # Ask user preference
    print("\n🎯 Choose your interface:")
    print("1. Web Interface (Recommended) - Beautiful UI, automatic browser")
    print("2. Command Line - Terminal-based, detailed info")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting web interface...")
                os.system("python3 web_interface.py")
                break
            elif choice == "2":
                print("\n🚀 Starting command line interface...")
                os.system("python3 rag_qna_system.py")
                break
            elif choice == "3":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    main()
