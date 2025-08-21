# PDF Embedding Setup with Qdrant Local

## 🎯 Overview
This guide will help you set up a local Qdrant vector database and create embeddings for your extracted PDF documents for semantic search.

## 📋 Prerequisites
- Python 3.8+
- Your extracted JSON files (202 files in `extraceted PDF data/`)
- Internet connection (for downloading models)

## 🚀 Quick Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements_embeddings.txt
```

### Step 2: Set Up Qdrant Local
```bash
python setup_qdrant.py
```

### Step 3: Create Embeddings
```bash
python create_embeddings.py
```

### Step 4: Test Search
```bash
python search_documents.py
```

## 📁 Project Structure
```
DOC_AI/
├── .env                          # Environment configuration
├── create_embeddings.py          # Main embedding script
├── search_documents.py           # Search interface
├── setup_qdrant.py              # Qdrant setup script
├── requirements_embeddings.txt   # Python dependencies
├── extraceted PDF data/         # Your JSON files (202 files)
└── EMBEDDING_SETUP.md           # This guide
```

## ⚙️ Configuration (.env file)

### Qdrant Settings
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=pdf_embeddings
```

### Embedding Model
```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
BATCH_SIZE=32
```

### Processing Settings
```env
CHUNK_SIZE=1000
OVERLAP_SIZE=200
```

## 🔧 Detailed Setup Steps

### 1. Environment Setup
The `.env` file is already configured for local Qdrant usage. Key settings:
- **Qdrant Host**: localhost
- **Qdrant Port**: 6333
- **Embedding Model**: all-MiniLM-L6-v2 (fast and efficient)
- **Chunk Size**: 1000 characters per chunk
- **Overlap**: 200 characters between chunks

### 2. Start Qdrant Server
```bash
python setup_qdrant.py
```
This script will:
- Check if Qdrant is installed
- Install Qdrant if needed
- Start the Qdrant server locally
- Verify the connection

### 3. Create Embeddings
```bash
python create_embeddings.py
```
This process will:
- Load all 202 JSON files
- Extract text from each file
- Create text chunks with overlap
- Generate embeddings using sentence-transformers
- Store embeddings in Qdrant with metadata

**Expected Output:**
```
2024-08-19 10:30:00 - INFO - Found 202 JSON files to process
2024-08-19 10:30:01 - INFO - Processing file 1/202: filename.json
...
2024-08-19 10:45:00 - INFO - EMBEDDING PROCESS COMPLETED
2024-08-19 10:45:00 - INFO - Total files found: 202
2024-08-19 10:45:00 - INFO - Successfully processed: 202
2024-08-19 10:45:00 - INFO - Total embeddings created: 1500+
```

### 4. Search Your Documents
```bash
python search_documents.py
```

**Interactive Search:**
```
🔍 Enter your search query: environmental report
📊 Number of results (default 10): 5

🔍 Searching for: 'environmental report'
📊 Returning top 5 results...
------------------------------------------------------------

1. 📄 Environmental Status Report for 2023-2024_20250818_144606
   📊 Similarity Score: 0.892
   📝 Type: full_document
   📄 Pages: 15
   💬 Preview: पर्यावरण सद्य:स्थिती अहवाल २०२३-२४...
```

## 🔍 Search Examples

### Government Documents
- "budget allocation"
- "tax collection"
- "municipal commissioner"
- "public notice"

### Departments
- "water supply department"
- "fire brigade"
- "tree authority"
- "environmental department"

### Specific Topics
- "RTI information"
- "ward office"
- "tree plantation"
- "stack emission monitoring"

## 📊 Performance Information

### Embedding Model
- **Model**: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Speed**: ~2000 sentences/second
- **Quality**: Good for semantic search

### Processing Time
- **202 files**: ~15-30 minutes
- **Embeddings per file**: 5-10 (depending on text length)
- **Total embeddings**: ~1500-2000

### Storage
- **Qdrant database**: ~100-200 MB
- **Embedding vectors**: ~384 dimensions each

## 🛠️ Troubleshooting

### Qdrant Connection Issues
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# Restart Qdrant
pkill qdrant
python setup_qdrant.py
```

### Memory Issues
If you encounter memory issues:
1. Reduce `BATCH_SIZE` in `.env` (default: 32)
2. Reduce `CHUNK_SIZE` in `.env` (default: 1000)
3. Process files in smaller batches

### Model Download Issues
If the embedding model fails to download:
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python create_embeddings.py
```

## 🎯 Advanced Usage

### Demo Searches
```bash
python search_documents.py demo
```

### Custom Search Script
```python
from create_embeddings import PDFEmbeddingProcessor

processor = PDFEmbeddingProcessor()
results = processor.search_similar("your query", top_k=10)

for result in results:
    print(f"Score: {result['score']}")
    print(f"File: {result['filename']}")
    print(f"Text: {result['text_preview']}")
```

### Collection Management
```python
from qdrant_client import QdrantClient

client = QdrantClient(host='localhost', port=6333)

# List collections
collections = client.get_collections()

# Get collection info
info = client.get_collection('pdf_embeddings')

# Delete collection (if needed)
client.delete_collection('pdf_embeddings')
```

## ✅ Success Indicators

When everything is working correctly, you should see:
- ✅ Qdrant server running on localhost:6333
- ✅ 202 files processed successfully
- ✅ 1500+ embeddings created
- ✅ Search returning relevant results
- ✅ Similarity scores between 0.7-1.0

## 🎉 Next Steps

After successful setup, you can:
1. **Search documents** using semantic similarity
2. **Build applications** using the embeddings
3. **Integrate with other tools** via Qdrant API
4. **Scale up** by adding more documents

---

**Need Help?** Check the logs in `embedding_process.log` for detailed information.
