# RAG QnA System Setup Guide

## 🚀 Overview

This guide will help you set up a RAG (Retrieval-Augmented Generation) QnA system using:
- **Gemini 2.5 Pro API** for text generation
- **Qdrant Vector Database** for document retrieval
- **Flask Web Interface** for user interaction
- **Sentence Transformers** for embeddings

## 📋 Prerequisites

- Python 3.8+
- Internet connection for API access
- Google Cloud account (for Gemini API)

## 🔑 Step 1: Get Gemini API Key

### Option A: Google AI Studio (Recommended)
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click on "Get API key" in the top right
4. Create a new API key or use existing one
5. Copy the API key

### Option B: Google Cloud Console
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the "Generative Language API"
4. Go to "APIs & Services" > "Credentials"
5. Create an API key
6. Copy the API key

## 🛠️ Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements_embeddings.txt
```

## ⚙️ Step 3: Configure Environment

1. **Edit the .env file**:
```bash
nano .env
```

2. **Replace the API key**:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

3. **Save the file**

## 🎯 Step 4: Test the System

### Command Line Interface
```bash
# Test the RAG system in terminal
python3 rag_qna_system.py
```

### Web Interface
```bash
# Start the web server
python3 web_interface.py
```

## 🌐 Step 5: Access Web Interface

1. The web interface will automatically open in your browser
2. If not, manually go to: `http://127.0.0.1:5000`
3. Start asking questions!

## 📊 Available Models

The system supports multiple Gemini models:

| Model | Description | Speed | Capability |
|-------|-------------|-------|------------|
| `gemini-2.0-flash-exp` | Latest, fastest | ⚡⚡⚡ | High |
| `gemini-1.5-pro` | Most capable | ⚡⚡ | Very High |
| `gemini-1.5-flash` | Balanced | ⚡⚡⚡ | High |
| `gemini-1.0-pro` | Legacy | ⚡⚡ | Medium |

## 🔍 Example Questions

### English Questions:
- "What are the environmental policies of Pune Municipal Corporation?"
- "How do I apply for a building permit?"
- "What are the fire safety requirements?"
- "Tell me about the budget allocation for 2024-2025"

### Marathi Questions:
- "पुणे महानगरपालिकेच्या पर्यावरण धोरणांबद्दल माहिती द्या"
- "इमारत परवाना कसा मिळवायचा?"
- "अग्निशमन सुरक्षा आवश्यकता काय आहेत?"

## 📁 Project Structure

```
DOC_AI/
├── rag_qna_system.py          # Main RAG system
├── web_interface.py           # Flask web server
├── create_embeddings_persistent.py  # Embedding creation
├── check_database.py          # Database checker
├── requirements_embeddings.txt # Dependencies
├── .env                       # Configuration
├── qdrant_storage/            # Vector database
├── extraceted PDF data/       # Source documents
└── templates/                 # Web templates (auto-created)
```

## 🔧 Configuration Options

### Environment Variables (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Your Gemini API key | Required |
| `QDRANT_COLLECTION_NAME` | Vector database collection | `pdf_embeddings` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `TOP_K_RESULTS` | Number of documents to retrieve | `5` |
| `FLASK_HOST` | Web server host | `127.0.0.1` |
| `FLASK_PORT` | Web server port | `5000` |

## 🚨 Troubleshooting

### Common Issues:

1. **"GEMINI_API_KEY not found"**
   - Check your .env file
   - Ensure the API key is correctly set

2. **"Collection not found"**
   - Run `python3 create_embeddings_persistent.py` first
   - Check if `qdrant_storage/` directory exists

3. **"Module not found"**
   - Install dependencies: `pip install -r requirements_embeddings.txt`

4. **"API quota exceeded"**
   - Check your Google Cloud billing
   - Monitor API usage in Google AI Studio

5. **Web interface not loading**
   - Check if port 5000 is available
   - Try different port in .env file

### Performance Tips:

1. **Use gemini-2.0-flash-exp** for fastest responses
2. **Reduce TOP_K_RESULTS** for faster search
3. **Monitor API usage** to avoid quota limits
4. **Use specific questions** for better results

## 📈 Usage Statistics

The system tracks:
- Response time (total, search, generation)
- Documents used for each answer
- Relevance scores
- Context length
- Model used

## 🔒 Security Notes

1. **Never commit your API key** to version control
2. **Use environment variables** for sensitive data
3. **Change FLASK_SECRET_KEY** in production
4. **Monitor API usage** to prevent abuse

## 🎉 Ready to Use!

Your RAG QnA system is now ready! Start with:

```bash
# Command line interface
python3 rag_qna_system.py

# Or web interface
python3 web_interface.py
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify your API key is valid
3. Ensure all dependencies are installed
4. Check the logs for error messages

---

**Happy Questioning! 🤖✨**
