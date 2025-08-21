# DOC_AI - Pune Municipal Corporation Document Q&A System

## 🏛️ Project Overview

DOC_AI is a comprehensive RAG (Retrieval-Augmented Generation) based Q&A system designed specifically for Pune Municipal Corporation (PMC) documents. The system provides intelligent question-answering capabilities for PMC's vast collection of administrative documents, supporting multiple languages including English, Marathi, and Hindi.

## ✨ Features

- **Multilingual Support**: English, Marathi, and Hindi
- **Comprehensive Coverage**: All PMC departments and regional offices
- **High-Quality OCR**: Advanced text extraction and processing
- **Intelligent Search**: Semantic search with keyword boosting
- **Web Interface**: User-friendly Flask-based web application
- **Real-time Processing**: Background OCR improvement and embedding generation

## 🏢 Departments Covered

- **Health Department** - Public health services and regulations
- **Water Supply Department** - Water connection and maintenance
- **Garden Department** - Park maintenance and horticulture
- **Electrical Department** - Street lighting and electrical services
- **Building Development** - Construction permits and regulations
- **Drainage Department** - Sewerage and drainage systems
- **Taxation Department** - Property tax and collection
- **Tree Authority** - Tree cutting permits and plantation
- **Regional Offices** - All ward offices and regional centers
- **RTI Act 2005** - Information disclosure and procedures

## 🛠️ Technical Stack

- **Backend**: Python 3.x
- **Web Framework**: Flask
- **Vector Database**: Qdrant (local persistent storage)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini 2.0 Flash Exp
- **OCR**: Tesseract with image preprocessing
- **Frontend**: HTML, CSS, JavaScript

## 📁 Project Structure

```
DOC_AI/
├── extraceted PDF data/          # Original PDF extraction data (205 files)
├── qdrant_storage/              # Vector database storage
├── qdrant_storage_high_quality/ # High-quality OCR embeddings
├── enhanced_rag_final.py        # Main RAG system
├── enhanced_web_interface.py    # Flask web interface
├── ocr_quality_fixer.py         # OCR improvement script
├── templates/                   # HTML templates
├── .env                        # Environment variables (API keys)
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR**
3. **Google Gemini API Key**

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yash-Amrutkar/DOC_AI.git
   cd DOC_AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

4. **Start the web interface:**
   ```bash
   python3 enhanced_web_interface.py
   ```

5. **Access the system:**
   - Open browser: `http://localhost:5000`
   - Start asking questions in English, Marathi, or Hindi

## 📊 System Statistics

- **Total Documents**: 205 PDF files
- **Total Embeddings**: 10,446 chunks
- **Languages Supported**: 3 (English, Marathi, Hindi)
- **Departments Covered**: 10+ major departments
- **Regional Offices**: All PMC ward offices

## 🎯 Sample Questions

### English Questions
- "What are the functions of the Health Department?"
- "What are the water connection procedures?"
- "What are the building permit requirements?"
- "What information is disclosed under RTI Act 2005?"

### Marathi Questions
- "आरोग्य विभागाची कार्ये काय आहेत?"
- "पाणी कनेक्शन प्रक्रिया काय आहे?"
- "बांधकाम परवानगी आवश्यकता काय आहेत?"
- "माहिती अधिकार अधिनियम 2005 अंतर्गत कोणती माहिती प्रसिद्ध केली जाते?"

## 🔧 Advanced Features

### OCR Quality Improvement
```bash
python3 ocr_quality_fixer.py
```

### High-Quality Embeddings Generation
```bash
python3 final_embedding_creator.py
```

### System Diagnostics
```bash
python3 diagnose.py
```

## 📈 Performance Metrics

- **Response Time**: < 3 seconds
- **Accuracy**: High semantic relevance
- **Multilingual**: Native language support
- **Scalability**: Handles 10,000+ document chunks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Yash Amrutkar** - Initial work - [Yash-Amrutkar](https://github.com/Yash-Amrutkar)

## 🙏 Acknowledgments

- Pune Municipal Corporation for document access
- Google Gemini for LLM capabilities
- Qdrant for vector database
- Sentence Transformers for embeddings

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact: [Your Contact Information]

---

**Note**: This system is specifically designed for Pune Municipal Corporation documents and may require customization for other municipal corporations.