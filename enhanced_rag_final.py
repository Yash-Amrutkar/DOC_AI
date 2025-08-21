#!/usr/bin/env python3
"""
Enhanced RAG System with Optimized Search Parameters
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any, Tuple, Set
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRAGFinal:
    def __init__(self):
        # Multiple storage paths for fallback
        self.storage_configs = [
            {
                'path': "./qdrant_storage", 
                'collection': "pdf_embeddings",
                'name': "Original"
            },
            {
                'path': "./qdrant_storage_high_quality",
                'collection': "pdf_embeddings_high_quality",
                'name': "High Quality OCR"
            }
        ]
        
        # Load API key from environment
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        # Optimized retrieval settings for better coverage
        self.primary_limit = 200  # Increased to get more candidates
        self.top_k_to_use = 30   # Increased for better context
        self.score_threshold_primary = 0.1  # Lower threshold for more results
        self.min_text_length = 50  # Increased for better quality
        self.max_text_per_doc = 1200  # Increased for more context
        
        self._init_components()
        
    def _init_components(self):
        """Initialize components with fallback to original embeddings"""
        self.client = None
        self.storage_path = None
        self.collection_name = None
        self.storage_name = None
        
        # Try to connect to each storage in order
        for config in self.storage_configs:
            try:
                client = QdrantClient(path=config['path'])
                # Test if collection exists
                collection_info = client.get_collection(config['collection'])
                
                if collection_info.points_count > 0:
                    self.client = client
                    self.storage_path = config['path']
                    self.collection_name = config['collection']
                    self.storage_name = config['name']
                    logger.info(f"✅ Connected to {config['name']} with {collection_info.points_count} embeddings")
                    break
                else:
                    logger.warning(f"⚠️ {config['name']} collection is empty, trying next...")
                    
            except Exception as e:
                logger.warning(f"Failed to connect to {config['name']}: {e}")
                continue
        
        if not self.client:
            raise Exception("❌ Could not connect to any embedding storage!")
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")

    def _clean_text_for_display(self, text: str) -> str:
        """Clean text for better display"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        return text[:self.max_text_per_doc] + "..." if len(text) > self.max_text_per_doc else text

    def _extract_keywords(self, s: str) -> Set[str]:
        """Extract keywords from text"""
        s = s.lower()
        tokens = re.findall(r"[\w]+", s)
        return {t for t in tokens if len(t) >= 3}

    def _deduplicate_results(self, docs: List[Dict]) -> List[Dict]:
        """Deduplicate results"""
        seen: Set[Tuple[str, int]] = set()
        unique: List[Dict] = []
        for d in docs:
            key = (d.get("filename", "Unknown"), int(d.get("page", 0)))
            if key in seen:
                continue
            seen.add(key)
            unique.append(d)
        return unique

    def _gemini_generate(self, prompt: str) -> str:
        """Helper to call Gemini for query expansion"""
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = requests.post(f"{url}?key={self.api_key}", headers=headers, json=data, timeout=30)
            if resp.status_code != 200:
                return ""
            js = resp.json()
            return js.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        except Exception:
            return ""

    def expand_query(self, question: str) -> List[str]:
        """Create query variants for better retrieval"""
        variants: List[str] = [question]
        
        # Simple keyword extraction
        words = question.lower().split()
        keywords = [w for w in words if len(w) > 3]
        variants.extend(keywords[:5])  # Increased keywords
        
        # Try Gemini for translation/keywords
        try:
            translate_prompt = (
                "Translate the following question to English and provide 5 short keyword variations.\n"
                "Return as JSON with fields: english, keywords (array).\n\nQuestion: " + question
            )
            out = self._gemini_generate(translate_prompt)
            
            if out:
                try:
                    m = re.search(r"\{[\s\S]*\}", out)
                    if m:
                        obj = json.loads(m.group(0))
                        english = obj.get("english", "").strip()
                        if english and english.lower() != question.lower():
                            variants.append(english)
                        
                        keywords = obj.get("keywords", [])
                        for k in keywords[:5]:  # Increased keywords
                            if k and k.lower() not in [v.lower() for v in variants]:
                                variants.append(k)
                except Exception:
                    pass
        except Exception:
            pass
        
        # Ensure uniqueness
        uniq: List[str] = []
        seenq: Set[str] = set()
        for v in variants:
            v2 = v.strip()
            if not v2:
                continue
            key = v2.lower()
            if key in seenq:
                continue
            seenq.add(key)
            uniq.append(v2)
        
        logger.info(f"🧭 Query variants: {uniq[:5]}{'...' if len(uniq)>5 else ''}")
        return uniq[:8]  # Increased variants
        
    def search_documents(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant documents with improved retrieval"""
        try:
            if top_k is None:
                top_k = self.top_k_to_use

            queries = self.expand_query(query)
            query_keywords = self._extract_keywords(" ".join(queries))
            merged: List[Dict] = []

            for q in queries:
                q_vec = self.embedding_model.encode(q)
                
                # Use search method with optimized parameters
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=q_vec,
                    limit=self.primary_limit,
                    score_threshold=self.score_threshold_primary,
                )
                
                for r in results:
                    payload = r.payload or {}
                    raw_text = payload.get("text", "")
                    cleaned = self._clean_text_for_display(raw_text)
                    
                    if len(cleaned) < self.min_text_length:
                        continue
                    
                    # Enhanced keyword overlap boosting
                    cand_keywords = self._extract_keywords(cleaned)
                    overlap = len(query_keywords.intersection(cand_keywords))
                    boosted = float(r.score or 0.0) + 0.1 * (overlap ** 0.5)  # Increased boost
                    
                    merged.append({
                        "text": cleaned,
                        "score": float(r.score or 0.0),
                        "boosted": boosted,
                        "overlap": overlap,
                        "filename": payload.get("filename", "Unknown"),
                        "page": payload.get("page", 0)
                    })

            # Sort by boosted score and deduplicate
            merged.sort(key=lambda x: x["boosted"], reverse=True)
            merged = self._deduplicate_results(merged)
            selected = merged[:top_k]
            
            logger.info(
                f"🔍 Selected {len(selected)} docs (top scores: " +
                f"{[round(d['boosted'],3) for d in selected[:3]]})"
            )
            return selected
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
        
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer with improved prompt engineering"""
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
            headers = {"Content-Type": "application/json"}
            
            # Enhanced prompt for better answers
            prompt = (
                "You are an expert assistant analyzing documents from Pune Municipal Corporation. "
                "The context may contain text in multiple languages (English, Marathi, Hindi).\n\n"
                "Instructions:\n"
                "1. Answer the question based on the provided context\n"
                "2. If the context contains relevant information, provide a detailed and helpful answer\n"
                "3. If the context is unclear or doesn't contain relevant information, say: 'I cannot find specific information about this in the provided documents.'\n"
                "4. Respond in the same language as the question\n"
                "5. Be comprehensive and provide specific details when available\n"
                "6. If the question is in Marathi, you can respond in Marathi\n"
                "7. Look for patterns and connections in the data\n"
                "8. If you find partial information, mention what you found and what's missing\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{context}\n\n"
                "Please provide a clear, detailed answer:"
            )

            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(f"{url}?key={self.api_key}", headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates'] and \
                   'content' in result['candidates'][0] and \
                   'parts' in result['candidates'][0]['content'] and \
                   result['candidates'][0]['content']['parts']:
                    return result['candidates'][0]['content']['parts'][0].get('text', '').strip()
                return "I cannot find specific information about this in the provided documents."
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return "I cannot find specific information about this in the provided documents."
                
        except requests.exceptions.Timeout:
            return "I cannot find specific information about this in the provided documents."
        except Exception as e:
            logger.error(f"❌ Answer generation error: {e}")
            return "I cannot find specific information about this in the provided documents."
            
    def answer_question(self, question: str) -> str:
        """Main method to answer questions"""
        logger.info(f"🤔 Question: {question}")
        logger.info(f"📊 Using storage: {self.storage_name}")
        
        docs = self.search_documents(question)
        if not docs:
            return "I cannot find specific information about this in the provided documents."
        
        # Build context from top documents
        context_parts: List[str] = []
        for d in docs:
            context_parts.append(
                f"Source: {d['filename']} (Page {d['page']}, relevance: {round(d['boosted'],3)})\n{d['text']}"
            )
        context = "\n\n".join(context_parts)
        
        answer = self.generate_answer(question, context)
        logger.info("✅ Answer generated successfully")
        return answer

def main():
    """Main function for testing"""
    try:
        print("🚀 Initializing Enhanced RAG System Final...")
        rag = EnhancedRAGFinal()
        
        print("\n✅ System ready! You can now ask questions.")
        print("💡 Example questions:")
        print("   - What is this document about?")
        print("   - What are the main topics?")
        print("   - Can you summarize the content?")
        print("   - जून 2025 मध्ये कोणत्या क्षेत्रीय कार्यालयाने सर्वाधिक काम केले?")
        print("   - Type 'quit' to exit\n")
        
        while True:
            question = input("🤔 Ask a question: ")
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if question.strip():
                print("🤔 Thinking...")
                answer = rag.answer_question(question)
                print(f"\n✅ Answer: {answer}\n")
            else:
                print("Please enter a question.\n")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
