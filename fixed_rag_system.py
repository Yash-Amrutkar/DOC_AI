#!/usr/bin/env python3
"""
Fixed RAG System with proper API key handling
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

# Load environment variables with override
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedRAGSystem:
    def __init__(self):
        self.storage_path = "./qdrant_storage"
        self.collection_name = "pdf_embeddings"
        # Load API key from environment
        self.api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCFmWEAywTYuMuRKUZBgpZ6MQ5v7Q7EDHw')
        # Retrieval settings tuned for noisy OCR
        self.primary_limit = 400
        self.top_k_to_use = 24
        self.score_threshold_primary = 0.0
        self.min_text_length = 20
        self.max_text_per_doc = 900
        # Keyword boosting weights
        self.keyword_min_len = 3
        self.keyword_boost_weight = 0.08
        self._init_components()
        
    def _init_components(self):
        """Initialize all components"""
        try:
            self.client = QdrantClient(path=self.storage_path)
            logger.info("✅ Connected to Qdrant database")
            
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean OCR text for better model consumption."""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\w\s\.,;:!\-()\[\]/%₹]+", "", text)
        text = re.sub(r"([,.;:!\-])\1{2,}", r"\1", text)
        if len(text) > self.max_text_per_doc:
            text = text[: self.max_text_per_doc] + "..."
        return text

    def _extract_keywords(self, s: str) -> Set[str]:
        s = s.lower()
        tokens = re.findall(r"[\w]+", s)
        return {t for t in tokens if len(t) >= self.keyword_min_len}

    def _deduplicate_results(self, docs: List[Dict]) -> List[Dict]:
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
        variants: List[str] = [question]
        translate_prompt = (
            "Translate the following question to English and provide 5 short keyword variations.\n"
            "Return as JSON with fields: english, keywords (array).\n\nQuestion: " + question
        )
        out = self._gemini_generate(translate_prompt)
        english = ""
        keywords: List[str] = []
        if out:
            try:
                m = re.search(r"\{[\s\S]*\}", out)
                if m:
                    obj = json.loads(m.group(0))
                    english = obj.get("english", "").strip()
                    if isinstance(obj.get("keywords"), list):
                        keywords = [str(k) for k in obj["keywords"] if k]
            except Exception:
                pass
        if english and english.lower() != question.lower():
            variants.append(english)
        for k in keywords[:5]:
            if k and k.lower() not in [v.lower() for v in variants]:
                variants.append(k)
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
        logger.info(f"🧭 Query variants: {uniq[:3]}{'...' if len(uniq)>3 else ''}")
        return uniq[:6]
        
    def search_documents(self, query: str, top_k: int = None) -> List[Dict]:
        try:
            if top_k is None:
                top_k = self.top_k_to_use

            queries = self.expand_query(query)
            query_keywords = self._extract_keywords(" ".join(queries))
            merged: List[Dict] = []

            for q in queries:
                q_vec = self.embedding_model.encode(q)
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=q_vec,
                    limit=self.primary_limit,
                    score_threshold=self.score_threshold_primary,
                )
                for r in results:
                    payload = r.payload or {}
                    raw_text = payload.get("text", "")
                    cleaned = self._clean_text(raw_text)
                    if len(cleaned) < self.min_text_length:
                        continue
                    cand_keywords = self._extract_keywords(cleaned)
                    overlap = len(query_keywords.intersection(cand_keywords))
                    boosted = float(r.score or 0.0) + self.keyword_boost_weight * (overlap ** 0.5)
                    merged.append({
                        "text": cleaned,
                        "score": float(r.score or 0.0),
                        "boosted": boosted,
                        "overlap": overlap,
                        "filename": payload.get("filename", "Unknown"),
                        "page": payload.get("page", 0)
                    })

            merged.sort(key=lambda x: x["boosted"], reverse=True)
            merged = self._deduplicate_results(merged)
            selected = merged[:top_k]
            logger.info(
                f"🔍 Selected {len(selected)} docs (top boosted: " +
                f"{[round(d['boosted'],3) for d in selected[:3]]}, " +
                f"overlaps: {[d['overlap'] for d in selected[:3]]})"
            )
            return selected
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
        
    def generate_answer(self, question: str, context: str) -> str:
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
            headers = {"Content-Type": "application/json"}
            
            prompt = (
                "You are a helpful assistant answering questions using noisy OCR text that may contain "
                "errors and mixed languages (Marathi/Hindi/English).\n"
                "- If the context is fragmented, infer the most likely meaning.\n"
                "- Prefer facts present in the context.\n"
                "- If insufficient information, say: 'I cannot find specific information about this in the provided documents.'\n"
                "- Respond in the same language as the question.\n\n"
                f"Question: {question}\n\n"
                f"Context (may contain OCR errors):\n{context}\n\n"
                "Provide a clear and concise answer:"
            )

            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(f"{url}?key={self.api_key}", headers=headers, json=data, timeout=45)
            
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
        logger.info(f"🤔 Question: {question}")
        docs = self.search_documents(question)
        if not docs:
            return "I cannot find specific information about this in the provided documents."
        context_parts: List[str] = []
        for d in docs:
            context_parts.append(
                f"Source: {d['filename']} (Page {d['page']}, score {round(d['score'],3)})\n{d['text']}"
            )
        # Join more context to help LLM overcome noise
        context = "\n\n".join(context_parts)
        answer = self.generate_answer(question, context)
        logger.info("✅ Answer generated successfully")
        return answer


def main():
    try:
        print("🚀 Initializing Fixed RAG System...")
        rag = FixedRAGSystem()
        
        print("\n✅ System ready! You can now ask questions.")
        print("💡 Example questions:")
        print("   - What is this document about?")
        print("   - What are the main topics?")
        print("   - Can you summarize the content?")
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
