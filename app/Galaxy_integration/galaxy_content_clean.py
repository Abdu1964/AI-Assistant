
from bs4 import BeautifulSoup
import pandas as pd
import json
import logging
import re
import asyncio
import os
import openai
import uuid
import requests
import trafilatura
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import models
from app.llm_handle.llm_models import get_llm_model


logger = logging.getLogger(__name__)


class HTMLProcessor:
    def __init__(self, qdrant_client, llm):
        self.qdrant = qdrant_client
        # MUCH SMALLER CHUNKS - better for embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Changed from 8000
            chunk_overlap=200
        )
        self.llm = llm

    def deep_clean_text(self, text: str) -> str:
        """Aggressive cleaning for better embeddings"""
        
        # Remove chart URLs and image references
        text = re.sub(r'http://chart\.apis\.google\.com[^\s\)]+', '', text)
        text = re.sub(r'!\[\]\([^)]+\)', '', text)
        
        # Remove HTML entities
        text = re.sub(r'&nbsp;?', ' ', text)
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        
        # Remove excessive pipes and table formatting
        text = re.sub(r'\|+', '|', text)  # Multiple pipes to single
        text = re.sub(r'(\|\s*\|){2,}', '| ', text)  # Empty table cells
        text = re.sub(r'\|\s*-+\s*\|', '', text)  # Table separators
        
        # Remove chart data arrays and codes
        text = re.sub(r'chd=e:[A-Za-z0-9\.]+', '', text)
        text = re.sub(r'ch[a-z]{2,3}=[^\s&]+', '', text)
        
        # Remove excessive dashes and underscores
        text = re.sub(r'-{3,}', '', text)
        text = re.sub(r'_{3,}', '', text)
        text = re.sub(r'\*{3,}', '', text)
        
        # Remove position/count arrays
        text = re.sub(r'Position,\d+(?:,\d+)*', '', text)
        text = re.sub(r'Count,\d+(?:,\d+)*', '', text)
        
        # Clean excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n +', '\n', text)
        
        # Remove lines that are mostly special characters
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            # Count alphanumeric vs special chars
            alnum = sum(c.isalnum() for c in line)
            total = len(line.strip())
            if total > 0 and alnum / total > 0.3:  # At least 30% real content
                clean_lines.append(line)
        
        text = '\n'.join(clean_lines)
        
        # Final cleanup
        text = text.strip()
        
        return text


    def extract_html(self, source: str) -> Optional[str]:
        """
        Extracts clean text content from a URL using Trafilatura.
        """

        # Handle list input (e.g., [url])
        if isinstance(source, list):
            if not source:
                logger.warning("Empty source list provided")
                return None
        source = source[0]

        url = str(source).strip()
        logger.info(f"Attempting Trafilatura extraction for {url}")

        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                logger.warning(f"Trafilatura failed to fetch {url}")
                return None

            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if not text:
                logger.warning(f"Trafilatura failed to extract text for {url}")
                return None

        except Exception as e:
            logger.error(f"Trafilatura extraction error for {url}: {e}")
            return None

        # --- Clean up extracted text ---
        text = re.sub(r'&nbsp;?', ' ', text)
        text = re.sub(r'^[A-Z]{1,3}(?:\s*\|\s*[A-Z]{1,3})+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-\s\|]+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\s*\|(?:\s*\d*\s*\|?)+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*$', '', text, flags=re.MULTILINE)

        text = text.strip()
        logger.info(f"✅ Extracted and cleaned text length: {len(text)} chars")

        return text

           
    def _summarize_with_llm(self, text):
        """Create DETAILED summary capturing all key information"""
        prompt = f"""
You are analyzing scientific/technical content. Create a comprehensive, detailed summary.

REQUIREMENTS:
1. Capture ALL important details, numbers, statistics, findings
2. Include specific values, percentages, counts mentioned
3. Preserve technical terms and methodology details
4. List all key topics and concepts discussed
5. Keep the summary well-structured and clear

Return ONLY valid JSON (no markdown, no extra text) with this EXACT structure:
{{
    "summary": "detailed multi-sentence summary capturing all key information and specific values"
}}

TEXT TO ANALYZE:
{text[:4000]}

Remember: Return ONLY the JSON object, nothing else.
"""
        
        try:
            result = self.llm.generate(prompt)
            
            # Parse JSON response
            if isinstance(result, str):
                # Remove markdown code blocks if present
                result = re.sub(r'```json\s*', '', result)
                result = re.sub(r'```\s*', '', result)
                result = result.strip()
                
                import json
                result = json.loads(result)
            
            if isinstance(result, dict) and "summary" in result:
                return result["summary"]
            else:
                return str(result)
                
        except Exception as e:
            print(f"⚠️ Summary generation failed: {e}")
            # Fallback: use first 500 chars as summary
            return text[:500].replace('\n', ' ').strip()

    def store_embedded(self, url: str, collection_name: str):
        """Process URL and store in Qdrant with detailed summaries."""
        html_text = self.extract_html(url)
        if not html_text:
            return "Failed to extract HTML"
        print(html_text)
        cleaned_text = self.deep_clean_text(html_text)
        print(cleaned_text)
        chunks = self.splitter.split_text(html_text)
        for i, chunk in enumerate(chunks):
            print(chunk)
            summary_text = self._summarize_with_llm(chunk)
            
            # Metadata per chunk, content_id as string
            chunk_metadata = {
                "content_id": url,       # string, not list
                "summary": summary_text,
                "chunk_index": i
            }

            # Upsert each chunk
            self.qdrant.upsert_data(
                collection_name=collection_name,
                data=None,
                is_content=True,
                chunks=[chunk],          # single chunk
                metadata=chunk_metadata   # single dict
            )
            logger.info(f"chunk{i} is saved in qdrant")
        logger.info(f"📦 finsihed chunking {len(chunks)} chunks")
        return url
        # Process each chunk with detailed summary
        enhanced_data = []
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}...")
            
            summary_text = self._summarize_with_llm(chunk)
            
            enhanced_data.append({
                "text": chunk,
                "content_id": url,
                "summary": summary_text,
                "chunk_index": i
            })
        
        # Insert in batches
        self.qdrant.ensure_collection_exists(collection_name)
        
        for i in range(0, len(enhanced_data), self.qdrant.batch_size):
            batch_data = enhanced_data[i : i + self.qdrant.batch_size]
            
            texts = [item["text"] for item in batch_data]
            embeddings = self.qdrant._get_embeddings(texts)
            
            points = []
            for item, emb in zip(batch_data, embeddings):
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=emb,
                        payload=item
                    )
                )
            
            self.qdrant.client.upsert(collection_name=collection_name, points=points)
        
        print(f"✅ Stored {len(chunks)} chunks to Qdrant")
        return f"Successfully processed {len(chunks)} chunks"