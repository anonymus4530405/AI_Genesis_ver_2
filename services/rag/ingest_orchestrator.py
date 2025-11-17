# services/rag/ingest_orchestrator.py

import re
from typing import Optional, Dict, List
from services.rag.tools import fetch_pdf_text, web_search, choose_best_source, SERPAPI_KEY
from services.rag.memory import MemoryManager
from services.rag.ingest import ingest_pdf_text
from services.rag.ingest_web import ingest_web
from services.rag.ingest_youtube import ingest_youtube

URL_REGEX = r"(https?://[^\s]+)"

class IngestOrchestrator:
    def __init__(self):
        self.memory = MemoryManager()

    def find_url_in_message(self, text: str) -> Optional[str]:
        m = re.search(URL_REGEX, text)
        return m.group(0) if m else None

    def auto_ingest_if_needed(self, query: str) -> Optional[Dict]:
        url_in_msg = self.find_url_in_message(query)
        if url_in_msg:
            return self._handle_url(url_in_msg)
        return self._discover_and_ingest(query)

    def _handle_url(self, url: str) -> Optional[Dict]:
        if self.memory.has_source(url):
            return None

        # PDF
        if url.lower().endswith(".pdf"):
            txt = fetch_pdf_text(url)
            if txt:
                ingest_pdf_text([txt], pdf_name=url)
                self.memory.register_source(url, "pdf", "PDF Document")
                return {"url": url, "type": "pdf", "text": txt}

        # YouTube
        if "youtube.com" in url or "youtu.be" in url:
            txt = ingest_youtube(url)
            if txt:
                self.memory.register_source(url, "youtube", "YouTube Video")
                return {"url": url, "type": "youtube", "text": str(txt)}

        # Web page
        txt = ingest_web(url)
        if txt:
            self.memory.register_source(url, "web", "Web Page")
            return {"url": url, "type": "web", "text": str(txt)}

        return None

    def _discover_and_ingest(self, query: str) -> Optional[Dict]:
        results = web_search(query)
        best = choose_best_source(results)

        # If no best source or already ingested, try Google fallback
        if not best or self.memory.has_source(best["url"]):
            results = self._google_search_fallback(query)
            best = choose_best_source(results)

            if not best:
                return {
                    "url": None,
                    "type": "none",
                    "text": "I could not find relevant information. Consider uploading a PDF, link, or checking online."
                }

        return self._ingest_url(best)

    def _ingest_url(self, best: Dict) -> Optional[Dict]:
        url = best["url"]
        title = best.get("title", "Source")
        if self.memory.has_source(url):
            return None

        # PDF
        if url.lower().endswith(".pdf"):
            txt = fetch_pdf_text(url)
            if txt:
                ingest_pdf_text([txt], pdf_name=url)
                self.memory.register_source(url, "pdf", title)
                return {"url": url, "type": "pdf", "text": txt}

        # YouTube
        if "youtube.com" in url or "youtu.be" in url:
            txt = ingest_youtube(url)
            if txt:
                self.memory.register_source(url, "youtube", title)
                return {"url": url, "type": "youtube", "text": str(txt)}

        # Web page
        txt = ingest_web(url)
        if txt:
            self.memory.register_source(url, "web", title)
            return {"url": url, "type": "web", "text": str(txt)}

        return {
            "url": url,
            "type": "none",
            "text": "I could not find relevant information. Consider uploading a PDF, link, or checking online."
        }

    def _google_search_fallback(self, query: str) -> List[Dict]:
        if not SERPAPI_KEY:
            return []
        try:
            from serpapi import GoogleSearch
            params = {"q": query, "engine": "google", "num": 5, "api_key": SERPAPI_KEY}
            search = GoogleSearch(params)
            resp = search.get_dict()
            results = []
            for item in resp.get("organic_results", []):
                results.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link")
                })
            return results
        except Exception as e:
            print("[ERROR] Google fallback search failed:", e)
            return []
