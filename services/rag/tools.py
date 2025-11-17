# services/rag/tools.py
"""
Tool wrappers for agentic RAG.
- web_search(query): returns list of dicts {title, snippet, url}
- youtube_search(query): list of possible youtube urls
- fetch_pdf_text(url): returns plain text (best effort)
- choose_best_source(results): helper
"""

import os
import requests
from typing import List, Dict, Optional
from urllib.parse import urlparse
from services.rag.utils import clean_text

# Optional: SerpAPI key or Google CSE
SERPAPI_KEY = os.getenv("SERPAPI_KEY", None)
SERPAPI_ENGINE = os.getenv("SERPAPI_ENGINE", "serpapi")  # name only for logic readability

def web_search(query: str, limit: int = 5) -> List[Dict]:
    """
    Returns a list of search results: [{title, snippet, url}, ...]
    If SERPAPI_KEY is present, uses SerpAPI; otherwise uses a lightweight fallback (bing/duckduckgo scraping).
    NOTE: For production, swap fallback for a proper search API.
    """
    query = query.strip()
    results = []
    if SERPAPI_KEY:
        try:
            from serpapi import GoogleSearch
            params = {
                "q": query,
                "engine": "google",
                "num": limit,
                "api_key": SERPAPI_KEY,
            }
            search = GoogleSearch(params)
            resp = search.get_dict()
            organic = resp.get("organic_results", []) or resp.get("organic", [])
            for item in organic[:limit]:
                results.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet") or item.get("snippet_highlighted_words") or "",
                    "url": item.get("link") or item.get("source")
                })
            return results
        except Exception as e:
            # fallback to simple search scraping below
            print("SerpAPI error or not installed:", e)

    # Simple fallback: use DuckDuckGo HTML instant answer API
    try:
        ddg_url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        r = requests.get(ddg_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        # DuckDuckGo instant answer provides AbstractURL, RelatedTopics etc.
        if data.get("AbstractURL"):
            results.append({"title": data.get("Heading") or query, "snippet": data.get("AbstractText", ""), "url": data.get("AbstractURL")})
        # fallback: return the query as single item if nothing found
        if not results:
            results.append({"title": query, "snippet": "", "url": f"https://duckduckgo.com/?q={query}"})
        return results[:limit]
    except Exception as e:
        print("duckduckgo fallback search failed:", e)
        # Last resort: return query as link to Google
        return [{"title": query, "snippet": "", "url": f"https://www.google.com/search?q={query.replace(' ', '+')}"}]


def youtube_search(query: str, limit: int = 5) -> List[Dict]:
    """
    Lightweight search for Youtube links. For production, replace with YouTube Data API.
    Returns list of dicts: {title, url, snippet}
    """
    # If YT API key provided, use it (not implemented here)
    try:
        # quick duckduckgo + "site:youtube.com" query approach
        q = f"{query} site:youtube.com"
        # delegate to web_search which will produce a general link (may not be YT)
        hits = web_search(q, limit=limit)
        yt_hits = [h for h in hits if "youtube.com" in (h.get("url") or "") or "youtu.be" in (h.get("url") or "")]
        return yt_hits[:limit] or hits[:limit]
    except Exception:
        return []


def fetch_pdf_text(url: str, timeout: int = 15) -> Optional[str]:
    """
    Best-effort fetch PDF and extract text using pdfminer.six if available,
    otherwise return None. This is synchronous and intended for agentic ingestion.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            return None
        # write to temp file and extract with pdfminer if installed
        import tempfile
        from io import BytesIO
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(resp.content)
        tmp.close()
        try:
            # try pdfminer.six
            from pdfminer.high_level import extract_text
            txt = extract_text(tmp.name)
            return clean_text(txt)
        except Exception as e:
            print("pdfminer not available or failed:", e)
            return None
    except Exception as e:
        print("fetch_pdf_text failed:", e)
        return None


def choose_best_source(search_results: List[Dict]) -> Optional[Dict]:
    """
    Simple heuristic to pick the best source from web_search/youtube_search results.
    Currently picks the first result that looks like an article or YouTube.
    """
    for r in search_results:
        u = r.get("url", "")
        if "youtube.com" in u or "youtu.be" in u:
            return r
    # prefer urls with common article domains or not search pages
    for r in search_results:
        u = r.get("url", "")
        parsed = urlparse(u)
        if parsed.netloc and not parsed.netloc.endswith("google.com") and not parsed.netloc.endswith("duckduckgo.com"):
            return r
    # fallback
    return search_results[0] if search_results else None
