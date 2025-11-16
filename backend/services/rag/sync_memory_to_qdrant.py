# services/rag/sync_memory_to_qdrant.py

import os
from services.rag.memory import MemoryManager
from services.rag.vectorstore import get_vectorstore
from services.rag.tools import fetch_pdf_text, web_search, youtube_search
from services.rag.ingest import ingest_pdf_text
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

def ingest_source(url: str):
    """
    Determine source type and ingest into Qdrant.
    """
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        from services.rag.ingest_youtube import ingest_youtube
        video_id = url.split("v=")[-1].split("&")[0]  # crude extraction
        ingest_youtube(video_id)
    elif url.lower().endswith(".pdf"):
        pdf_text = fetch_pdf_text(url)
        if pdf_text:
            ingest_pdf_text([pdf_text], pdf_name=url)
    else:
        from services.rag.ingest_web import ingest_web
        ingest_web(url)

def main():
    print("\n=== Syncing Supabase Memory Store to Qdrant ===\n")
    memory = MemoryManager()
    vs = get_vectorstore()

    all_sources = memory.data.get("sources", {})
    print(f"Found {len(all_sources)} sources in memory.")

    for i, (url, meta) in enumerate(all_sources.items(), 1):
        print(f"[{i}/{len(all_sources)}] Ingesting: {url}")
        try:
            ingest_source(url)
        except Exception as e:
            print(f"Failed to ingest {url}: {e}")

    print("\n✅ Memory synced to Qdrant successfully!\n")

if __name__ == "__main__":
    main()
