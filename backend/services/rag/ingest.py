# services/rag/ingest.py
import fitz
from pathlib import Path
import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from bs4 import BeautifulSoup
from services.rag.utils import is_pdf, is_supported_file
from services.rag.vectorstore import get_vectorstore

# --------------------------
# PDF Loader
# --------------------------
def load_pdf(path: str):
    doc = fitz.open(path)
    return [page.get_text("text") for page in doc]

# --------------------------
# TXT / MD Loader
# --------------------------
def load_txt(path: str):
    return [Path(path).read_text()]

# --------------------------
# Web Page Loader
# --------------------------
def load_web(url: str):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text() for p in paragraphs])
    return [text] if text.strip() else []

# --------------------------
# YouTube Transcript Loader
# --------------------------
def load_youtube(url: str):
    video_id = url.split("v=")[-1].split("&")[0]
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        text = " ".join([seg.text for seg in fetched_transcript])
        return [text] if text.strip() else []
    except (TranscriptsDisabled, NoTranscriptFound):
        return []

# --------------------------
# Generic Ingest Function
# --------------------------
def ingest_file(filepath_or_url: str, source_type: str = "auto"):
    """
    source_type: "pdf", "txt", "web", "youtube", or "auto"
    """
    vs = get_vectorstore()
    chunks = []

    # Auto-detect
    if source_type == "auto":
        if filepath_or_url.startswith("http"):
            if "youtube.com" in filepath_or_url or "youtu.be" in filepath_or_url:
                source_type = "youtube"
            else:
                source_type = "web"
        else:
            fp = Path(filepath_or_url)
            if not is_supported_file(fp):
                raise ValueError("Unsupported file type")
            source_type = "pdf" if is_pdf(fp) else "txt"

    # Load content based on source
    if source_type == "pdf":
        chunks = load_pdf(filepath_or_url)
    elif source_type == "txt":
        chunks = load_txt(filepath_or_url)
    elif source_type == "web":
        chunks = load_web(filepath_or_url)
    elif source_type == "youtube":
        chunks = load_youtube(filepath_or_url)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    if chunks:
        vs.add_documents(chunks)

    return chunks

# --------------------------
# Convenience Wrappers
# --------------------------
def ingest_pdf(path: str):
    return ingest_file(path, source_type="pdf")

def ingest_text_blob(text: str, src="blob"):
    vs = get_vectorstore()
    vs.add_documents([text])
    return [text]

def ingest_web(url: str):
    return ingest_file(url, source_type="web")

def ingest_youtube(url: str):
    return ingest_file(url, source_type="youtube")
