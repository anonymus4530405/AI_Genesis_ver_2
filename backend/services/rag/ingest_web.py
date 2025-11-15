# services/rag/ingest_web.py
import requests
from bs4 import BeautifulSoup
from services.rag.vectorstore import get_vectorstore

def split_text_into_chunks(text: str, max_words: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def ingest_web(url: str):
    # Fetch webpage
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    # Extract main text
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text() for p in paragraphs]).strip()

    if not text:
        return []

    # Split text into chunks
    chunks = split_text_into_chunks(text)

    # Add to vectorstore
    vs = get_vectorstore()
    vs.add_documents(chunks)

    return chunks
