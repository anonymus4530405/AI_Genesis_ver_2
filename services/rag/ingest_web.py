import uuid
import requests
from bs4 import BeautifulSoup
from services.rag.vectorstore import get_vectorstore

vs = get_vectorstore()
CHUNK_SIZE = 800  # max characters per chunk

def ingest_web(url: str):
    """
    Ingest a web page into the vector store in character-based chunks.
    Keeps each chunk around CHUNK_SIZE characters for embedding purposes.
    """
    try:
        html = requests.get(url, timeout=15).text
    except Exception as e:
        return {"status": "failed", "reason": str(e)}

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) >= 50]

    buffer = ""
    chunks_count = 0

    for line in lines:
        # Add line to buffer
        if len(buffer) + len(line) + 1 <= CHUNK_SIZE:
            buffer += " " + line
        else:
            # If buffer is full, add to vector store
            vs.add_documents(
                docs=[buffer.strip()],
                ids=[str(uuid.uuid4())],
                payloads=[{"source_type": "web", "url": url}]
            )
            chunks_count += 1
            buffer = line  # start new buffer with current line

    # Add any leftover text as final chunk
    if buffer.strip():
        vs.add_documents(
            docs=[buffer.strip()],
            ids=[str(uuid.uuid4())],
            payloads=[{"source_type": "web", "url": url}]
        )
        chunks_count += 1

    return {"status": "success", "chunks": chunks_count}
