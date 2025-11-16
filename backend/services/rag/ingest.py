# services/rag/ingest.py

import uuid
from services.rag.vectorstore import get_vectorstore

vs = get_vectorstore()


def ingest_text(text: str, source_type="manual", metadata=None):
    """
    Ingest a single text document into the vector store.
    Always includes the text in the payload to avoid empty retrievals.
    """
    meta = metadata or {}
    meta["source_type"] = source_type
    meta["text"] = text  # ✅ Store the actual text
    vs.add_documents(
        docs=[text],
        ids=[str(uuid.uuid4())],
        payloads=[meta]
    )
    return True


def ingest_pdf_text(pages: list[str], pdf_name: str):
    """
    Ingest multiple PDF pages into the vector store.
    Each page is stored as a separate document with metadata.
    """
    for page in pages:
        ingest_text(page, source_type="pdf", metadata={"pdf_name": pdf_name})
    return {"status": "success", "pages_ingested": len(pages)}


def ingest_youtube(text: str, video_name: str = None):
    """
    Ingest YouTube transcript or text content into the vector store.
    """
    ingest_text(
        text,
        source_type="youtube",
        metadata={"video_name": video_name or "youtube"}
    )
    return text


def ingest_web_page(text: str, page_name: str = None):
    """
    Ingest web page text content into the vector store.
    """
    ingest_text(
        text,
        source_type="web",
        metadata={"page_name": page_name or "web"}
    )
    return text
