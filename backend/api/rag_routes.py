# services/api/rag_routes.py
from fastapi import APIRouter, UploadFile, File, Form
import tempfile, os, re
from services.rag.ingest import ingest_pdf, ingest_text_blob
from services.rag.ingest_web import ingest_web
from services.rag.ingest_youtube import ingest_youtube
from services.rag.graph import agentic_rag_answer
from services.rag.vectorstore import get_vectorstore

router = APIRouter()

# Regex to detect URLs in messages
URL_PATTERN = re.compile(r"https?://[^\s]+")

# --------------------------
# Health check
# --------------------------
@router.get("/health")
def health():
    return {"status": "ok"}

# --------------------------
# PDF upload
# --------------------------
@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(await file.read())
        tmp_path = tf.name
    try:
        chunks = ingest_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)
    return {"filename": file.filename, "ingested_chunks": len(chunks)}

# --------------------------
# Text upload
# --------------------------
@router.post("/upload_text")
async def upload_text(text: str = Form(...), source: str = Form("text_blob")):
    chunks = ingest_text_blob(text, source)
    return {"source": source, "ingested_chunks": len(chunks)}

@router.post("/chat_or_ingest")
def chat_or_ingest(message: str = Form(...)):
    message = message.strip()
    urls = URL_PATTERN.findall(message)

    # If message contains URL
    if urls:
        combined_chunks = []
        for url in urls:
            url = url.rstrip('.,;!?')  # remove trailing punctuation
            try:
                if "youtube.com" in url or "youtu.be" in url:
                    chunks = ingest_youtube(url)
                else:
                    chunks = ingest_web(url)
            except Exception as e:
                chunks = []
                print(f"Error ingesting URL {url}: {e}")
            combined_chunks.extend(chunks)

        # If message has extra text after URL, treat as question
        question_text = URL_PATTERN.sub("", message).strip()
        if not question_text:
            # Default question if only URL was given
            question_text = "Please summarize or explain the content of this URL."

        # Ask RAG agent
        try:
            # Temporarily override the state to use the newly ingested chunks
            vs = get_vectorstore()
            # You could also directly pass chunks to the agent if your RAG supports it
            answer = agentic_rag_answer.invoke({"question": question_text})["answer"]
        except Exception as e:
            answer = f"❌ Error generating answer: {e}"
            print(f"/chat_or_ingest error: {e}")

        return {"type": "chat", "answer": answer}

    # Otherwise treat it as a normal query
    try:
        answer = agentic_rag_answer.invoke({"question": message})["answer"]
    except Exception as e:
        answer = f"❌ Error generating answer: {e}"
        print(f"/chat_or_ingest error: {e}")

    return {"type": "chat", "answer": answer}

