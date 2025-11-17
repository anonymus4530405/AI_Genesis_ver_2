from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF for PDF extraction

# Corrected Agentic RAG
from services.rag.api_adapter import run_agentic_rag
from services.rag.ingest import ingest_text, ingest_pdf_text
from services.rag.ingest_web import ingest_web
from services.rag.ingest_youtube import ingest_youtube
from services.rag.vectorstore import get_vectorstore

router = APIRouter()
vs = get_vectorstore()


# -------------------------------
# Models
# -------------------------------
class QueryRequest(BaseModel):
    query: str


class IngestURL(BaseModel):
    url: str


class IngestYouTubeRequest(BaseModel):
    video_id: str


# -------------------------------
# Agentic RAG Query
# -------------------------------
@router.post("/query")
async def query_rag(req: QueryRequest):
    try:
        result = run_agentic_rag(req.query)
        return {
            "query": req.query,
            "answer": result.get("response"),
            "intent": result.get("intent"),
            "new_ingestion_done": result.get("new_ingestion"),
            "meta": result.get("meta", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Manual Text Ingestion
# -------------------------------
@router.post("/ingest/text")
async def ingest_text_route(text: str):
    try:
        ingest_text(text, source_type="manual")
        return {"status": "success", "ingested": text[:80] + "..."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# PDF Ingestion
# -------------------------------
@router.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        pages = [page.get_text() for page in pdf_doc]
        ingest_pdf_text(pages, pdf_name=file.filename)

        return {
            "status": "success",
            "pages": len(pages),
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Web URL Ingestion
# -------------------------------
@router.post("/ingest/web")
async def ingest_web_route(req: IngestURL):
    try:
        ingest_web(req.url)
        return {"status": "success", "source": req.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# YouTube Ingestion
# -------------------------------
@router.post("/ingest/youtube")
async def ingest_youtube_route(req: IngestYouTubeRequest):
    try:
        ingest_youtube(req.video_id)
        return {"status": "success", "video_id": req.video_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Clear Vector Store by Source Type
# -------------------------------
@router.delete("/clear/{source_type}")
async def clear_source(source_type: str):
    try:
        vs.delete_by_source(source_type)
        return {"status": "cleared", "source_type": source_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Health Check
# -------------------------------
@router.get("/health")
async def health_check():
    return {"status": "ok"}
