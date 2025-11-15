# test_pipeline.py
import os
from services.rag.ingest import ingest_pdf, ingest_text_blob , ingest_youtube , ingest_web
from services.rag.graph import agentic_rag_answer

# # --------------------------
# # Test PDF Ingestion
# # --------------------------
# PDF_PATH = "sample.pdf"

# print("=== TEST: PDF Ingestion ===")
# try:
#     chunks = ingest_pdf(PDF_PATH)
#     if not chunks:
#         print("⚠️ No content ingested from PDF!")
#     else:
#         print(f"✅ Chunks stored: {len(chunks)}")
# except Exception as e:
#     print(f"❌ Error during ingestion: {e}")
#     chunks = []

# # --------------------------
# # Test Text Blob Ingestion
# # --------------------------
# TEXT_BLOB = "This is a sample text blob for ingestion testing."

# try:
#     text_chunks = ingest_text_blob(TEXT_BLOB, src="test_blob")
#     print(f"✅ Text blob ingested, chunks stored: {len(text_chunks)}")
# except Exception as e:
#     print(f"❌ Error during text blob ingestion: {e}")

# # --------------------------
# # Test RAG Agent Query
# # --------------------------
# QUERY = "What does the document talk about?"

# print("\n=== TEST: Querying Agent ===")
# try:
#     result = agentic_rag_answer.invoke({"question": QUERY})
#     answer = result.get("answer", "")
#     if answer:
#         print(f"✅ ANSWER: {answer}")
#     else:
#         print("⚠️ No answer returned by the agent.")
# except Exception as e:
#     print(f"❌ Error querying agent: {e}")

# --------------------------
# Test Web URL Ingestion
# --------------------------
WEB_URL = "https://www.scrapethissite.com/pages/"
web_chunks = ingest_web(WEB_URL)
print(f"Web page ingested, chunks stored: {len(web_chunks)}")

# --------------------------
# Test YouTube URL Ingestion
# --------------------------
YT_URL = "https://www.youtube.com/watch?v=o_XVt5rdpFY"
yt_chunks = ingest_youtube(YT_URL)
print(f"YouTube video ingested, chunks stored: {len(yt_chunks)}")