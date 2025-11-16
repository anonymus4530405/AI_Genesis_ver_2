# backend/test_pipeline_full.py

from services.rag.api_adapter import run_agentic_rag

tests = [
    "What is AI?",
    "Explain this video: https://www.youtube.com/watch?v=hl2IeK4Ogl0",
    "Summarize this page: https://en.wikipedia.org/wiki/Python_(programming_language)",
    "flashcards: Deep Learning",
    "quiz: Artificial Intelligence",
    
    # ✅ REAL PDF INGESTION TEST
    "Explain this PDF: https://arxiv.org/pdf/1706.03762.pdf",   # Transformers paper PDF

    "Explain Machine Learning vs Deep Learning"
]

print("\n==============================")
print("🔥 RUNNING FULL AGENTIC RAG PIPELINE TEST")
print("==============================\n")

for i, query in enumerate(tests, 1):
    print(f"\n----- TEST {i} -----")
    print("Query:", query)
    try:
        result = run_agentic_rag(query)
        print("\nResponse:\n", result.get("response"))
    except Exception as e:
        print("\n❌ ERROR:")
        print(e)
