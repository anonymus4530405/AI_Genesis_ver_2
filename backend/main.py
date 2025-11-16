from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.rag_routes import router as rag_router

app = FastAPI(title="Agentic RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router, prefix="/rag")

@app.get("/")
def root():
    return {"message": "Agentic RAG backend running!"}
