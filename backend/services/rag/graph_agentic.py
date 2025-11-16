# services/rag/graph_agentic.py
"""
Agentic RAG Graph:
- Multi-step decisions
- Automatic ingestion for missing knowledge
- Intent-based generation: answer, summary, flashcards, quiz
- Chunking, error handling, memory-aware
"""

from typing import Dict, Any, List
from services.rag.vectorstore import VectorStore
from services.rag.llm import groq_llm
from services.rag.validators import is_low_context, detect_user_intent
from services.rag.ingest_orchestrator import IngestOrchestrator
from services.rag.memory import MemoryManager

CHUNK_SIZE = 800  # characters per chunk for ingestion

class AgenticRAGState:
    def __init__(self, user_message: str):
        self.query = user_message
        self.intent = None
        self.retrieved_chunks: List[Dict] = []
        self.final_answer: str = ""
        self.new_ingestion_done = False
        self.extra_ingest_info: Dict = {}

class AgenticRAGGraph:
    def __init__(self):
        self.vector_db = VectorStore()
        self.ingestor = IngestOrchestrator()
        self.memory = MemoryManager()

    # ---------- STEP 1: Detect Intent ----------
    def step_detect_intent(self, state: AgenticRAGState):
        state.intent = detect_user_intent(state.query)
        return state

    # ---------- STEP 2: Retrieve ----------
    def step_retrieve(self, state: AgenticRAGState):
        chunks = self.vector_db.query(state.query, k=5)
        state.retrieved_chunks = [c for c in chunks if c.get("text", "").strip()]
        return state

    # ---------- STEP 3: Decide on Ingestion ----------
    def step_check_and_ingest(self, state: AgenticRAGState):
        if not is_low_context(state.retrieved_chunks):
            return state  # sufficient context

        try:
            ingest_result = self.ingestor.auto_ingest_if_needed(state.query)
            if ingest_result:
                state.new_ingestion_done = True
                state.extra_ingest_info = ingest_result
        except Exception as e:
            print("Ingestion failed:", e)

        return state

    # ---------- STEP 4: Re-retrieve if needed ----------
    def step_reretrieve_if_needed(self, state: AgenticRAGState):
        if state.new_ingestion_done:
            chunks = self.vector_db.query(state.query, k=5)
            state.retrieved_chunks = [c for c in chunks if c.get("text", "").strip()]
        return state

    # ---------- STEP 5: Generate Answer ----------
    def step_generate_answer(self, state: AgenticRAGState):
        context_text = "\n\n".join([c.get("text", "") for c in state.retrieved_chunks])

        if not context_text.strip():
            state.final_answer = "I could not find relevant information. Consider uploading a PDF, link, or checking online."
            return state

        if state.intent == "summarize":
            prompt = f"Summarize the following information clearly:\n\n{context_text}"
        elif state.intent == "flashcards":
            prompt = f"Generate 10 study flashcards in Q&A format:\n\n{context_text}"
        elif state.intent == "quiz":
            prompt = f"Generate a 10-question multiple-choice quiz (4 options each) from this content:\n\n{context_text}"
        else:
            prompt = (
                f"You are an expert assistant. Use the context below to answer the question.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {state.query}\nAnswer clearly:"
            )

        try:
            state.final_answer = groq_llm(prompt)
        except Exception as e:
            state.final_answer = f"LLM generation failed: {e}"

        return state

    # ---------- STEP 6: Update Memory ----------
    def step_update_memory(self, state: AgenticRAGState):
        if state.extra_ingest_info:
            topic = state.query.lower().strip()[:60]
            self.memory.add_topic_source(topic, state.extra_ingest_info.get("url", "unknown"))
        return state

    # ---------- FULL EXECUTION ----------
    def run(self, user_message: str) -> Dict[str, Any]:
        state = AgenticRAGState(user_message)

        state = self.step_detect_intent(state)
        state = self.step_retrieve(state)
        state = self.step_check_and_ingest(state)
        state = self.step_reretrieve_if_needed(state)
        state = self.step_generate_answer(state)
        state = self.step_update_memory(state)

        return {
            "response": state.final_answer,
            "intent": state.intent,
            "new_ingestion": state.new_ingestion_done,
            "meta": state.extra_ingest_info,
            "retrieved_chunks": state.retrieved_chunks  # Only non-empty chunks
        }
