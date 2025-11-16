# services/rag/validators.py

"""
Validators used by the Agentic Graph to decide next actions.
"""

from typing import List, Dict

LOW_CONFIDENCE_THRESHOLD = 0.35  # tune later


def is_low_context(retrieved_chunks: List[Dict]) -> bool:
    """
    Determines if retrieved context is too weak to answer.
    """
    if not retrieved_chunks:
        return True

    avg_score = sum(c.get("score", 0) for c in retrieved_chunks) / len(retrieved_chunks)
    return avg_score < LOW_CONFIDENCE_THRESHOLD


def detect_user_intent(user_message: str) -> str:
    """
    Detect tasks:
      - 'answer'
      - 'summarize'
      - 'flashcards'
      - 'quiz'
    """
    msg = user_message.lower()

    if "summary" in msg or "summarize" in msg:
        return "summarize"

    if "flashcard" in msg or "anki" in msg or "study cards" in msg:
        return "flashcards"
    
    if "quiz" in msg or "test me" in msg or "mcqs" in msg:
        return "quiz"

    return "answer"
