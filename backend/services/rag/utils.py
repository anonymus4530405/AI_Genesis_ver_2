def build_prompt(query: str, retrieved_docs: list[dict]) -> str:
    context = "\n\n".join(
        [f"Source: {d.get('source_type','unknown')}\n{d.get('text','')}" for d in retrieved_docs]
    )

    return f"""
Your job is to answer the user's question using the provided context.

USER QUESTION:
{query}

CONTEXT:
{context}

If the answer is not fully found in the context, say:
"I need to search more — should I fetch from web/Youtube/PDF?"
"""


# services/rag/utils.py

import re

def clean_text(t: str) -> str:
    """
    Basic text cleaning:
      - remove extra whitespace
      - normalize newlines
    """
    if not isinstance(t, str):
        return ""
    t = re.sub(r"\s+", " ", t)       # collapse multiple spaces
    t = t.strip()
    return t

