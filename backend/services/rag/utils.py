# services/rag/utils.py
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str, default=None):
    return os.getenv(key, default)

def clean_text(t: str):
    if not isinstance(t, str):
        return t
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def is_pdf(path: str | Path):
    return str(path).lower().endswith(".pdf")

def is_supported_file(path: str | Path):
    return str(path).lower().endswith((".pdf", ".txt", ".md"))
