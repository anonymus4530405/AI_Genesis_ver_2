# services/rag/embeddings.py
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Ensure your HF_API_KEY is set as HUGGINGFACEHUB_API_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_API_KEY", "")

def get_embeddings():
    """
    Returns a Hugging Face API embeddings object for use in LangChain/FAISS.
    """
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
