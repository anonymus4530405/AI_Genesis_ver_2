# services/rag/embeddings.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Load .env file
load_dotenv()  

hf_key = os.getenv("HF_API_KEY")
if not hf_key:
    raise ValueError("HF_API_KEY not set!")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key

def get_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
