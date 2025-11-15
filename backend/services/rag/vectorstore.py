# services/rag/vectorstore.py
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from services.rag.embeddings import get_embeddings
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

_vectorstore = None

def get_vectorstore():
    """
    Singleton getter for VectorStore
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore()
    return _vectorstore

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        self.collection_name = QDRANT_COLLECTION
        self.embedder = get_embeddings()

    # --------------------------
    # Add Documents to Qdrant
    # --------------------------
    def add_documents(self, docs: list[str], ids: list = None, payloads: list[dict] = None):
        """
        docs: list of document strings
        ids: optional list of IDs (int or UUID strings)
        payloads: optional list of payload dicts
        """
        # Generate embeddings
        vectors = self.embedder.embed_documents(docs)

        # Generate IDs if not provided
        if ids is None:
            ids = list(range(len(docs)))

        # Generate payloads if not provided
        if payloads is None:
            payloads = [{"text": doc} for doc in docs]

        # Validate lengths
        if not (len(docs) == len(vectors) == len(ids) == len(payloads)):
            raise ValueError("Length mismatch between docs, vectors, ids, payloads")

        # Create PointStructs
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i]
            )
            for i in range(len(docs))
        ]

        # Upsert into Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    # --------------------------
    # Query
    # --------------------------
    def query(self, text: str, k: int = 3):
        """
        Query Qdrant for most similar documents
        """
        vector = self.embedder.embed_query(text)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=k
        )

        return [hit.payload.get("text", "") for hit in search_result]
