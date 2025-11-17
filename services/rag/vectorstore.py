# services/rag/vectorstore.py
import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from services.rag.embeddings import get_embeddings

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

_vectorstore_instance = None


def get_vectorstore():
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStore()
    return _vectorstore_instance


class VectorStore:
    def __init__(self):
        if not QDRANT_URL or not QDRANT_COLLECTION:
            raise ValueError("Missing Qdrant configuration")

        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = QDRANT_COLLECTION
        self.embedder = get_embeddings()

        # Ensure collection exists
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vector_size=self.embedder.dimension,
                distance="Cosine"
            )

    def add_documents(self, docs: list[str], ids: list = None, payloads: list[dict] = None):
        if not isinstance(docs, list):
            docs = [docs]
        vectors = self.embedder.embed_documents(docs)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in docs]
        if payloads is None:
            payloads = [{"text": d} for d in docs]

        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(docs))]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, text: str, k: int = 5, metadata_filter: dict = None):
        vector = self.embedder.embed_query(text)
        q_filter = None
        if metadata_filter:
            conditions = [FieldCondition(key=key, match=MatchValue(value=value)) for key, value in metadata_filter.items()]
            q_filter = Filter(must=conditions)

        # Run the search
        result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=k,
            query_filter=q_filter
        )

        # Only include results that have actual payload
        chunks = [hit.payload for hit in result if hit.payload is not None]
        return chunks


    def delete_by_source(self, source_type: str):
        self.client.delete(
            collection_name=self.collection_name,
            filter=Filter(must=[FieldCondition(key="source_type", match=MatchValue(value=source_type))])
        )
