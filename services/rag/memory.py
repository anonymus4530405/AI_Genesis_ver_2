# services/rag/memory.py
import os
from typing import List, Optional, Dict
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE_NAME = "memory_store"  # your Supabase table with jsonb column

class MemoryManager:
    def __init__(self):
        self.data: Dict = self._load()

    def _load(self) -> Dict:
        res = supabase.table(TABLE_NAME).select("data").execute()
        if res.data and len(res.data) > 0:
            return res.data[0]["data"]
        else:
            # Initialize if empty
            return {"sources": {}, "topics": {}, "summaries": {}}

    def save(self):
        # Upsert the single row (id=1)
        supabase.table(TABLE_NAME).upsert({"id": 1, "data": self.data}).execute()

    def has_source(self, url: str) -> bool:
        return url in self.data.get("sources", {})

    def register_source(self, url: str, source_type: str, title: Optional[str]):
        if url not in self.data["sources"]:
            self.data["sources"][url] = {"type": source_type, "title": title or ""}
            self.save()

    def add_topic_source(self, topic: str, url: str):
        if topic not in self.data["topics"]:
            self.data["topics"][topic] = []
        if url not in self.data["topics"][topic]:
            self.data["topics"][topic].append(url)
            self.save()

    def get_topic_sources(self, topic: str) -> List[str]:
        return self.data.get("topics", {}).get(topic, [])

    def save_summary(self, url: str, summary: str):
        self.data["summaries"][url] = summary
        self.save()

    def get_summary(self, url: str) -> Optional[str]:
        return self.data.get("summaries", {}).get(url)
