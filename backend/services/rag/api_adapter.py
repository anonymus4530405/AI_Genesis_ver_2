# services/rag/api_adapter.py

from services.rag.graph_agentic import AgenticRAGGraph

agent_graph = AgenticRAGGraph()


def run_agentic_rag(query: str):
    return agent_graph.run(query)
