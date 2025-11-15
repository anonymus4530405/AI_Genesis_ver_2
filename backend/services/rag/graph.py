from langgraph.graph import StateGraph, START, END
from services.rag.vectorstore import get_vectorstore
from services.rag.llm import groq_llm

class RAGState(dict):
    question: str
    retrieved_docs: list[str]
    answer: str

vs = get_vectorstore()

# --------------------------
# Retrieve node: get top-k documents
# --------------------------
def retrieve_node(state: RAGState):
    docs = vs.query(state["question"], k=3)  # returns list of texts
    state["retrieved_docs"] = docs
    return state

# --------------------------
# Generate node: LLM answer
# --------------------------
def generate_node(state: RAGState):
    ctx = "\n\n".join(state.get("retrieved_docs", []))
    prompt = f"Use ONLY this context:\n\n{ctx}\n\nQuestion: {state['question']}\nAnswer:"
    state["answer"] = groq_llm(prompt)  # make sure groq_llm returns string
    return state

# --------------------------
# Build the graph
# --------------------------
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

agentic_rag_answer = graph.compile()
