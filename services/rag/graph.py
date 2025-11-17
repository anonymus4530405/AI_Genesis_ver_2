from fastapi import HTTPException
from services.rag.vectorstore import get_vectorstore
from services.rag.utils import build_prompt
from services.rag.llm import LLM
from services.rag.ingest_web import ingest_web
from services.rag.ingest_youtube import ingest_youtube

vs = get_vectorstore()
llm = LLM()


def retrieve_node(state):
    query = state["query"]
    docs = vs.query(query, k=4)
    state["retrieved_docs"] = docs
    return state


def decide_node(state):
    docs = state["retrieved_docs"]
    if len(docs) == 0:
        state["decision"] = "search_web"
    else:
        state["decision"] = "answer"
    return state


def search_web_node(state):
    query = state["query"]
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    ingest_web(url)
    state["from_search"] = True
    return state


def generate_answer_node(state):
    prompt = build_prompt(state["query"], state.get("retrieved_docs", []))
    state["answer"] = llm.generate(prompt)
    return state


def agentic_rag(query: str):
    state = {"query": query}

    state = retrieve_node(state)
    state = decide_node(state)

    if state["decision"] == "search_web":
        state = search_web_node(state)
        state = retrieve_node(state)

    state = generate_answer_node(state)
    return state["answer"]
