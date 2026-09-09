"""Semantic (nearest-neighbor) search over the real historical incident log — same FAISS
index as the RAG retriever (rag/langchain_retriever.py), queried directly for standalone use
(e.g. from the MCP server) rather than through a LangChain retriever wrapper."""
from rag.langchain_retriever import get_vectorstore


def find_similar_incidents(farm_id: str, query: str, top_k: int = 5) -> list[dict]:
    store = get_vectorstore(farm_id)
    results = store.similarity_search_with_score(query, k=top_k)
    return [{"text": doc.page_content, "source": doc.metadata.get("source"), "distance": float(score)} for doc, score in results]
