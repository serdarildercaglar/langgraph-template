"""
State management for LangGraph.

Note: For most use cases, you don't need custom state.
LangGraph's create_react_agent uses MessagesState internally.

This module exists for custom graph scenarios where you need
to extend the base state with additional fields.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Re-export MessagesState for convenience
from langgraph.graph import MessagesState

__all__ = ["MessagesState", "RAGState"]


class RAGState(TypedDict):
    """
    Extended state for RAG workflows.
    Use this when building custom graphs that need retrieval context.

    Example:
        from langgraph.graph import StateGraph
        from graph.state import RAGState

        builder = StateGraph(RAGState)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("generate", generate_node)
    """
    messages: Annotated[list[AnyMessage], add_messages]
    context: list[str]  # Retrieved documents
    query: str  # Original query for retrieval
