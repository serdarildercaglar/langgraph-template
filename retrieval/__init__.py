"""Retrieval module - Embeddings, Milvus, and RAG pipeline."""

from retrieval.embeddings import (
    EmbeddingService,
    embed_text,
    embed_texts,
    get_embedding_service,
)
from retrieval.milvus_client import (
    AsyncMilvusClient,
    get_milvus_client,
)
from retrieval.rag import (
    RAGPipeline,
    get_rag_pipeline,
)

__all__ = [
    # Embeddings
    "EmbeddingService",
    "embed_text",
    "embed_texts",
    "get_embedding_service",
    # Milvus
    "AsyncMilvusClient",
    "get_milvus_client",
    # RAG
    "RAGPipeline",
    "get_rag_pipeline",
]
