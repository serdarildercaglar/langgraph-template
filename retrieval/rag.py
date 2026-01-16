"""
RAG (Retrieval Augmented Generation) pipeline.
Integrates embedding service and Milvus for document retrieval.
"""

import uuid
from typing import Any

from retrieval.embeddings import EmbeddingService, get_embedding_service
from retrieval.milvus_client import AsyncMilvusClient, get_milvus_client


class RAGPipeline:
    """
    RAG pipeline for document ingestion and retrieval.
    Provides a high-level interface for RAG operations.
    """
    
    def __init__(
        self,
        collection_name: str | None = None,
        embedding_service: EmbeddingService | None = None,
        milvus_client: AsyncMilvusClient | None = None,
    ):
        from config.settings import get_settings
        
        settings = get_settings()
        self.collection_name = collection_name or settings.milvus.collection_name
        self._embedding_service = embedding_service
        self._milvus_client = milvus_client
    
    @property
    def embeddings(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service
    
    @property
    def milvus(self) -> AsyncMilvusClient:
        if self._milvus_client is None:
            self._milvus_client = get_milvus_client()
        return self._milvus_client
    
    async def initialize(self) -> None:
        """
        Initialize the RAG pipeline.
        Creates collection if it doesn't exist.
        """
        await self.milvus.create_collection(
            collection_name=self.collection_name,
            dimension=self.embeddings.dimension,
        )
    
    async def ingest_documents(
        self,
        documents: list[dict[str, Any]],
        id_field: str = "id",
        content_field: str = "content",
        metadata_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest documents into the RAG system.
        
        Args:
            documents: List of document dicts
            id_field: Field containing document ID (auto-generated if missing)
            content_field: Field containing text content
            metadata_fields: Fields to include in metadata
        
        Returns:
            Ingestion result with counts
        """
        # Prepare documents
        prepared = []
        for doc in documents:
            doc_id = doc.get(id_field) or str(uuid.uuid4())
            content = doc[content_field]
            
            # Extract metadata
            metadata = {}
            if metadata_fields:
                metadata = {k: doc.get(k) for k in metadata_fields if k in doc}
            
            prepared.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
            })
        
        # Generate embeddings
        texts = [d["content"] for d in prepared]
        embeddings = await self.embeddings.embed_batch(texts)
        
        # Add embeddings to documents
        for doc, embedding in zip(prepared, embeddings):
            doc["vector"] = embedding
        
        # Insert into Milvus
        result = await self.milvus.insert_batch(
            collection_name=self.collection_name,
            documents=prepared,
        )
        
        return result
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_expr: Milvus filter expression
            min_score: Minimum similarity score threshold
        
        Returns:
            List of relevant documents with scores
        """
        # Embed query
        query_embedding = await self.embeddings.embed(query)
        
        # Search Milvus
        results = await self.milvus.search(
            collection_name=self.collection_name,
            query_vectors=[query_embedding],
            top_k=top_k,
            filter_expr=filter_expr,
        )
        
        # Get first query's results
        docs = results[0] if results else []
        
        # Apply minimum score filter
        if min_score is not None:
            docs = [d for d in docs if d["score"] >= min_score]
        
        return docs
    
    async def retrieve_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
        
        Returns:
            List of results per query
        """
        # Embed all queries
        query_embeddings = await self.embeddings.embed_batch(queries)
        
        # Search Milvus
        results = await self.milvus.search_batch(
            collection_name=self.collection_name,
            query_vectors=query_embeddings,
            top_k=top_k,
            **kwargs,
        )
        
        return results
    
    def format_context(
        self,
        documents: list[dict[str, Any]],
        max_length: int | None = None,
    ) -> str:
        """
        Format retrieved documents as context string.
        
        Args:
            documents: Retrieved documents
            max_length: Maximum context length (chars)
        
        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            score = doc.get("score", 0)
            
            part = f"[Document {i}] (relevance: {score:.2f})\n{content}\n"
            
            if max_length and total_length + len(part) > max_length:
                break
            
            context_parts.append(part)
            total_length += len(part)
        
        return "\n".join(context_parts)
    
    async def delete_documents(
        self,
        ids: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> dict[str, Any]:
        """Delete documents by IDs or filter."""
        return await self.milvus.delete(
            collection_name=self.collection_name,
            ids=ids,
            filter_expr=filter_expr,
        )


# Module-level singleton
_pipeline: RAGPipeline | None = None


def get_rag_pipeline(collection_name: str | None = None) -> RAGPipeline:
    """Get or create the RAG pipeline."""
    global _pipeline
    if _pipeline is None or (collection_name and _pipeline.collection_name != collection_name):
        _pipeline = RAGPipeline(collection_name=collection_name)
    return _pipeline
