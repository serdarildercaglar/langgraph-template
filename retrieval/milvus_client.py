"""
Async Milvus client wrapper.
Provides batch and async operations for vector storage.
"""

import asyncio
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from config.settings import MilvusSettings, get_settings


class AsyncMilvusClient:
    """
    Async wrapper for Milvus operations.
    Provides batch insert, search, and delete operations.
    """
    
    def __init__(self, settings: MilvusSettings | None = None):
        self._settings = settings or get_settings().milvus
        self._client: MilvusClient | None = None
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent)
    
    def _get_client(self) -> MilvusClient:
        """Get or create Milvus client."""
        if self._client is None:
            self._client = MilvusClient(
                uri=self._settings.uri,
                user=self._settings.user if self._settings.user else None,
                password=self._settings.password.get_secret_value() if self._settings.password.get_secret_value() else None,
                db_name=self._settings.database,
                timeout=self._settings.timeout,
            )
        return self._client
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """Run blocking Milvus operations in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "COSINE",
    ) -> None:
        """
        Create a collection with standard schema.
        
        Schema:
        - id: VARCHAR (primary key)
        - embedding: FLOAT_VECTOR
        - content: VARCHAR
        - metadata: JSON
        """
        async with self._semaphore:
            client = self._get_client()
            
            # Check if exists
            has = await self._run_in_executor(
                client.has_collection,
                collection_name=collection_name,
            )
            
            if has:
                return
            
            # Create collection
            await self._run_in_executor(
                client.create_collection,
                collection_name=collection_name,
                dimension=dimension,
                metric_type=metric_type,
                auto_id=False,
                id_type="string",
                max_length=256,
            )
    
    async def insert(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Insert documents into collection.
        
        Args:
            collection_name: Target collection
            documents: List of dicts with 'id', 'embedding', 'content', 'metadata'
        
        Returns:
            Insert result with counts
        """
        async with self._semaphore:
            client = self._get_client()
            
            result = await self._run_in_executor(
                client.insert,
                collection_name=collection_name,
                data=documents,
            )
            
            return {"insert_count": result.get("insert_count", len(documents))}
    
    async def insert_batch(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Insert documents in batches.
        
        Args:
            collection_name: Target collection
            documents: List of documents
        
        Returns:
            Aggregate insert result
        """
        batch_size = self._settings.batch_size
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        total_inserted = 0
        for batch in batches:
            result = await self.insert(collection_name, batch)
            total_inserted += result["insert_count"]
        
        return {"insert_count": total_inserted}
    
    async def search(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Collection to search
            query_vectors: Query embeddings
            top_k: Number of results per query
            filter_expr: Milvus filter expression
            output_fields: Fields to return
        
        Returns:
            List of results per query vector
        """
        async with self._semaphore:
            client = self._get_client()
            
            results = await self._run_in_executor(
                client.search,
                collection_name=collection_name,
                data=query_vectors,
                limit=top_k,
                filter=filter_expr,
                output_fields=output_fields or ["content", "metadata"],
            )
            
            # Format results
            formatted = []
            for query_result in results:
                query_hits = []
                for hit in query_result:
                    query_hits.append({
                        "id": hit["id"],
                        "score": hit["distance"],
                        "content": hit.get("entity", {}).get("content", ""),
                        "metadata": hit.get("entity", {}).get("metadata", {}),
                    })
                formatted.append(query_hits)
            
            return formatted
    
    async def search_batch(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        top_k: int = 10,
        **kwargs,
    ) -> list[list[dict[str, Any]]]:
        """
        Search with batching for large query sets.
        """
        batch_size = self._settings.batch_size
        batches = [query_vectors[i:i + batch_size] for i in range(0, len(query_vectors), batch_size)]
        
        tasks = [
            self.search(collection_name, batch, top_k, **kwargs)
            for batch in batches
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        return [r for batch_result in results for r in batch_result]
    
    async def delete(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> dict[str, Any]:
        """
        Delete documents by IDs or filter.
        
        Args:
            collection_name: Target collection
            ids: List of document IDs to delete
            filter_expr: Filter expression for deletion
        
        Returns:
            Delete result
        """
        async with self._semaphore:
            client = self._get_client()
            
            result = await self._run_in_executor(
                client.delete,
                collection_name=collection_name,
                ids=ids,
                filter=filter_expr,
            )
            
            return {"delete_count": result}
    
    async def close(self) -> None:
        """Close the Milvus client."""
        if self._client:
            self._client.close()
            self._client = None


# Module-level singleton
_client: AsyncMilvusClient | None = None


def get_milvus_client() -> AsyncMilvusClient:
    """Get or create the Milvus client singleton."""
    global _client
    if _client is None:
        _client = AsyncMilvusClient()
    return _client
