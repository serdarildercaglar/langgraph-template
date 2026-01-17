"""
Async Milvus client with multi-vector and hybrid search support.
Uses pymilvus SDK v2 native async.
"""

import asyncio
from typing import Any, Literal

from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient as MilvusAsyncClient,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    RRFRanker,
    WeightedRanker,
)

from config.settings import MilvusSettings, get_settings


class AsyncMilvusClient:
    """Async Milvus client with multi-vector and hybrid search."""

    def __init__(self, settings: MilvusSettings | None = None):
        self._settings = settings or get_settings().milvus
        self._client: MilvusAsyncClient | None = None
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent)

    async def _get_client(self) -> MilvusAsyncClient:
        if self._client is None:
            self._client = MilvusAsyncClient(
                uri=self._settings.uri,
                user=self._settings.user or None,
                password=self._settings.password.get_secret_value() or None,
                db_name=self._settings.database,
                timeout=self._settings.timeout,
            )
        return self._client

    async def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Create collection with custom schema.

        Args:
            collection_name: Collection name
            schema: Collection schema with fields
            index_params: Index params per vector field
        """
        async with self._semaphore:
            client = await self._get_client()
            if await client.has_collection(collection_name):
                return
            await client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

    async def insert(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Insert documents."""
        async with self._semaphore:
            client = await self._get_client()
            result = await client.insert(collection_name=collection_name, data=documents)
            return {"insert_count": result.get("insert_count", len(documents))}

    async def insert_batch(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Insert documents in parallel batches."""
        if not documents:
            return {"insert_count": 0}

        batch_size = self._settings.batch_size
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        results = await asyncio.gather(*[self.insert(collection_name, b) for b in batches])
        return {"insert_count": sum(r["insert_count"] for r in results)}

    def _format_hits(self, hits: list) -> list[dict[str, Any]]:
        """Format search hits to consistent structure."""
        return [
            {
                "id": hit["id"],
                "score": hit["distance"],
                "content": hit.get("entity", {}).get("content", ""),
                "metadata": hit.get("entity", {}).get("metadata", {}),
            }
            for hit in hits
        ]

    async def search(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        anns_field: str,
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Vector search on a specific field.

        Args:
            collection_name: Collection to search
            query_vectors: Query embeddings
            anns_field: Vector field name to search
            top_k: Number of results per query
            filter_expr: Pre-filter expression
            output_fields: Fields to return
        """
        async with self._semaphore:
            client = await self._get_client()
            results = await client.search(
                collection_name=collection_name,
                data=query_vectors,
                anns_field=anns_field,
                limit=top_k,
                filter=filter_expr,
                output_fields=output_fields or ["content", "metadata"],
            )
            return [self._format_hits(query_result) for query_result in results]

    async def hybrid_search(
        self,
        collection_name: str,
        requests: list[dict[str, Any]],
        top_k: int = 10,
        filter_expr: str | None = None,
        rerank: Literal["rrf", "weighted"] = "rrf",
        weights: list[float] | None = None,
        rrf_k: int = 60,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search across multiple vector fields with reranking.

        Args:
            collection_name: Collection to search
            requests: List of dicts with 'data' (vectors) and 'anns_field' (field name)
            top_k: Number of final results
            filter_expr: Pre-filter expression
            rerank: "rrf" or "weighted"
            weights: Weights for weighted rerank (must match requests length)
            rrf_k: K parameter for RRF (default 60)
            output_fields: Fields to return
        """
        async with self._semaphore:
            client = await self._get_client()

            ann_requests = [
                AnnSearchRequest(
                    data=req["data"],
                    anns_field=req["anns_field"],
                    param=req.get("params", {"metric_type": "COSINE"}),
                    limit=top_k * 2,
                )
                for req in requests
            ]

            if rerank == "weighted":
                w = weights or [1.0 / len(requests)] * len(requests)
                ranker = WeightedRanker(*w)
            else:
                ranker = RRFRanker(k=rrf_k)

            results = await client.hybrid_search(
                collection_name=collection_name,
                reqs=ann_requests,
                ranker=ranker,
                limit=top_k,
                filter=filter_expr,
                output_fields=output_fields or ["content", "metadata"],
            )

            return self._format_hits(results)

    async def delete(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> dict[str, Any]:
        """Delete by IDs or filter."""
        async with self._semaphore:
            client = await self._get_client()
            result = await client.delete(
                collection_name=collection_name,
                ids=ids,
                filter=filter_expr,
            )
            return {"delete_count": result}

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None


def build_schema(
    dense_fields: list[dict[str, Any]],
    fts_fields: list[dict[str, Any]] | None = None,
    partition_key: str | None = None,
    enable_dynamic: bool = True,
) -> tuple[CollectionSchema, dict[str, Any]]:
    """
    Build collection schema with dense vectors and FTS support.

    Args:
        dense_fields: Dense vector configs:
            - name: Vector field name
            - source: Source text field
            - dim: Dimension
            - metric: "COSINE", "L2", "IP" (default: COSINE)
        fts_fields: Full-text search configs (BM25):
            - name: Sparse vector field name
            - source: Source text field
            - analyzer: "standard", "english", "chinese" (default: standard)
        partition_key: Field for partition key
        enable_dynamic: Allow dynamic fields

    Returns:
        Tuple of (schema, index_params)
    """
    fts_fields = fts_fields or []

    # Collect unique source fields
    source_fields = set()
    for df in dense_fields:
        source_fields.add(df["source"])
    for ff in fts_fields:
        source_fields.add(ff["source"])

    # Base fields
    fields = [
        FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=256),
    ]

    # Source text fields (with analyzer for FTS)
    fts_sources = {ff["source"]: ff.get("analyzer", "standard") for ff in fts_fields}
    for source in source_fields:
        if source in fts_sources:
            fields.append(FieldSchema(
                source,
                DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params={"type": fts_sources[source]},
            ))
        else:
            fields.append(FieldSchema(source, DataType.VARCHAR, max_length=65535))

    # Partition key
    if partition_key:
        fields.append(FieldSchema(
            partition_key,
            DataType.VARCHAR,
            max_length=256,
            is_partition_key=True,
        ))

    # Dense vector fields
    index_params = {}
    for df in dense_fields:
        name = df["name"]
        dim = df["dim"]
        metric = df.get("metric", "COSINE")

        fields.append(FieldSchema(name, DataType.FLOAT_VECTOR, dim=dim))
        index_params[name] = {
            "index_type": "HNSW",
            "metric_type": metric,
            "params": {"M": 16, "efConstruction": 256},
        }

    # Sparse vector fields (FTS/BM25)
    functions = []
    for ff in fts_fields:
        name = ff["name"]
        source = ff["source"]

        fields.append(FieldSchema(name, DataType.SPARSE_FLOAT_VECTOR))
        index_params[name] = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
        }

        functions.append(Function(
            name=f"{name}_bm25",
            input_field_names=[source],
            output_field_names=[name],
            function_type=FunctionType.BM25,
        ))

    schema = CollectionSchema(fields, enable_dynamic_field=enable_dynamic)
    for func in functions:
        schema.add_function(func)

    return schema, index_params


_client: AsyncMilvusClient | None = None


def get_milvus_client() -> AsyncMilvusClient:
    """Get or create singleton client."""
    global _client
    if _client is None:
        _client = AsyncMilvusClient()
    return _client
