# Retrieval Module

Multi-vector ve hybrid search destekli Milvus RAG pipeline.

## Özellikler

- **Multi-vector**: Aynı document için birden fazla embedding (title, content, summary)
- **FTS/BM25**: Full-text search, keyword matching
- **Hybrid Search**: Dense + Sparse birlikte, RRF/Weighted reranking
- **Partition Key**: Kategori bazlı hızlı filtreleme
- **Batch Processing**: Çoklu query ve collection desteği

## Kurulum

```bash
pip install pymilvus>=2.5.0
```

## Hızlı Başlangıç

### 1. JSON Data Hazırla

```json
// data/collections.json
[
  {
    "collection": "docs",
    "partition_key": "category",
    "fields": {
      "dense": [
        {"name": "title_vector", "source": "title"},
        {"name": "content_vector", "source": "content"}
      ],
      "fts": [
        {"name": "content_sparse", "source": "content", "analyzer": "standard"}
      ]
    },
    "documents": [
      {
        "id": "doc1",
        "title": "API Authentication",
        "content": "OAuth2 and JWT token guide...",
        "category": "api"
      }
    ]
  }
]
```

### 2. Collection Oluştur ve Data Yükle

```python
from retrieval.rag import load_from_json

# JSON'dan collection oluştur ve document'ları yükle
pipelines = await load_from_json("data/collections.json")

# Pipeline'a eriş
docs_pipeline = pipelines["docs"]
```

### 3. Arama Yap

```python
# Dense search (semantic)
results = await docs_pipeline.search_dense(
    query="how to authenticate",
    vector_field="content_vector",
    top_k=5,
    filter_expr='category == "api"',
)

# FTS search (keyword/BM25)
results = await docs_pipeline.search_fts(
    query="OAuth JWT token",
    sparse_field="content_sparse",
    top_k=5,
)

# Hybrid search (dense + FTS)
results = await docs_pipeline.search_hybrid(
    query="authentication guide",
    dense_fields=["title_vector", "content_vector"],
    fts_fields=["content_sparse"],
    top_k=5,
    rerank="rrf",
)
```

---

## JSON Schema

### Zorunlu Alanlar

```json
{
  "collection": "string",           // Collection adı
  "fields": {
    "dense": [                      // En az 1 dense field
      {
        "name": "string",           // Vector field adı
        "source": "string"          // Kaynak text field
      }
    ]
  },
  "documents": []                   // Document listesi
}
```

### Opsiyonel Alanlar

```json
{
  "partition_key": "category",      // Partition key field
  "fields": {
    "dense": [
      {
        "dim": 1024,                // Default: embedding service dim
        "metric": "COSINE"          // COSINE, L2, IP
      }
    ],
    "fts": [                        // Full-text search
      {
        "name": "content_sparse",
        "source": "content",
        "analyzer": "standard"      // standard, english, chinese
      }
    ]
  }
}
```

### Document

```json
{
  "id": "doc1",                     // Opsiyonel, yoksa UUID
  "title": "...",                   // source field'lar
  "content": "...",
  "category": "api"                 // partition_key varsa
}
```

---

## Retrieve API

### Request Format

```python
from api.models import RetrieveRequest, SearchRequest

request = RetrieveRequest(requests=[
    SearchRequest(
        collection="docs",
        queries=["how to authenticate", "OAuth setup"],
        dense_fields=["content_vector"],
        fts_fields=["content_sparse"],
        top_k=5,
        filter="category == 'api'",
        rerank="rrf",
    )
])
```

### Response Format

```python
{
    "collections": [
        {
            "collection": "docs",
            "search_type": "hybrid",
            "results": [
                {
                    "query": "how to authenticate",
                    "hits": [
                        {"id": "doc1", "score": 0.92, "content": "...", "metadata": {}}
                    ]
                }
            ]
        }
    ],
    "total_hits": 10,
    "latency_ms": 45.2
}
```

---

## Kullanım Örnekleri

### Dense Search (Semantic)

```python
from retrieval.rag import RAGPipeline

pipeline = RAGPipeline(collection_name="docs")

# Tek query
results = await pipeline.search_dense(
    query="machine learning basics",
    vector_field="content_vector",
    top_k=5,
)

# Filtreli arama
results = await pipeline.search_dense(
    query="database optimization",
    vector_field="content_vector",
    filter_expr='category == "database" AND version >= 2',
    top_k=10,
)
```

### FTS Search (BM25/Keyword)

```python
# Keyword arama - embedding yok, raw text
results = await pipeline.search_fts(
    query="PostgreSQL index optimization",
    sparse_field="content_sparse",
    top_k=5,
)
```

### Hybrid Search

```python
# Dense + FTS, RRF reranking
results = await pipeline.search_hybrid(
    query="how to setup authentication",
    dense_fields=["title_vector", "content_vector"],
    fts_fields=["content_sparse"],
    top_k=10,
    rerank="rrf",
)

# Weighted reranking
results = await pipeline.search_hybrid(
    query="OAuth2 implementation",
    dense_fields=["content_vector"],
    fts_fields=["content_sparse"],
    rerank="weighted",
    weights=[0.7, 0.3],  # 70% dense, 30% FTS
)
```

### Multi-Collection Search

```python
import asyncio
from retrieval.rag import RAGPipeline

async def search_multiple_collections(query: str):
    docs = RAGPipeline("docs")
    faq = RAGPipeline("faq")

    # Parallel arama
    results = await asyncio.gather(
        docs.search_dense(query, "content_vector"),
        faq.search_dense(query, "question_vector"),
    )

    return {
        "docs": results[0],
        "faq": results[1],
    }
```

### Batch Query Search

```python
async def batch_search(queries: list[str]):
    pipeline = RAGPipeline("docs")

    # Her query için parallel embedding + search
    tasks = [
        pipeline.search_hybrid(
            query=q,
            dense_fields=["content_vector"],
            fts_fields=["content_sparse"],
        )
        for q in queries
    ]

    return await asyncio.gather(*tasks)

# Kullanım
results = await batch_search([
    "how to authenticate",
    "database setup",
    "API rate limiting",
])
```

---

## Tam Örnek: JSON'dan Retrieve'e

```python
import asyncio
import time
from retrieval.rag import load_from_json, RAGPipeline
from api.models import RetrieveRequest, SearchRequest, RetrieveResponse, CollectionResult, QueryResult, Hit


async def main():
    # 1. JSON'dan yükle
    print("Loading collections from JSON...")
    pipelines = await load_from_json("data/collections.json")
    print(f"Loaded {len(pipelines)} collections: {list(pipelines.keys())}")

    # 2. Search request
    request = RetrieveRequest(requests=[
        SearchRequest(
            collection="technical_docs",
            queries=["how to authenticate", "OAuth setup"],
            dense_fields=["title_vector", "content_vector"],
            fts_fields=["content_sparse"],
            top_k=5,
            filter='category == "api"',
            rerank="rrf",
        ),
        SearchRequest(
            collection="faq",
            queries=["reset password"],
            dense_fields=["question_vector"],
            top_k=3,
        ),
    ])

    # 3. Execute search
    start = time.perf_counter()

    collection_results = []
    total_hits = 0

    for req in request.requests:
        pipeline = pipelines.get(req.collection)
        if not pipeline:
            continue

        query_results = []
        for query in req.queries:
            # Search type belirle
            if req.dense_fields and req.fts_fields:
                hits = await pipeline.search_hybrid(
                    query=query,
                    dense_fields=req.dense_fields,
                    fts_fields=req.fts_fields,
                    top_k=req.top_k,
                    filter_expr=req.filter,
                    rerank=req.rerank,
                    weights=req.weights,
                )
                search_type = "hybrid"
            elif req.fts_fields:
                hits = await pipeline.search_fts(
                    query=query,
                    sparse_field=req.fts_fields[0],
                    top_k=req.top_k,
                    filter_expr=req.filter,
                )
                search_type = "fts"
            else:
                hits = await pipeline.search_dense(
                    query=query,
                    vector_field=req.dense_fields[0],
                    top_k=req.top_k,
                    filter_expr=req.filter,
                )
                search_type = "dense"

            query_results.append(QueryResult(
                query=query,
                hits=[Hit(**h) for h in hits],
            ))
            total_hits += len(hits)

        collection_results.append(CollectionResult(
            collection=req.collection,
            results=query_results,
            search_type=search_type,
        ))

    latency = (time.perf_counter() - start) * 1000

    # 4. Response
    response = RetrieveResponse(
        collections=collection_results,
        total_hits=total_hits,
        latency_ms=latency,
    )

    print(f"\nResults ({response.latency_ms:.1f}ms):")
    for coll in response.collections:
        print(f"\n[{coll.collection}] ({coll.search_type})")
        for qr in coll.results:
            print(f"  Query: {qr.query}")
            for hit in qr.hits[:3]:
                print(f"    - {hit.id}: {hit.score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Milvus Schema

JSON'dan oluşturulan collection schema:

```
Collection: technical_docs
├── id (VARCHAR, primary key)
├── title (VARCHAR)
├── content (VARCHAR, analyzer=standard)
├── category (VARCHAR, partition_key)
├── title_vector (FLOAT_VECTOR, dim=1024, HNSW)
├── content_vector (FLOAT_VECTOR, dim=1024, HNSW)
└── content_sparse (SPARSE_FLOAT_VECTOR, BM25)
```

## Filter Syntax

Milvus filter expression örnekleri:

```python
# Eşitlik
'category == "api"'

# Karşılaştırma
'version >= 2'
'price < 100'

# IN operatörü
'category IN ["api", "database"]'

# AND/OR
'category == "api" AND version >= 2'
'status == "active" OR priority > 5'

# LIKE (prefix search)
'title LIKE "Auth%"'

# Array contains
'ARRAY_CONTAINS(tags, "python")'
```
