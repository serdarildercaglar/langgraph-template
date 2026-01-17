"""
Embedding service with async batch processing.
Designed for production with rate limiting and batching.
"""

import asyncio
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import EmbeddingSettings, get_settings


class EmbeddingService:
    """
    Async embedding service with batch processing.
    Uses vLLM-deployed embedding model via OpenAI-compatible API.
    """
    
    def __init__(self, settings: EmbeddingSettings | None = None):
        self._settings = settings or get_settings().embedding
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent)
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy client initialization."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.base_url,
                headers={
                    "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client
    
    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts."""
        async with self._semaphore:
            response = await self.client.post(
                "/embeddings",
                json={
                    "model": self._settings.model_name,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()
            
            # Sort by index to ensure correct order
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embeddings]
    
    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        results = await self._embed_batch([text])
        return results[0]
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts with automatic batching.
        Respects batch_size and max_concurrent settings.
        """
        if not texts:
            return []
        
        batch_size = self._settings.batch_size
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process batches concurrently (limited by semaphore)
        tasks = [self._embed_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        return [emb for batch_result in results for emb in batch_result]
    
    async def embed_documents(
        self,
        documents: list[dict[str, Any]],
        text_key: str = "content",
    ) -> list[dict[str, Any]]:
        """
        Embed documents and add embeddings to them.
        
        Args:
            documents: List of document dicts
            text_key: Key containing text to embed
        
        Returns:
            Documents with 'embedding' field added
        """
        texts = [doc[text_key] for doc in documents]
        embeddings = await self.embed_batch(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc["embedding"] = embedding
        
        return documents
    
    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self._settings.dimension
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Module-level singleton
_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
