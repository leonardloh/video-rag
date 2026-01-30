"""Context manager for RAG operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .context_store import (
    ContextStore,
    ContextWindow,
    Document,
    InMemoryContextStore,
)

if TYPE_CHECKING:
    from ..models.gemini.gemini_embeddings import GeminiEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class ContextManagerConfig:
    """Configuration for ContextManager."""

    max_context_tokens: int = 100000
    generate_embeddings: bool = True


class ContextManager:
    """Manages video context for RAG operations."""

    def __init__(
        self,
        embeddings: GeminiEmbeddings,
        store: Optional[ContextStore] = None,
        max_context_tokens: int = 100000,
    ) -> None:
        """
        Initialize context manager.

        Args:
            embeddings: Embedding generator
            store: Context store (defaults to in-memory)
            max_context_tokens: Maximum tokens in context window
        """
        self._embeddings = embeddings
        self._store = store if store is not None else InMemoryContextStore()
        self._max_context_tokens = max_context_tokens

        # Function registry for CA-RAG
        self._functions: dict[str, Any] = {}

    async def add_document(
        self,
        text: str,
        metadata: dict[str, Any],
        generate_embedding: bool = True,
    ) -> str:
        """
        Add a document to the context.

        Args:
            text: Document text (e.g., caption)
            metadata: Metadata (chunk_idx, timestamps, etc.)
            generate_embedding: Whether to generate embedding

        Returns:
            Document ID
        """
        embedding = None
        if generate_embedding:
            try:
                result = await self._embeddings.embed_document(text)
                embedding = result
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        return await self._store.add_document(
            text=text,
            metadata=metadata,
            embedding=embedding,
        )

    async def add_documents_batch(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        generate_embeddings: bool = True,
    ) -> list[str]:
        """
        Add multiple documents in batch.

        Args:
            texts: List of document texts
            metadatas: List of metadata dicts
            generate_embeddings: Whether to generate embeddings

        Returns:
            List of document IDs
        """
        embeddings: Optional[list[list[float]]] = None
        if generate_embeddings:
            try:
                embeddings = await self._embeddings.embed_documents(texts)
            except Exception as e:
                logger.warning(f"Failed to generate embeddings: {e}")

        doc_ids = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            embedding = embeddings[i] if embeddings else None
            doc_id = await self._store.add_document(
                text=text,
                metadata=metadata,
                embedding=embedding,
            )
            doc_ids.append(doc_id)

        return doc_ids

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            List of documents with scores
        """
        # Generate query embedding
        try:
            query_embedding = await self._embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []

        # Search
        results = await self._store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        return [
            {
                "id": doc.id,
                "text": doc.text,
                "metadata": doc.metadata,
                "score": score,
            }
            for doc, score in results
        ]

    async def get_context_window(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        include_all: bool = False,
    ) -> ContextWindow:
        """
        Get a context window for generation.

        Args:
            query: Query to retrieve relevant context
            max_tokens: Maximum tokens (uses default if None)
            include_all: Include all documents (ignore query)

        Returns:
            ContextWindow with documents
        """
        max_tokens = max_tokens or self._max_context_tokens

        if include_all:
            docs = await self._store.get_all_documents()
        else:
            results = await self.retrieve(query, top_k=100)
            docs = [
                Document(
                    id=r["id"],
                    text=r["text"],
                    metadata=r["metadata"],
                )
                for r in results
            ]

        # Build context window within token limit
        selected_docs: list[Document] = []
        total_tokens = 0
        truncated = False

        for doc in docs:
            doc_tokens = self._estimate_tokens(doc.text)
            if total_tokens + doc_tokens > max_tokens:
                truncated = True
                break
            selected_docs.append(doc)
            total_tokens += doc_tokens

        return ContextWindow(
            documents=selected_docs,
            total_tokens=total_tokens,
            truncated=truncated,
        )

    async def get_all_captions(
        self,
        stream_id: Optional[str] = None,
    ) -> list[str]:
        """
        Get all captions, optionally filtered by stream.

        Args:
            stream_id: Optional stream filter

        Returns:
            List of caption texts
        """
        filter_metadata = {"stream_id": stream_id} if stream_id else None
        docs = await self._store.get_all_documents(filter_metadata)
        return [doc.text for doc in docs]

    async def get_captions_by_time_range(
        self,
        start_time: str,
        end_time: str,
        stream_id: Optional[str] = None,
    ) -> list[str]:
        """
        Get captions within a time range.

        Args:
            start_time: Start timestamp (HH:MM:SS)
            end_time: End timestamp (HH:MM:SS)
            stream_id: Optional stream filter

        Returns:
            List of caption texts
        """
        filter_metadata = {"stream_id": stream_id} if stream_id else None
        docs = await self._store.get_all_documents(filter_metadata)

        # Filter by time range
        filtered = []
        for doc in docs:
            doc_start = doc.metadata.get("start_time", "")
            doc_end = doc.metadata.get("end_time", "")
            if doc_start >= start_time and doc_end <= end_time:
                filtered.append(doc.text)

        return filtered

    async def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """
        Get a document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document dict if found
        """
        doc = await self._store.get_document(doc_id)
        if doc is None:
            return None

        return {
            "id": doc.id,
            "text": doc.text,
            "metadata": doc.metadata,
        }

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.

        Args:
            doc_id: Document identifier

        Returns:
            True if deleted
        """
        return await self._store.delete_document(doc_id)

    async def clear_context(self, stream_id: Optional[str] = None) -> int:
        """
        Clear context, optionally for a specific stream.

        Args:
            stream_id: Optional stream to clear

        Returns:
            Count of documents deleted
        """
        if stream_id:
            # Delete documents for specific stream
            docs = await self._store.get_all_documents({"stream_id": stream_id})
            count = 0
            for doc in docs:
                if await self._store.delete_document(doc.id):
                    count += 1
            return count
        else:
            return await self._store.clear()

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses rough approximation of ~4 characters per token.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    # Function registry methods for CA-RAG

    def register_function(self, name: str, function: Any) -> None:
        """
        Register a RAG function.

        Args:
            name: Function name
            function: Function instance
        """
        self._functions[name] = function

    def get_function(self, name: str) -> Optional[Any]:
        """
        Get a registered function.

        Args:
            name: Function name

        Returns:
            Function instance if found
        """
        return self._functions.get(name)

    async def call(self, function_name: str, **kwargs: Any) -> Any:
        """
        Call a registered function.

        Args:
            function_name: Name of function to call
            **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            ValueError: If function not found
        """
        func = self._functions.get(function_name)
        if func is None:
            raise ValueError(f"Function '{function_name}' not found")

        if hasattr(func, "execute"):
            return await func.execute(**kwargs)  # type: ignore[misc]
        elif callable(func):
            return await func(**kwargs)  # type: ignore[misc]
        else:
            raise ValueError(f"Function '{function_name}' is not callable")

    def list_functions(self) -> list[str]:
        """
        List registered function names.

        Returns:
            List of function names
        """
        return list(self._functions.keys())

    @property
    def store(self) -> ContextStore:
        """Get the underlying context store."""
        return self._store

    @property
    def max_context_tokens(self) -> int:
        """Get maximum context tokens."""
        return self._max_context_tokens
