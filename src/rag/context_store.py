"""Context store abstractions for RAG document storage and retrieval."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np


@dataclass
class Document:
    """A document in the context store."""

    id: str
    text: str
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""

    documents: list[Document]
    scores: list[float]
    query: str
    top_k: int


@dataclass
class ContextWindow:
    """A window of context for generation."""

    documents: list[Document]
    total_tokens: int  # Estimated token count
    truncated: bool = False


class ContextStore(ABC):
    """Abstract base class for context storage."""

    @abstractmethod
    async def add_document(
        self,
        text: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> str:
        """
        Add a document to the store.

        Args:
            text: Document text
            metadata: Document metadata
            embedding: Optional pre-computed embedding

        Returns:
            Document ID
        """
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document if found, None otherwise
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            List of (document, score) tuples
        """
        pass

    @abstractmethod
    async def get_all_documents(
        self,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        """
        Get all documents, optionally filtered.

        Args:
            filter_metadata: Optional metadata filter

        Returns:
            List of documents
        """
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.

        Args:
            doc_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def clear(self) -> int:
        """
        Clear all documents.

        Returns:
            Count of documents deleted
        """
        pass


class InMemoryContextStore(ContextStore):
    """Simple in-memory context store with cosine similarity search."""

    def __init__(self) -> None:
        """Initialize empty store."""
        self._documents: dict[str, Document] = {}

    async def add_document(
        self,
        text: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> str:
        """Add a document to the store."""
        doc_id = str(uuid.uuid4())
        self._documents[doc_id] = Document(
            id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )
        return doc_id

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents using cosine similarity."""
        results: list[tuple[Document, float]] = []
        query_np = np.array(query_embedding)
        query_norm = np.linalg.norm(query_np)

        if query_norm == 0:
            return results

        for doc in self._documents.values():
            # Apply metadata filter
            if filter_metadata:
                if not all(
                    doc.metadata.get(k) == v for k, v in filter_metadata.items()
                ):
                    continue

            # Skip if no embedding
            if doc.embedding is None:
                continue

            # Calculate cosine similarity
            doc_np = np.array(doc.embedding)
            doc_norm = np.linalg.norm(doc_np)

            if doc_norm == 0:
                continue

            similarity = float(np.dot(query_np, doc_np) / (query_norm * doc_norm))
            results.append((doc, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    async def get_all_documents(
        self,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        """Get all documents, optionally filtered."""
        docs = list(self._documents.values())

        if filter_metadata:
            docs = [
                doc
                for doc in docs
                if all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
            ]

        # Sort by chunk_idx if available
        docs.sort(key=lambda d: d.metadata.get("chunk_idx", 0))

        return docs

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    async def clear(self) -> int:
        """Clear all documents."""
        count = len(self._documents)
        self._documents.clear()
        return count

    def __len__(self) -> int:
        """Return number of documents."""
        return len(self._documents)
