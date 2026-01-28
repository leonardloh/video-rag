# RAG Context Manager Specification

## Overview

The `ContextManager` manages video context for RAG (Retrieval-Augmented Generation) operations. It stores captions and metadata, supports vector-based retrieval, and provides context for summarization and chat.

## Gap Analysis

### Original Implementation
- `src/vss-engine/config/config.yaml` - Defines context_manager with functions
- Uses Neo4j graph DB for temporal relationships
- Uses Milvus vector DB for embeddings
- Complex CA-RAG (Context-Aware RAG) with graph traversal

### PoC Requirement
- Simplified context manager
- Optional Milvus integration (can run without)
- In-memory fallback for simple deployments
- Support batch summarization and retrieval

## Component Location

```
./src/rag/context_manager.py
```

## Dependencies

```python
# From requirements.txt
pymilvus>=2.3.0  # Optional
numpy>=1.24.0
```

## Data Classes

```python
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime


@dataclass
class Document:
    """A document in the context store."""
    id: str
    text: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
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
```

## Class Interface

```python
from typing import Optional, List
from abc import ABC, abstractmethod


class ContextStore(ABC):
    """Abstract base class for context storage."""

    @abstractmethod
    async def add_document(
        self,
        text: str,
        metadata: dict,
        embedding: Optional[list[float]] = None,
    ) -> str:
        """Add a document to the store."""
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def get_all_documents(
        self,
        filter_metadata: Optional[dict] = None,
    ) -> List[Document]:
        """Get all documents, optionally filtered."""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all documents. Returns count deleted."""
        pass


class ContextManager:
    """Manages video context for RAG operations."""

    def __init__(
        self,
        embeddings: "GeminiEmbeddings",
        store: Optional[ContextStore] = None,
        max_context_tokens: int = 100000,
    ):
        """
        Initialize context manager.

        Args:
            embeddings: Embedding generator
            store: Context store (defaults to in-memory)
            max_context_tokens: Maximum tokens in context window
        """
        pass

    async def add_document(
        self,
        text: str,
        metadata: dict,
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
        pass

    async def add_documents_batch(
        self,
        texts: list[str],
        metadatas: list[dict],
        generate_embeddings: bool = True,
    ) -> list[str]:
        """Add multiple documents in batch."""
        pass

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            List of documents with scores
        """
        pass

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
        pass

    async def get_all_captions(
        self,
        stream_id: Optional[str] = None,
    ) -> list[str]:
        """Get all captions, optionally filtered by stream."""
        pass

    async def get_captions_by_time_range(
        self,
        start_time: str,
        end_time: str,
    ) -> list[str]:
        """Get captions within a time range."""
        pass

    async def clear_context(self, stream_id: Optional[str] = None) -> int:
        """Clear context, optionally for a specific stream."""
        pass

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimate: ~4 characters per token
        return len(text) // 4
```

## In-Memory Store Implementation

```python
import uuid
from typing import Optional, List
import numpy as np


class InMemoryContextStore(ContextStore):
    """Simple in-memory context store."""

    def __init__(self):
        self._documents: dict[str, Document] = {}

    async def add_document(
        self,
        text: str,
        metadata: dict,
        embedding: Optional[list[float]] = None,
    ) -> str:
        doc_id = str(uuid.uuid4())
        self._documents[doc_id] = Document(
            id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )
        return doc_id

    async def get_document(self, doc_id: str) -> Optional[Document]:
        return self._documents.get(doc_id)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        results = []
        query_np = np.array(query_embedding)

        for doc in self._documents.values():
            # Apply metadata filter
            if filter_metadata:
                if not all(
                    doc.metadata.get(k) == v
                    for k, v in filter_metadata.items()
                ):
                    continue

            # Skip if no embedding
            if doc.embedding is None:
                continue

            # Calculate cosine similarity
            doc_np = np.array(doc.embedding)
            similarity = float(
                np.dot(query_np, doc_np) /
                (np.linalg.norm(query_np) * np.linalg.norm(doc_np))
            )
            results.append((doc, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    async def get_all_documents(
        self,
        filter_metadata: Optional[dict] = None,
    ) -> List[Document]:
        docs = list(self._documents.values())

        if filter_metadata:
            docs = [
                doc for doc in docs
                if all(
                    doc.metadata.get(k) == v
                    for k, v in filter_metadata.items()
                )
            ]

        # Sort by chunk_idx if available
        docs.sort(key=lambda d: d.metadata.get("chunk_idx", 0))

        return docs

    async def delete_document(self, doc_id: str) -> bool:
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    async def clear(self) -> int:
        count = len(self._documents)
        self._documents.clear()
        return count
```

## Milvus Store Implementation

```python
from typing import Optional, List
from pymilvus import (
    Collection,
    connections,
    utility,
    CollectionSchema,
    FieldSchema,
    DataType,
)


class MilvusContextStore(ContextStore):
    """Milvus-based context store."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "vss_poc_context",
        embedding_dim: int = 768,
    ):
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

        # Connect to Milvus
        connections.connect(host=host, port=port)

        # Create collection if not exists
        self._ensure_collection()

    def _ensure_collection(self):
        if utility.has_collection(self._collection_name):
            self._collection = Collection(self._collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._embedding_dim),
            FieldSchema(name="stream_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_idx", dtype=DataType.INT64),
            FieldSchema(name="start_time", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="end_time", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields=fields)
        self._collection = Collection(name=self._collection_name, schema=schema)

        # Create index
        self._collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )

    async def add_document(
        self,
        text: str,
        metadata: dict,
        embedding: Optional[list[float]] = None,
    ) -> str:
        doc_id = str(uuid.uuid4())

        self._collection.insert([
            [doc_id],
            [text],
            [embedding or [0.0] * self._embedding_dim],
            [metadata.get("stream_id", "")],
            [metadata.get("chunk_idx", 0)],
            [metadata.get("start_time", "")],
            [metadata.get("end_time", "")],
        ])
        self._collection.flush()

        return doc_id

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        self._collection.load()

        # Build filter expression
        expr = None
        if filter_metadata:
            conditions = []
            if "stream_id" in filter_metadata:
                conditions.append(f'stream_id == "{filter_metadata["stream_id"]}"')
            if conditions:
                expr = " && ".join(conditions)

        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
            output_fields=["text", "stream_id", "chunk_idx", "start_time", "end_time"],
        )

        documents = []
        for hit in results[0]:
            doc = Document(
                id=hit.id,
                text=hit.entity.get("text"),
                metadata={
                    "stream_id": hit.entity.get("stream_id"),
                    "chunk_idx": hit.entity.get("chunk_idx"),
                    "start_time": hit.entity.get("start_time"),
                    "end_time": hit.entity.get("end_time"),
                },
            )
            documents.append((doc, hit.score))

        return documents

    async def get_all_documents(
        self,
        filter_metadata: Optional[dict] = None,
    ) -> List[Document]:
        self._collection.load()

        expr = None
        if filter_metadata and "stream_id" in filter_metadata:
            expr = f'stream_id == "{filter_metadata["stream_id"]}"'

        # Query all documents
        results = self._collection.query(
            expr=expr or "id != ''",
            output_fields=["id", "text", "stream_id", "chunk_idx", "start_time", "end_time"],
        )

        documents = [
            Document(
                id=r["id"],
                text=r["text"],
                metadata={
                    "stream_id": r["stream_id"],
                    "chunk_idx": r["chunk_idx"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                },
            )
            for r in results
        ]

        # Sort by chunk_idx
        documents.sort(key=lambda d: d.metadata.get("chunk_idx", 0))

        return documents

    async def delete_document(self, doc_id: str) -> bool:
        self._collection.delete(f'id == "{doc_id}"')
        return True

    async def clear(self) -> int:
        count = self._collection.num_entities
        self._collection.delete("id != ''")
        return count
```

## Context Manager Implementation

```python
class ContextManager:
    """Manages video context for RAG operations."""

    def __init__(
        self,
        embeddings: "GeminiEmbeddings",
        store: Optional[ContextStore] = None,
        max_context_tokens: int = 100000,
    ):
        self._embeddings = embeddings
        self._store = store or InMemoryContextStore()
        self._max_context_tokens = max_context_tokens

    async def add_document(
        self,
        text: str,
        metadata: dict,
        generate_embedding: bool = True,
    ) -> str:
        embedding = None
        if generate_embedding:
            embedding = await self._embeddings.embed_document(text)

        return await self._store.add_document(
            text=text,
            metadata=metadata,
            embedding=embedding,
        )

    async def add_documents_batch(
        self,
        texts: list[str],
        metadatas: list[dict],
        generate_embeddings: bool = True,
    ) -> list[str]:
        embeddings = None
        if generate_embeddings:
            embeddings = await self._embeddings.embed_documents(texts)

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
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        # Generate query embedding
        query_embedding = await self._embeddings.embed_query(query)

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
        selected_docs = []
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
        filter_metadata = {"stream_id": stream_id} if stream_id else None
        docs = await self._store.get_all_documents(filter_metadata)
        return [doc.text for doc in docs]

    async def get_captions_by_time_range(
        self,
        start_time: str,
        end_time: str,
    ) -> list[str]:
        docs = await self._store.get_all_documents()

        # Filter by time range
        filtered = []
        for doc in docs:
            doc_start = doc.metadata.get("start_time", "")
            doc_end = doc.metadata.get("end_time", "")
            if doc_start >= start_time and doc_end <= end_time:
                filtered.append(doc.text)

        return filtered

    async def clear_context(self, stream_id: Optional[str] = None) -> int:
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
        return len(text) // 4
```

## Configuration

```yaml
# config/config.yaml
rag:
  enabled: true
  max_context_tokens: 100000

  vector_db:
    enabled: false  # Set to true to use Milvus
    type: milvus
    host: localhost
    port: 19530
    collection_name: "vss_poc_context"
```

## Testing

```python
# tests/test_context_manager.py

import pytest
from poc.src.rag.context_manager import ContextManager, InMemoryContextStore


class TestContextManager:
    async def test_add_document(self):
        """Test adding a document."""
        pass

    async def test_retrieve(self):
        """Test retrieval."""
        pass

    async def test_context_window(self):
        """Test context window building."""
        pass

    async def test_clear_context(self):
        """Test clearing context."""
        pass
```
