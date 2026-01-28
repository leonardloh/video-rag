# Gemini Embeddings Specification

## Overview

The `GeminiEmbeddings` class provides text embedding generation for vector search and retrieval. This replaces the NVIDIA `llama-3.2-nv-embedqa-1b-v2` embeddings model used in the original VSS engine.

## Gap Analysis

### Original Implementation
- `src/vss-engine/config/config.yaml` - Defines `nvidia_embedding` tool
- Model: `nvidia/llama-3.2-nv-embedqa-1b-v2`
- Base URL: `https://integrate.api.nvidia.com/v1`
- Used for vector DB indexing and retrieval

### PoC Requirement
- Use Gemini `text-embedding-004` for embeddings
- Support document and query embedding task types
- Maintain compatibility with Milvus vector DB

## Component Location

```
./src/models/gemini/gemini_embeddings.py
```

## Dependencies

```python
# From requirements.txt
google-generativeai>=0.8.0
numpy>=1.24.0
```

## Data Classes

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskType(Enum):
    """Task type for embedding generation."""
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embedding: list[float]
    dimensions: int
    task_type: TaskType


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding generation."""
    embeddings: list[list[float]]
    dimensions: int
    task_type: TaskType
    total_tokens: int
```

## Class Interface

```python
from typing import Optional

import google.generativeai as genai
import numpy as np


class GeminiEmbeddings:
    """Gemini embedding generation for vector search."""

    # Default model
    DEFAULT_MODEL = "models/text-embedding-004"

    # Embedding dimensions
    EMBEDDING_DIMENSIONS = 768

    # Maximum batch size
    MAX_BATCH_SIZE = 100

    # Maximum text length (in characters)
    MAX_TEXT_LENGTH = 10000

    def __init__(
        self,
        api_key: str,
        model: str = "models/text-embedding-004",
    ):
        """
        Initialize Gemini embeddings client.

        Args:
            api_key: Google AI Studio API key
            model: Embedding model to use
        """
        self._api_key = api_key
        self._model = model

        genai.configure(api_key=api_key)

    async def embed_text(
        self,
        text: str,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> list[float]:
        """
        Generate embedding for single text.

        Args:
            text: Text to embed
            task_type: Task type for embedding

        Returns:
            List of floats representing the embedding
        """
        pass

    async def embed_batch(
        self,
        texts: list[str],
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            task_type: Task type for embedding

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        pass

    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.

        Convenience method that uses RETRIEVAL_QUERY task type.

        Args:
            query: Search query text

        Returns:
            Query embedding
        """
        return await self.embed_text(query, TaskType.RETRIEVAL_QUERY)

    async def embed_document(self, document: str) -> list[float]:
        """
        Generate embedding for a document.

        Convenience method that uses RETRIEVAL_DOCUMENT task type.

        Args:
            document: Document text

        Returns:
            Document embedding
        """
        return await self.embed_text(document, TaskType.RETRIEVAL_DOCUMENT)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            documents: List of document texts

        Returns:
            List of document embeddings
        """
        return await self.embed_batch(documents, TaskType.RETRIEVAL_DOCUMENT)

    def _truncate_text(self, text: str) -> str:
        """Truncate text to maximum length."""
        if len(text) > self.MAX_TEXT_LENGTH:
            return text[:self.MAX_TEXT_LENGTH]
        return text

    def _chunk_batch(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches of MAX_BATCH_SIZE."""
        return [
            texts[i:i + self.MAX_BATCH_SIZE]
            for i in range(0, len(texts), self.MAX_BATCH_SIZE)
        ]

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return self.EMBEDDING_DIMENSIONS

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
```

## Implementation Notes

### Single Text Embedding

```python
async def embed_text(
    self,
    text: str,
    task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
) -> list[float]:
    # Truncate if necessary
    text = self._truncate_text(text)

    # Generate embedding
    result = genai.embed_content(
        model=self._model,
        content=text,
        task_type=task_type.value,
    )

    return result['embedding']
```

### Batch Embedding

```python
async def embed_batch(
    self,
    texts: list[str],
    task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
) -> list[list[float]]:
    # Truncate all texts
    texts = [self._truncate_text(t) for t in texts]

    # Split into batches
    batches = self._chunk_batch(texts)

    all_embeddings = []
    for batch in batches:
        result = genai.embed_content(
            model=self._model,
            content=batch,
            task_type=task_type.value,
        )
        all_embeddings.extend(result['embedding'])

    return all_embeddings
```

### Async Implementation

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor


class GeminiEmbeddings:
    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        self._api_key = api_key
        self._model = model
        self._executor = ThreadPoolExecutor(max_workers=4)
        genai.configure(api_key=api_key)

    async def embed_text(
        self,
        text: str,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_sync,
            text,
            task_type,
        )

    def _embed_sync(self, text: str, task_type: TaskType) -> list[float]:
        text = self._truncate_text(text)
        result = genai.embed_content(
            model=self._model,
            content=text,
            task_type=task_type.value,
        )
        return result['embedding']
```

## Configuration

```yaml
# config/config.yaml
gemini:
  embeddings:
    model: "models/text-embedding-004"
    task_type: "RETRIEVAL_DOCUMENT"  # Default task type
    dimensions: 768
```

## Environment Variables

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
```

## Integration with Vector DB

### Milvus Integration

```python
# ./src/rag/retriever.py

from pymilvus import Collection, connections, utility

class MilvusRetriever:
    """Vector retrieval using Milvus and Gemini embeddings."""

    def __init__(
        self,
        embeddings: GeminiEmbeddings,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "vss_poc",
    ):
        self._embeddings = embeddings
        self._collection_name = collection_name

        # Connect to Milvus
        connections.connect(host=host, port=port)

        # Create collection if not exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure collection exists with correct schema."""
        if not utility.has_collection(self._collection_name):
            from pymilvus import CollectionSchema, FieldSchema, DataType

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="chunk_idx", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields=fields)
            Collection(name=self._collection_name, schema=schema)

    async def add_documents(
        self,
        texts: list[str],
        timestamps: list[str],
        chunk_indices: list[int],
    ):
        """Add documents to the vector store."""
        embeddings = await self._embeddings.embed_documents(texts)

        collection = Collection(self._collection_name)
        collection.insert([
            texts,
            embeddings,
            timestamps,
            chunk_indices,
        ])
        collection.flush()

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for similar documents."""
        query_embedding = await self._embeddings.embed_query(query)

        collection = Collection(self._collection_name)
        collection.load()

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text", "timestamp", "chunk_idx"],
        )

        return [
            {
                "text": hit.entity.get("text"),
                "timestamp": hit.entity.get("timestamp"),
                "chunk_idx": hit.entity.get("chunk_idx"),
                "score": hit.score,
            }
            for hit in results[0]
        ]
```

### In-Memory Vector Store (No Milvus)

```python
# ./src/rag/simple_retriever.py

import numpy as np
from dataclasses import dataclass


@dataclass
class Document:
    text: str
    embedding: list[float]
    timestamp: str
    chunk_idx: int


class SimpleRetriever:
    """Simple in-memory vector retriever."""

    def __init__(self, embeddings: GeminiEmbeddings):
        self._embeddings = embeddings
        self._documents: list[Document] = []

    async def add_document(
        self,
        text: str,
        timestamp: str,
        chunk_idx: int,
    ):
        """Add a document to the store."""
        embedding = await self._embeddings.embed_document(text)
        self._documents.append(Document(
            text=text,
            embedding=embedding,
            timestamp=timestamp,
            chunk_idx=chunk_idx,
        ))

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for similar documents."""
        query_embedding = await self._embeddings.embed_query(query)

        # Calculate similarities
        similarities = []
        for doc in self._documents:
            sim = GeminiEmbeddings.cosine_similarity(query_embedding, doc.embedding)
            similarities.append((sim, doc))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top_k results
        return [
            {
                "text": doc.text,
                "timestamp": doc.timestamp,
                "chunk_idx": doc.chunk_idx,
                "score": sim,
            }
            for sim, doc in similarities[:top_k]
        ]

    def clear(self):
        """Clear all documents."""
        self._documents = []
```

## Testing

```python
# tests/test_gemini_embeddings.py

import pytest
import numpy as np
from poc.src.models.gemini.gemini_embeddings import GeminiEmbeddings, TaskType


class TestGeminiEmbeddings:
    async def test_embed_text(self):
        """Test single text embedding."""
        embeddings = GeminiEmbeddings(api_key="test")
        result = await embeddings.embed_text("Hello world")
        assert len(result) == 768

    async def test_embed_batch(self):
        """Test batch embedding."""
        embeddings = GeminiEmbeddings(api_key="test")
        texts = ["Hello", "World"]
        results = await embeddings.embed_batch(texts)
        assert len(results) == 2
        assert all(len(r) == 768 for r in results)

    async def test_query_vs_document(self):
        """Test that query and document embeddings differ."""
        embeddings = GeminiEmbeddings(api_key="test")
        text = "What is the weather?"
        query_emb = await embeddings.embed_query(text)
        doc_emb = await embeddings.embed_document(text)
        # They should be different due to task type
        assert query_emb != doc_emb

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert GeminiEmbeddings.cosine_similarity(a, b) == pytest.approx(1.0)

        c = [0.0, 1.0, 0.0]
        assert GeminiEmbeddings.cosine_similarity(a, c) == pytest.approx(0.0)

    async def test_truncation(self):
        """Test text truncation for long texts."""
        embeddings = GeminiEmbeddings(api_key="test")
        long_text = "a" * 20000
        truncated = embeddings._truncate_text(long_text)
        assert len(truncated) == 10000
```

## Comparison with Original

| Feature | Original (NVIDIA) | PoC (Gemini) |
|---------|-------------------|--------------|
| Model | llama-3.2-nv-embedqa-1b-v2 | text-embedding-004 |
| Dimensions | 1024 | 768 |
| Task Types | Single | Multiple (document, query, etc.) |
| Max Batch | 96 | 100 |
| Provider | NVIDIA NIM | Google AI Studio |

## Migration Notes

When migrating from NVIDIA embeddings to Gemini:

1. **Dimension Change**: Update Milvus collection schema from 1024 to 768 dimensions
2. **Re-index**: Existing embeddings must be regenerated
3. **Task Types**: Use appropriate task type for queries vs documents
4. **API Limits**: Monitor rate limits (different from NVIDIA)
