"""Gemini Embeddings for vector search and retrieval."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import google.generativeai as genai
import numpy as np


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
    ) -> None:
        """
        Initialize Gemini embeddings client.

        Args:
            api_key: Google AI Studio API key
            model: Embedding model to use
        """
        self._api_key = api_key
        self._model = model

        genai.configure(api_key=api_key)
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _truncate_text(self, text: str) -> str:
        """Truncate text to maximum length."""
        if len(text) > self.MAX_TEXT_LENGTH:
            return text[: self.MAX_TEXT_LENGTH]
        return text

    def _chunk_batch(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches of MAX_BATCH_SIZE."""
        return [
            texts[i : i + self.MAX_BATCH_SIZE]
            for i in range(0, len(texts), self.MAX_BATCH_SIZE)
        ]

    def _embed_sync(
        self,
        text: str,
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> list[float]:
        """Synchronous embedding for use with executor."""
        text = self._truncate_text(text)

        result = genai.embed_content(
            model=self._model,
            content=text,
            task_type=task_type.value,
        )

        return result["embedding"]

    def _embed_batch_sync(
        self,
        texts: list[str],
        task_type: TaskType = TaskType.RETRIEVAL_DOCUMENT,
    ) -> list[list[float]]:
        """Synchronous batch embedding for use with executor."""
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
            all_embeddings.extend(result["embedding"])

        return all_embeddings

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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_sync,
            text,
            task_type,
        )

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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_batch_sync,
            texts,
            task_type,
        )

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
