"""Tests for Gemini Embeddings."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.gemini.gemini_embeddings import (
    BatchEmbeddingResult,
    EmbeddingResult,
    GeminiEmbeddings,
    TaskType,
)


class TestGeminiEmbeddings:
    """Tests for GeminiEmbeddings class."""

    @pytest.fixture
    def embeddings(self) -> GeminiEmbeddings:
        """Create an embeddings instance."""
        with patch("google.generativeai.configure"):
            return GeminiEmbeddings(api_key="test_key")

    def test_init(self, embeddings: GeminiEmbeddings) -> None:
        """Test initialization."""
        assert embeddings._api_key == "test_key"
        assert embeddings._model == "models/text-embedding-004"

    def test_init_custom_model(self) -> None:
        """Test initialization with custom model."""
        with patch("google.generativeai.configure"):
            emb = GeminiEmbeddings(api_key="test_key", model="custom-model")

        assert emb._model == "custom-model"

    def test_dimensions(self, embeddings: GeminiEmbeddings) -> None:
        """Test embedding dimensions."""
        assert embeddings.dimensions == 768

    def test_truncate_text(self, embeddings: GeminiEmbeddings) -> None:
        """Test text truncation."""
        short_text = "Hello world"
        assert embeddings._truncate_text(short_text) == short_text

        long_text = "a" * 20000
        truncated = embeddings._truncate_text(long_text)
        assert len(truncated) == 10000

    def test_chunk_batch(self, embeddings: GeminiEmbeddings) -> None:
        """Test batch chunking."""
        texts = [f"text{i}" for i in range(250)]
        chunks = embeddings._chunk_batch(texts)

        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 50

    def test_cosine_similarity_identical(self) -> None:
        """Test cosine similarity with identical vectors."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        similarity = GeminiEmbeddings.cosine_similarity(a, b)
        assert similarity == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self) -> None:
        """Test cosine similarity with orthogonal vectors."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        similarity = GeminiEmbeddings.cosine_similarity(a, b)
        assert similarity == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self) -> None:
        """Test cosine similarity with opposite vectors."""
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        similarity = GeminiEmbeddings.cosine_similarity(a, b)
        assert similarity == pytest.approx(-1.0)

    def test_cosine_similarity_partial(self) -> None:
        """Test cosine similarity with partial similarity."""
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        similarity = GeminiEmbeddings.cosine_similarity(a, b)
        # cos(45°) ≈ 0.707
        assert 0.7 < similarity < 0.75

    @pytest.mark.asyncio
    async def test_embed_text(self, embeddings: GeminiEmbeddings) -> None:
        """Test single text embedding."""
        mock_embedding = [0.1] * 768

        with patch(
            "google.generativeai.embed_content",
            return_value={"embedding": mock_embedding},
        ):
            result = await embeddings.embed_text("Hello world")

        assert len(result) == 768
        assert result[0] == 0.1

    @pytest.mark.asyncio
    async def test_embed_text_with_task_type(self, embeddings: GeminiEmbeddings) -> None:
        """Test embedding with specific task type."""
        mock_embedding = [0.2] * 768

        with patch(
            "google.generativeai.embed_content",
            return_value={"embedding": mock_embedding},
        ) as mock_embed:
            result = await embeddings.embed_text("Query text", TaskType.RETRIEVAL_QUERY)

        mock_embed.assert_called_once()
        call_args = mock_embed.call_args
        assert call_args[1]["task_type"] == "RETRIEVAL_QUERY"

    @pytest.mark.asyncio
    async def test_embed_batch(self, embeddings: GeminiEmbeddings) -> None:
        """Test batch embedding."""
        mock_embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]

        with patch(
            "google.generativeai.embed_content",
            return_value={"embedding": mock_embeddings},
        ):
            results = await embeddings.embed_batch(["text1", "text2", "text3"])

        assert len(results) == 3
        assert len(results[0]) == 768

    @pytest.mark.asyncio
    async def test_embed_query(self, embeddings: GeminiEmbeddings) -> None:
        """Test query embedding."""
        mock_embedding = [0.1] * 768

        with patch(
            "google.generativeai.embed_content",
            return_value={"embedding": mock_embedding},
        ) as mock_embed:
            result = await embeddings.embed_query("What is the weather?")

        mock_embed.assert_called_once()
        call_args = mock_embed.call_args
        assert call_args[1]["task_type"] == "RETRIEVAL_QUERY"

    @pytest.mark.asyncio
    async def test_embed_document(self, embeddings: GeminiEmbeddings) -> None:
        """Test document embedding."""
        mock_embedding = [0.1] * 768

        with patch(
            "google.generativeai.embed_content",
            return_value={"embedding": mock_embedding},
        ) as mock_embed:
            result = await embeddings.embed_document("This is a document.")

        mock_embed.assert_called_once()
        call_args = mock_embed.call_args
        assert call_args[1]["task_type"] == "RETRIEVAL_DOCUMENT"

    @pytest.mark.asyncio
    async def test_embed_documents(self, embeddings: GeminiEmbeddings) -> None:
        """Test multiple document embedding."""
        mock_embeddings = [[0.1] * 768, [0.2] * 768]

        with patch(
            "google.generativeai.embed_content",
            return_value={"embedding": mock_embeddings},
        ):
            results = await embeddings.embed_documents(["doc1", "doc2"])

        assert len(results) == 2


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_types(self) -> None:
        """Test task type values."""
        assert TaskType.RETRIEVAL_DOCUMENT.value == "RETRIEVAL_DOCUMENT"
        assert TaskType.RETRIEVAL_QUERY.value == "RETRIEVAL_QUERY"
        assert TaskType.SEMANTIC_SIMILARITY.value == "SEMANTIC_SIMILARITY"
        assert TaskType.CLASSIFICATION.value == "CLASSIFICATION"
        assert TaskType.CLUSTERING.value == "CLUSTERING"


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating an embedding result."""
        result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            dimensions=3,
            task_type=TaskType.RETRIEVAL_DOCUMENT,
        )

        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.dimensions == 3
        assert result.task_type == TaskType.RETRIEVAL_DOCUMENT


class TestBatchEmbeddingResult:
    """Tests for BatchEmbeddingResult dataclass."""

    def test_create_batch_result(self) -> None:
        """Test creating a batch embedding result."""
        result = BatchEmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            dimensions=2,
            task_type=TaskType.RETRIEVAL_DOCUMENT,
            total_tokens=100,
        )

        assert len(result.embeddings) == 2
        assert result.dimensions == 2
        assert result.total_tokens == 100
