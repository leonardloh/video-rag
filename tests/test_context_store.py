"""Unit tests for Context Store."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.rag.context_store import (
    ContextWindow,
    Document,
    InMemoryContextStore,
    RetrievalResult,
)


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document(self) -> None:
        """Test creating a Document."""
        doc = Document(
            id="doc_001",
            text="Test caption",
            embedding=[0.1, 0.2, 0.3],
            metadata={"chunk_idx": 0, "stream_id": "video_123"},
        )

        assert doc.id == "doc_001"
        assert doc.text == "Test caption"
        assert len(doc.embedding) == 3
        assert doc.metadata["chunk_idx"] == 0

    def test_default_values(self) -> None:
        """Test default values."""
        doc = Document(id="doc_001", text="Test")

        assert doc.embedding is None
        assert doc.metadata == {}
        assert isinstance(doc.created_at, datetime)


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a RetrievalResult."""
        docs = [Document(id="doc_001", text="Test")]
        result = RetrievalResult(
            documents=docs,
            scores=[0.95],
            query="test query",
            top_k=5,
        )

        assert len(result.documents) == 1
        assert result.scores[0] == 0.95
        assert result.query == "test query"
        assert result.top_k == 5


class TestContextWindow:
    """Tests for ContextWindow dataclass."""

    def test_create_window(self) -> None:
        """Test creating a ContextWindow."""
        docs = [Document(id="doc_001", text="Test")]
        window = ContextWindow(
            documents=docs,
            total_tokens=100,
            truncated=False,
        )

        assert len(window.documents) == 1
        assert window.total_tokens == 100
        assert window.truncated is False

    def test_default_truncated(self) -> None:
        """Test default truncated value."""
        window = ContextWindow(documents=[], total_tokens=0)
        assert window.truncated is False


class TestInMemoryContextStore:
    """Tests for InMemoryContextStore class."""

    @pytest.fixture
    def store(self) -> InMemoryContextStore:
        """Create test store."""
        return InMemoryContextStore()

    @pytest.mark.asyncio
    async def test_add_document(self, store: InMemoryContextStore) -> None:
        """Test adding a document."""
        doc_id = await store.add_document(
            text="Test caption",
            metadata={"chunk_idx": 0},
            embedding=[0.1] * 768,
        )

        assert doc_id is not None
        assert len(doc_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_get_document(self, store: InMemoryContextStore) -> None:
        """Test getting a document."""
        doc_id = await store.add_document(
            text="Test caption",
            metadata={"chunk_idx": 0},
        )

        doc = await store.get_document(doc_id)

        assert doc is not None
        assert doc.id == doc_id
        assert doc.text == "Test caption"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, store: InMemoryContextStore) -> None:
        """Test getting non-existent document."""
        doc = await store.get_document("nonexistent")
        assert doc is None

    @pytest.mark.asyncio
    async def test_search(self, store: InMemoryContextStore) -> None:
        """Test searching for documents."""
        # Add documents with embeddings
        await store.add_document(
            text="Forklift moving pallets",
            metadata={"chunk_idx": 0},
            embedding=[0.9, 0.1, 0.0] + [0.0] * 765,
        )
        await store.add_document(
            text="Worker walking",
            metadata={"chunk_idx": 1},
            embedding=[0.1, 0.9, 0.0] + [0.0] * 765,
        )

        # Search with query similar to first document
        results = await store.search(
            query_embedding=[0.8, 0.2, 0.0] + [0.0] * 765,
            top_k=2,
        )

        assert len(results) == 2
        # First result should be more similar
        assert results[0][0].text == "Forklift moving pallets"
        assert results[0][1] > results[1][1]

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store: InMemoryContextStore) -> None:
        """Test searching with metadata filter."""
        await store.add_document(
            text="Caption 1",
            metadata={"stream_id": "video_1", "chunk_idx": 0},
            embedding=[0.5] * 768,
        )
        await store.add_document(
            text="Caption 2",
            metadata={"stream_id": "video_2", "chunk_idx": 0},
            embedding=[0.5] * 768,
        )

        results = await store.search(
            query_embedding=[0.5] * 768,
            top_k=10,
            filter_metadata={"stream_id": "video_1"},
        )

        assert len(results) == 1
        assert results[0][0].metadata["stream_id"] == "video_1"

    @pytest.mark.asyncio
    async def test_search_no_embedding(self, store: InMemoryContextStore) -> None:
        """Test that documents without embeddings are skipped."""
        await store.add_document(
            text="No embedding",
            metadata={"chunk_idx": 0},
            embedding=None,
        )

        results = await store.search(
            query_embedding=[0.5] * 768,
            top_k=10,
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_zero_query(self, store: InMemoryContextStore) -> None:
        """Test search with zero vector query."""
        await store.add_document(
            text="Test",
            metadata={},
            embedding=[0.5] * 768,
        )

        results = await store.search(
            query_embedding=[0.0] * 768,
            top_k=10,
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_all_documents(self, store: InMemoryContextStore) -> None:
        """Test getting all documents."""
        await store.add_document(text="Caption 1", metadata={"chunk_idx": 1})
        await store.add_document(text="Caption 2", metadata={"chunk_idx": 0})
        await store.add_document(text="Caption 3", metadata={"chunk_idx": 2})

        docs = await store.get_all_documents()

        assert len(docs) == 3
        # Should be sorted by chunk_idx
        assert docs[0].metadata["chunk_idx"] == 0
        assert docs[1].metadata["chunk_idx"] == 1
        assert docs[2].metadata["chunk_idx"] == 2

    @pytest.mark.asyncio
    async def test_get_all_documents_filtered(self, store: InMemoryContextStore) -> None:
        """Test getting documents with filter."""
        await store.add_document(
            text="Caption 1",
            metadata={"stream_id": "video_1", "chunk_idx": 0},
        )
        await store.add_document(
            text="Caption 2",
            metadata={"stream_id": "video_2", "chunk_idx": 0},
        )

        docs = await store.get_all_documents(filter_metadata={"stream_id": "video_1"})

        assert len(docs) == 1
        assert docs[0].metadata["stream_id"] == "video_1"

    @pytest.mark.asyncio
    async def test_delete_document(self, store: InMemoryContextStore) -> None:
        """Test deleting a document."""
        doc_id = await store.add_document(text="Test", metadata={})

        result = await store.delete_document(doc_id)
        assert result is True

        doc = await store.get_document(doc_id)
        assert doc is None

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, store: InMemoryContextStore) -> None:
        """Test deleting non-existent document."""
        result = await store.delete_document("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self, store: InMemoryContextStore) -> None:
        """Test clearing all documents."""
        await store.add_document(text="Caption 1", metadata={})
        await store.add_document(text="Caption 2", metadata={})
        await store.add_document(text="Caption 3", metadata={})

        count = await store.clear()

        assert count == 3
        docs = await store.get_all_documents()
        assert len(docs) == 0

    def test_len(self, store: InMemoryContextStore) -> None:
        """Test __len__ method."""
        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_len_after_add(self, store: InMemoryContextStore) -> None:
        """Test __len__ after adding documents."""
        await store.add_document(text="Test 1", metadata={})
        await store.add_document(text="Test 2", metadata={})

        assert len(store) == 2
