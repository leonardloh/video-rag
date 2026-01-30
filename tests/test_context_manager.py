"""Unit tests for Context Manager."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.context_manager import ContextManager, ContextManagerConfig
from src.rag.context_store import ContextWindow, Document, InMemoryContextStore


class TestContextManagerConfig:
    """Tests for ContextManagerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ContextManagerConfig()
        assert config.max_context_tokens == 100000
        assert config.generate_embeddings is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ContextManagerConfig(
            max_context_tokens=50000,
            generate_embeddings=False,
        )
        assert config.max_context_tokens == 50000
        assert config.generate_embeddings is False


class TestContextManager:
    """Tests for ContextManager class."""

    @pytest.fixture
    def mock_embeddings(self) -> AsyncMock:
        """Create mock embeddings client."""
        mock = AsyncMock()
        mock.embed_document = AsyncMock(return_value=[0.1] * 768)
        mock.embed_documents = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
        mock.embed_query = AsyncMock(return_value=[0.15] * 768)
        return mock

    @pytest.fixture
    def store(self) -> InMemoryContextStore:
        """Create test store."""
        return InMemoryContextStore()

    @pytest.fixture
    def manager(
        self,
        mock_embeddings: AsyncMock,
        store: InMemoryContextStore,
    ) -> ContextManager:
        """Create ContextManager instance."""
        return ContextManager(
            embeddings=mock_embeddings,
            store=store,
            max_context_tokens=10000,
        )

    def test_init(self, mock_embeddings: AsyncMock) -> None:
        """Test ContextManager initialization."""
        manager = ContextManager(
            embeddings=mock_embeddings,
            max_context_tokens=50000,
        )

        assert manager._embeddings == mock_embeddings
        assert manager._max_context_tokens == 50000
        assert isinstance(manager._store, InMemoryContextStore)

    def test_init_with_store(
        self,
        mock_embeddings: AsyncMock,
    ) -> None:
        """Test initialization with custom store."""
        custom_store = InMemoryContextStore()
        manager = ContextManager(
            embeddings=mock_embeddings,
            store=custom_store,
        )

        assert manager._store is custom_store

    @pytest.mark.asyncio
    async def test_add_document(
        self,
        manager: ContextManager,
        mock_embeddings: AsyncMock,
    ) -> None:
        """Test adding a document."""
        doc_id = await manager.add_document(
            text="Test caption",
            metadata={"chunk_idx": 0, "stream_id": "video_123"},
        )

        assert doc_id is not None
        mock_embeddings.embed_document.assert_called_once_with("Test caption")

    @pytest.mark.asyncio
    async def test_add_document_no_embedding(
        self,
        manager: ContextManager,
        mock_embeddings: AsyncMock,
    ) -> None:
        """Test adding a document without embedding."""
        doc_id = await manager.add_document(
            text="Test caption",
            metadata={"chunk_idx": 0},
            generate_embedding=False,
        )

        assert doc_id is not None
        mock_embeddings.embed_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_document_embedding_failure(
        self,
        manager: ContextManager,
        mock_embeddings: AsyncMock,
    ) -> None:
        """Test adding document when embedding fails."""
        mock_embeddings.embed_document.side_effect = Exception("API Error")

        # Should not raise, just log warning
        doc_id = await manager.add_document(
            text="Test caption",
            metadata={"chunk_idx": 0},
        )

        assert doc_id is not None

    @pytest.mark.asyncio
    async def test_add_documents_batch(
        self,
        manager: ContextManager,
        mock_embeddings: AsyncMock,
    ) -> None:
        """Test batch adding documents."""
        doc_ids = await manager.add_documents_batch(
            texts=["Caption 1", "Caption 2"],
            metadatas=[{"chunk_idx": 0}, {"chunk_idx": 1}],
        )

        assert len(doc_ids) == 2
        mock_embeddings.embed_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve(
        self,
        manager: ContextManager,
        mock_embeddings: AsyncMock,
    ) -> None:
        """Test retrieving documents."""
        # Add documents first
        await manager.add_document(
            text="Forklift moving",
            metadata={"chunk_idx": 0},
        )
        await manager.add_document(
            text="Worker walking",
            metadata={"chunk_idx": 1},
        )

        results = await manager.retrieve(query="forklift", top_k=5)

        assert len(results) == 2
        mock_embeddings.embed_query.assert_called_once_with("forklift")

    @pytest.mark.asyncio
    async def test_retrieve_with_filter(
        self,
        manager: ContextManager,
    ) -> None:
        """Test retrieving with metadata filter."""
        await manager.add_document(
            text="Caption 1",
            metadata={"stream_id": "video_1", "chunk_idx": 0},
        )
        await manager.add_document(
            text="Caption 2",
            metadata={"stream_id": "video_2", "chunk_idx": 0},
        )

        results = await manager.retrieve(
            query="test",
            filter_metadata={"stream_id": "video_1"},
        )

        assert len(results) == 1
        assert results[0]["metadata"]["stream_id"] == "video_1"

    @pytest.mark.asyncio
    async def test_retrieve_embedding_failure(
        self,
        manager: ContextManager,
        mock_embeddings: AsyncMock,
    ) -> None:
        """Test retrieve when embedding fails."""
        mock_embeddings.embed_query.side_effect = Exception("API Error")

        results = await manager.retrieve(query="test")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_context_window(
        self,
        manager: ContextManager,
    ) -> None:
        """Test getting context window."""
        await manager.add_document(
            text="Short caption",
            metadata={"chunk_idx": 0},
        )

        window = await manager.get_context_window(query="test")

        assert isinstance(window, ContextWindow)
        assert len(window.documents) >= 0

    @pytest.mark.asyncio
    async def test_get_context_window_include_all(
        self,
        manager: ContextManager,
    ) -> None:
        """Test getting context window with all documents."""
        await manager.add_document(
            text="Caption 1",
            metadata={"chunk_idx": 0},
        )
        await manager.add_document(
            text="Caption 2",
            metadata={"chunk_idx": 1},
        )

        window = await manager.get_context_window(query="", include_all=True)

        assert len(window.documents) == 2

    @pytest.mark.asyncio
    async def test_get_context_window_truncation(
        self,
        mock_embeddings: AsyncMock,
        store: InMemoryContextStore,
    ) -> None:
        """Test context window truncation."""
        # Create manager with very small token limit
        manager = ContextManager(
            embeddings=mock_embeddings,
            store=store,
            max_context_tokens=10,  # Very small
        )

        # Add document that exceeds limit
        await manager.add_document(
            text="This is a very long caption that exceeds the token limit",
            metadata={"chunk_idx": 0},
        )

        window = await manager.get_context_window(query="", include_all=True)

        assert window.truncated is True

    @pytest.mark.asyncio
    async def test_get_all_captions(
        self,
        manager: ContextManager,
    ) -> None:
        """Test getting all captions."""
        await manager.add_document(
            text="Caption 1",
            metadata={"chunk_idx": 0},
        )
        await manager.add_document(
            text="Caption 2",
            metadata={"chunk_idx": 1},
        )

        captions = await manager.get_all_captions()

        assert len(captions) == 2
        assert "Caption 1" in captions
        assert "Caption 2" in captions

    @pytest.mark.asyncio
    async def test_get_all_captions_filtered(
        self,
        manager: ContextManager,
    ) -> None:
        """Test getting captions filtered by stream."""
        await manager.add_document(
            text="Caption 1",
            metadata={"stream_id": "video_1", "chunk_idx": 0},
        )
        await manager.add_document(
            text="Caption 2",
            metadata={"stream_id": "video_2", "chunk_idx": 0},
        )

        captions = await manager.get_all_captions(stream_id="video_1")

        assert len(captions) == 1
        assert captions[0] == "Caption 1"

    @pytest.mark.asyncio
    async def test_get_captions_by_time_range(
        self,
        manager: ContextManager,
    ) -> None:
        """Test getting captions by time range."""
        await manager.add_document(
            text="Caption 1",
            metadata={"start_time": "00:00:00", "end_time": "00:01:00", "chunk_idx": 0},
        )
        await manager.add_document(
            text="Caption 2",
            metadata={"start_time": "00:01:00", "end_time": "00:02:00", "chunk_idx": 1},
        )
        await manager.add_document(
            text="Caption 3",
            metadata={"start_time": "00:02:00", "end_time": "00:03:00", "chunk_idx": 2},
        )

        captions = await manager.get_captions_by_time_range(
            start_time="00:00:00",
            end_time="00:02:00",
        )

        assert len(captions) == 2

    @pytest.mark.asyncio
    async def test_get_document(
        self,
        manager: ContextManager,
    ) -> None:
        """Test getting a document by ID."""
        doc_id = await manager.add_document(
            text="Test caption",
            metadata={"chunk_idx": 0},
        )

        doc = await manager.get_document(doc_id)

        assert doc is not None
        assert doc["text"] == "Test caption"

    @pytest.mark.asyncio
    async def test_get_document_not_found(
        self,
        manager: ContextManager,
    ) -> None:
        """Test getting non-existent document."""
        doc = await manager.get_document("nonexistent")
        assert doc is None

    @pytest.mark.asyncio
    async def test_delete_document(
        self,
        manager: ContextManager,
    ) -> None:
        """Test deleting a document."""
        doc_id = await manager.add_document(
            text="Test caption",
            metadata={"chunk_idx": 0},
        )

        result = await manager.delete_document(doc_id)
        assert result is True

        doc = await manager.get_document(doc_id)
        assert doc is None

    @pytest.mark.asyncio
    async def test_clear_context(
        self,
        manager: ContextManager,
    ) -> None:
        """Test clearing all context."""
        await manager.add_document(text="Caption 1", metadata={"chunk_idx": 0})
        await manager.add_document(text="Caption 2", metadata={"chunk_idx": 1})

        count = await manager.clear_context()

        assert count == 2
        captions = await manager.get_all_captions()
        assert len(captions) == 0

    @pytest.mark.asyncio
    async def test_clear_context_by_stream(
        self,
        manager: ContextManager,
    ) -> None:
        """Test clearing context for specific stream."""
        await manager.add_document(
            text="Caption 1",
            metadata={"stream_id": "video_1", "chunk_idx": 0},
        )
        await manager.add_document(
            text="Caption 2",
            metadata={"stream_id": "video_2", "chunk_idx": 0},
        )

        count = await manager.clear_context(stream_id="video_1")

        assert count == 1
        captions = await manager.get_all_captions()
        assert len(captions) == 1

    def test_estimate_tokens(self, manager: ContextManager) -> None:
        """Test token estimation."""
        # ~4 characters per token
        text = "This is a test"  # 14 characters
        tokens = manager._estimate_tokens(text)
        assert tokens == 3  # 14 // 4

    def test_register_function(self, manager: ContextManager) -> None:
        """Test registering a function."""
        mock_func = MagicMock()
        manager.register_function("test_func", mock_func)

        assert manager.get_function("test_func") == mock_func

    def test_get_function_not_found(self, manager: ContextManager) -> None:
        """Test getting non-existent function."""
        func = manager.get_function("nonexistent")
        assert func is None

    @pytest.mark.asyncio
    async def test_call_function(self, manager: ContextManager) -> None:
        """Test calling a registered function."""
        mock_func = AsyncMock()
        mock_func.execute = AsyncMock(return_value="result")
        manager.register_function("test_func", mock_func)

        result = await manager.call("test_func", arg1="value1")

        assert result == "result"
        mock_func.execute.assert_called_once_with(arg1="value1")

    @pytest.mark.asyncio
    async def test_call_callable_function(self, manager: ContextManager) -> None:
        """Test calling a callable function without execute method."""
        async def test_func(**kwargs: Any) -> str:
            return f"called with {kwargs}"

        manager.register_function("test_func", test_func)

        result = await manager.call("test_func", arg1="value1")

        assert "arg1" in result

    @pytest.mark.asyncio
    async def test_call_function_not_found(self, manager: ContextManager) -> None:
        """Test calling non-existent function."""
        with pytest.raises(ValueError, match="not found"):
            await manager.call("nonexistent")

    def test_list_functions(self, manager: ContextManager) -> None:
        """Test listing registered functions."""
        manager.register_function("func1", MagicMock())
        manager.register_function("func2", MagicMock())

        functions = manager.list_functions()

        assert "func1" in functions
        assert "func2" in functions

    def test_store_property(
        self,
        manager: ContextManager,
    ) -> None:
        """Test store property."""
        assert isinstance(manager.store, InMemoryContextStore)

    def test_max_context_tokens_property(self, manager: ContextManager) -> None:
        """Test max_context_tokens property."""
        assert manager.max_context_tokens == 10000
