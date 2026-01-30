"""Unit tests for Milvus Client."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.milvus_client import (
    MilvusClient,
    MilvusConfig,
    SearchResult,
    VectorDocument,
)


class TestMilvusConfig:
    """Tests for MilvusConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MilvusConfig()
        assert config.host == "localhost"
        assert config.port == 19530
        assert config.collection_name == "vss_poc_captions"
        assert config.embedding_dim == 768
        assert config.alias == "default"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = MilvusConfig(
            host="milvus.example.com",
            port=19531,
            collection_name="custom_collection",
            embedding_dim=1024,
        )
        assert config.host == "milvus.example.com"
        assert config.port == 19531
        assert config.collection_name == "custom_collection"
        assert config.embedding_dim == 1024


class TestVectorDocument:
    """Tests for VectorDocument dataclass."""

    def test_create_document(self) -> None:
        """Test creating a VectorDocument."""
        doc = VectorDocument(
            id="test_001",
            text="Test caption",
            embedding=[0.1] * 768,
            stream_id="stream_123",
            chunk_idx=0,
            start_time="00:00:00",
            end_time="00:01:00",
        )

        assert doc.id == "test_001"
        assert doc.text == "Test caption"
        assert len(doc.embedding) == 768
        assert doc.stream_id == "stream_123"
        assert doc.chunk_idx == 0
        assert doc.start_time == "00:00:00"
        assert doc.end_time == "00:01:00"

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        doc = VectorDocument(
            id="test_001",
            text="Test",
            embedding=[0.1],
            stream_id="stream",
            chunk_idx=0,
            start_time="00:00:00",
            end_time="00:01:00",
        )

        assert doc.start_pts == 0
        assert doc.end_pts == 0
        assert doc.cv_meta == ""
        assert doc.created_at > 0  # Should have a timestamp


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self) -> None:
        """Test creating a SearchResult."""
        doc = VectorDocument(
            id="test_001",
            text="Test",
            embedding=[],
            stream_id="stream",
            chunk_idx=0,
            start_time="00:00:00",
            end_time="00:01:00",
        )

        result = SearchResult(document=doc, score=0.95)

        assert result.document.id == "test_001"
        assert result.score == 0.95


class TestMilvusClient:
    """Tests for MilvusClient class."""

    @pytest.fixture
    def config(self) -> MilvusConfig:
        """Create test configuration."""
        return MilvusConfig(
            host="localhost",
            port=19530,
            collection_name="test_collection",
        )

    @pytest.fixture
    def sample_document(self) -> VectorDocument:
        """Create a sample document."""
        return VectorDocument(
            id="test_001",
            text="A forklift moves pallets in the warehouse",
            embedding=[0.1] * 768,
            stream_id="video_123",
            chunk_idx=0,
            start_time="00:00:00",
            end_time="00:01:00",
            start_pts=0,
            end_pts=60000000000,
            cv_meta='{"objects": ["forklift", "pallet"]}',
            created_at=int(time.time()),
        )

    def test_init(self, config: MilvusConfig) -> None:
        """Test MilvusClient initialization."""
        client = MilvusClient(config)
        assert client._config == config
        assert client._collection is None
        assert client._connected is False

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.connections")
    async def test_connect(
        self,
        mock_connections: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test connecting to Milvus."""
        client = MilvusClient(config)
        await client.connect()

        mock_connections.connect.assert_called_once_with(
            alias=config.alias,
            host=config.host,
            port=config.port,
        )
        assert client._connected is True

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.connections")
    async def test_close(
        self,
        mock_connections: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test closing connection."""
        client = MilvusClient(config)
        client._connected = True

        await client.close()

        mock_connections.disconnect.assert_called_once_with(alias=config.alias)
        assert client._connected is False

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    async def test_ensure_collection_exists(
        self,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test ensure_collection when collection exists."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        client = MilvusClient(config)
        await client.ensure_collection()

        mock_utility.has_collection.assert_called_once()
        assert client._collection == mock_collection

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    @patch("src.db.milvus_client.CollectionSchema")
    async def test_ensure_collection_creates_new(
        self,
        mock_schema_class: MagicMock,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test ensure_collection creates new collection."""
        mock_utility.has_collection.return_value = False
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        client = MilvusClient(config)
        await client.ensure_collection()

        mock_collection_class.assert_called_once()
        mock_collection.create_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_not_initialized(
        self,
        config: MilvusConfig,
    ) -> None:
        """Test _get_collection raises error when not initialized."""
        client = MilvusClient(config)

        with pytest.raises(RuntimeError, match="Collection not initialized"):
            client._get_collection()

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    async def test_insert(
        self,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
        sample_document: VectorDocument,
    ) -> None:
        """Test inserting a document."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        client = MilvusClient(config)
        await client.ensure_collection()
        doc_id = await client.insert(sample_document)

        assert doc_id == sample_document.id
        mock_collection.insert.assert_called_once()
        mock_collection.flush.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    async def test_insert_batch(
        self,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test batch inserting documents."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        documents = [
            VectorDocument(
                id=f"test_{i:03d}",
                text=f"Caption {i}",
                embedding=[0.1] * 768,
                stream_id="video_123",
                chunk_idx=i,
                start_time=f"00:{i:02d}:00",
                end_time=f"00:{i+1:02d}:00",
            )
            for i in range(5)
        ]

        client = MilvusClient(config)
        await client.ensure_collection()
        doc_ids = await client.insert_batch(documents)

        assert len(doc_ids) == 5
        assert doc_ids[0] == "test_000"
        mock_collection.flush.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    async def test_search(
        self,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test searching for documents."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        # Mock search results
        mock_hit = MagicMock()
        mock_hit.id = "test_001"
        mock_hit.score = 0.95
        mock_hit.entity.get = lambda key, default="": {
            "text": "Test caption",
            "stream_id": "video_123",
            "chunk_idx": 0,
            "start_time": "00:00:00",
            "end_time": "00:01:00",
            "start_pts": 0,
            "end_pts": 60000000000,
            "cv_meta": "",
            "created_at": 0,
        }.get(key, default)

        mock_collection.search.return_value = [[mock_hit]]

        client = MilvusClient(config)
        await client.ensure_collection()
        results = await client.search([0.1] * 768, top_k=5)

        assert len(results) == 1
        assert results[0].document.id == "test_001"
        assert results[0].score == 0.95
        mock_collection.load.assert_called_once()
        mock_collection.search.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    async def test_search_by_stream(
        self,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test searching within a stream."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection
        mock_collection.search.return_value = [[]]

        client = MilvusClient(config)
        await client.ensure_collection()
        await client.search_by_stream([0.1] * 768, "video_123")

        # Verify filter expression was used
        call_args = mock_collection.search.call_args
        assert 'stream_id == "video_123"' in str(call_args)

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    async def test_get_by_id(
        self,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test getting document by ID."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        mock_collection.query.return_value = [{
            "id": "test_001",
            "text": "Test caption",
            "stream_id": "video_123",
            "chunk_idx": 0,
            "start_time": "00:00:00",
            "end_time": "00:01:00",
            "start_pts": 0,
            "end_pts": 60000000000,
            "cv_meta": "",
            "created_at": 0,
        }]

        client = MilvusClient(config)
        await client.ensure_collection()
        doc = await client.get_by_id("test_001")

        assert doc is not None
        assert doc.id == "test_001"
        assert doc.text == "Test caption"

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    async def test_get_by_id_not_found(
        self,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test getting non-existent document."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection
        mock_collection.query.return_value = []

        client = MilvusClient(config)
        await client.ensure_collection()
        doc = await client.get_by_id("nonexistent")

        assert doc is None

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    async def test_delete(
        self,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test deleting a document."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        client = MilvusClient(config)
        await client.ensure_collection()
        result = await client.delete("test_001")

        assert result is True
        mock_collection.delete.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    async def test_delete_by_stream(
        self,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test deleting all documents for a stream."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection
        mock_collection.query.return_value = [{"id": "1"}, {"id": "2"}, {"id": "3"}]

        client = MilvusClient(config)
        await client.ensure_collection()
        count = await client.delete_by_stream("video_123")

        assert count == 3
        mock_collection.delete.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    async def test_drop_collection(
        self,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test dropping collection."""
        mock_utility.has_collection.return_value = True

        client = MilvusClient(config)
        await client.drop_collection()

        mock_utility.drop_collection.assert_called_once()
        assert client._collection is None

    @pytest.mark.asyncio
    @patch("src.db.milvus_client.utility")
    @patch("src.db.milvus_client.Collection")
    async def test_get_collection_stats(
        self,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
        config: MilvusConfig,
    ) -> None:
        """Test getting collection statistics."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection.num_entities = 100
        mock_collection.schema = "test_schema"
        mock_collection_class.return_value = mock_collection

        client = MilvusClient(config)
        await client.ensure_collection()
        stats = await client.get_collection_stats()

        assert stats["name"] == config.collection_name
        assert stats["num_entities"] == 100

    def test_generate_id(self) -> None:
        """Test ID generation."""
        id1 = MilvusClient.generate_id()
        id2 = MilvusClient.generate_id()

        assert id1 != id2
        assert len(id1) == 36  # UUID format

    def test_index_params(self) -> None:
        """Test index parameters are defined."""
        assert MilvusClient.INDEX_PARAMS["metric_type"] == "COSINE"
        assert MilvusClient.INDEX_PARAMS["index_type"] == "IVF_FLAT"

    def test_search_params(self) -> None:
        """Test search parameters are defined."""
        assert MilvusClient.SEARCH_PARAMS["metric_type"] == "COSINE"
