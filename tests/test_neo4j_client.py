"""Unit tests for Neo4j Client."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.neo4j_client import (
    GraphNode,
    GraphRelationship,
    Neo4jClient,
    Neo4jConfig,
)


def create_mock_session(return_data: list[dict[str, Any]]) -> tuple[MagicMock, AsyncMock]:
    """Create a mock driver with session that returns specified data."""
    mock_driver = AsyncMock()
    mock_driver.verify_connectivity = AsyncMock()
    mock_driver.close = AsyncMock()

    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=return_data)
    mock_session.run = AsyncMock(return_value=mock_result)

    @asynccontextmanager
    async def session_context(*args: Any, **kwargs: Any) -> AsyncIterator[AsyncMock]:
        yield mock_session

    mock_driver.session = MagicMock(side_effect=session_context)

    return mock_driver, mock_session


class TestNeo4jConfig:
    """Tests for Neo4jConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = Neo4jConfig()
        assert config.host == "localhost"
        assert config.port == 7687
        assert config.username == "neo4j"
        assert config.password == ""
        assert config.database == "neo4j"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = Neo4jConfig(
            host="neo4j.example.com",
            port=7688,
            username="admin",
            password="secret",
            database="custom_db",
        )
        assert config.host == "neo4j.example.com"
        assert config.port == 7688
        assert config.username == "admin"
        assert config.password == "secret"
        assert config.database == "custom_db"

    def test_uri_property(self) -> None:
        """Test URI property."""
        config = Neo4jConfig(host="localhost", port=7687)
        assert config.uri == "bolt://localhost:7687"


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_create_node(self) -> None:
        """Test creating a GraphNode."""
        node = GraphNode(
            id="node_001",
            labels=["VideoChunk"],
            properties={"chunk_id": "chunk_001", "stream_id": "video_123"},
        )

        assert node.id == "node_001"
        assert "VideoChunk" in node.labels
        assert node.properties["chunk_id"] == "chunk_001"


class TestGraphRelationship:
    """Tests for GraphRelationship dataclass."""

    def test_create_relationship(self) -> None:
        """Test creating a GraphRelationship."""
        rel = GraphRelationship(
            id="rel_001",
            start_node_id="node_001",
            end_node_id="node_002",
            type="FOLLOWS",
            properties={"weight": 1.0},
        )

        assert rel.id == "rel_001"
        assert rel.start_node_id == "node_001"
        assert rel.end_node_id == "node_002"
        assert rel.type == "FOLLOWS"
        assert rel.properties["weight"] == 1.0

    def test_default_properties(self) -> None:
        """Test default empty properties."""
        rel = GraphRelationship(
            id="rel_001",
            start_node_id="node_001",
            end_node_id="node_002",
            type="FOLLOWS",
        )

        assert rel.properties == {}


class TestNeo4jClient:
    """Tests for Neo4jClient class."""

    @pytest.fixture
    def config(self) -> Neo4jConfig:
        """Create test configuration."""
        return Neo4jConfig(
            host="localhost",
            port=7687,
            username="neo4j",
            password="test_password",
        )

    def test_init(self, config: Neo4jConfig) -> None:
        """Test Neo4jClient initialization."""
        client = Neo4jClient(config)
        assert client._config == config
        assert client._driver is None

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_connect(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test connecting to Neo4j."""
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()

        mock_graph_db.driver.assert_called_once_with(
            config.uri,
            auth=(config.username, config.password),
        )
        mock_driver.verify_connectivity.assert_called_once()
        assert client._driver == mock_driver

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_close(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test closing connection."""
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_driver.close = AsyncMock()
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        await client.close()

        mock_driver.close.assert_called_once()
        assert client._driver is None

    def test_get_driver_not_connected(self, config: Neo4jConfig) -> None:
        """Test _get_driver raises error when not connected."""
        client = Neo4jClient(config)

        with pytest.raises(RuntimeError, match="Not connected"):
            client._get_driver()

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_execute_query(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test executing a Cypher query."""
        mock_driver, mock_session = create_mock_session([{"count": 5}])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        results = await client.execute_query("MATCH (n) RETURN count(n) as count")

        assert results == [{"count": 5}]
        mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_create_node(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test creating a node."""
        mock_driver, _ = create_mock_session([{
            "n": {"id": "node_001", "chunk_id": "chunk_001"},
            "element_id": "elem_001",
        }])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        node = await client.create_node(
            labels=["VideoChunk"],
            properties={"id": "node_001", "chunk_id": "chunk_001"},
        )

        assert isinstance(node, GraphNode)
        assert node.id == "node_001"
        assert "VideoChunk" in node.labels

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_create_relationship(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test creating a relationship."""
        mock_driver, _ = create_mock_session([{
            "r": {"id": "rel_001"},
            "element_id": "elem_001",
        }])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        rel = await client.create_relationship(
            start_node_id="node_001",
            end_node_id="node_002",
            rel_type="FOLLOWS",
            properties={"weight": 1.0},
        )

        assert isinstance(rel, GraphRelationship)
        assert rel.type == "FOLLOWS"
        assert rel.start_node_id == "node_001"
        assert rel.end_node_id == "node_002"

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_get_node_by_id(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test getting a node by ID."""
        mock_driver, _ = create_mock_session([{
            "n": {"id": "node_001", "chunk_id": "chunk_001"},
            "labels": ["VideoChunk"],
        }])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        node = await client.get_node_by_id("node_001")

        assert node is not None
        assert node.id == "node_001"
        assert "VideoChunk" in node.labels

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_get_node_by_id_not_found(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test getting non-existent node."""
        mock_driver, _ = create_mock_session([])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        node = await client.get_node_by_id("nonexistent")

        assert node is None

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_find_nodes(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test finding nodes by label."""
        mock_driver, _ = create_mock_session([
            {"n": {"id": "node_001"}, "labels": ["VideoChunk"]},
            {"n": {"id": "node_002"}, "labels": ["VideoChunk"]},
        ])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        nodes = await client.find_nodes("VideoChunk")

        assert len(nodes) == 2
        assert all(isinstance(n, GraphNode) for n in nodes)

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_get_neighbors(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test getting neighboring nodes."""
        mock_rel = MagicMock()
        mock_rel.get.return_value = "rel_001"
        mock_rel.type = "FOLLOWS"
        mock_rel.__iter__ = lambda self: iter([])

        mock_driver, _ = create_mock_session([{
            "r": mock_rel,
            "m": {"id": "node_002"},
            "labels": ["VideoChunk"],
        }])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        neighbors = await client.get_neighbors("node_001", rel_type="FOLLOWS")

        assert len(neighbors) == 1
        rel, node = neighbors[0]
        assert isinstance(rel, GraphRelationship)
        assert isinstance(node, GraphNode)

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_delete_node(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test deleting a node."""
        mock_driver, _ = create_mock_session([{"deleted": 1}])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        result = await client.delete_node("node_001")

        assert result is True

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_delete_by_stream(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test deleting all nodes for a stream."""
        mock_driver, _ = create_mock_session([{"deleted": 5}])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        count = await client.delete_by_stream("video_123")

        assert count == 5

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_clear_database(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test clearing the database."""
        mock_driver, _ = create_mock_session([{"deleted": 100}])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        count = await client.clear_database()

        assert count == 100

    @pytest.mark.asyncio
    @patch("src.db.neo4j_client.AsyncGraphDatabase")
    async def test_create_indexes(
        self,
        mock_graph_db: MagicMock,
        config: Neo4jConfig,
    ) -> None:
        """Test creating indexes."""
        mock_driver, mock_session = create_mock_session([])
        mock_graph_db.driver.return_value = mock_driver

        client = Neo4jClient(config)
        await client.connect()
        await client.create_indexes()

        # Should have called run multiple times for each index
        assert mock_session.run.call_count >= 8

    def test_generate_id(self) -> None:
        """Test ID generation."""
        id1 = Neo4jClient.generate_id()
        id2 = Neo4jClient.generate_id()

        assert id1 != id2
        assert len(id1) == 36  # UUID format
