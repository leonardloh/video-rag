"""Unit tests for Graph Ingestion."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.graph_ingestion import (
    ExtractedEntity,
    ExtractedEvent,
    GraphIngestion,
    IngestionResult,
)


class TestExtractedEntity:
    """Tests for ExtractedEntity dataclass."""

    def test_create_entity(self) -> None:
        """Test creating an ExtractedEntity."""
        entity = ExtractedEntity(
            name="forklift",
            type="VEHICLE",
            confidence=0.95,
            attributes={"color": "yellow"},
        )

        assert entity.name == "forklift"
        assert entity.type == "VEHICLE"
        assert entity.confidence == 0.95
        assert entity.attributes["color"] == "yellow"

    def test_default_values(self) -> None:
        """Test default values."""
        entity = ExtractedEntity(name="worker", type="PERSON")

        assert entity.confidence == 1.0
        assert entity.attributes == {}


class TestExtractedEvent:
    """Tests for ExtractedEvent dataclass."""

    def test_create_event(self) -> None:
        """Test creating an ExtractedEvent."""
        event = ExtractedEvent(
            description="Forklift moving pallets",
            event_type="OPERATION",
            severity="LOW",
            participants=["forklift", "pallet"],
            start_time="00:00:05",
            end_time="00:00:15",
        )

        assert event.description == "Forklift moving pallets"
        assert event.event_type == "OPERATION"
        assert event.severity == "LOW"
        assert "forklift" in event.participants

    def test_default_values(self) -> None:
        """Test default values."""
        event = ExtractedEvent(
            description="Test event",
            event_type="MOVEMENT",
            severity="MEDIUM",
        )

        assert event.participants == []
        assert event.confidence == 1.0
        assert event.start_time == ""
        assert event.end_time == ""


class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating an IngestionResult."""
        result = IngestionResult(
            chunk_node_id="chunk_001",
            entity_count=3,
            event_count=2,
            relationship_count=5,
            entities=["forklift", "worker", "pallet"],
            events=["Moving pallets"],
        )

        assert result.chunk_node_id == "chunk_001"
        assert result.entity_count == 3
        assert result.event_count == 2
        assert result.relationship_count == 5


class TestGraphIngestion:
    """Tests for GraphIngestion class."""

    @pytest.fixture
    def mock_neo4j(self) -> AsyncMock:
        """Create mock Neo4j client."""
        mock = AsyncMock()
        mock.create_node = AsyncMock(return_value=MagicMock(id="node_001"))
        mock.create_relationship = AsyncMock(return_value=MagicMock(id="rel_001"))
        mock.execute_query = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM."""
        mock = AsyncMock()
        mock.generate = AsyncMock()
        return mock

    @pytest.fixture
    def mock_embeddings(self) -> AsyncMock:
        """Create mock embeddings."""
        mock = AsyncMock()
        mock.embed_documents = AsyncMock(return_value=[[0.1] * 768])
        mock.cosine_similarity = MagicMock(return_value=0.9)
        return mock

    @pytest.fixture
    def ingestion(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> GraphIngestion:
        """Create GraphIngestion instance."""
        return GraphIngestion(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
            embeddings=mock_embeddings,
        )

    def test_init(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test GraphIngestion initialization."""
        ingestion = GraphIngestion(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
        )

        assert ingestion._neo4j == mock_neo4j
        assert ingestion._llm == mock_llm
        assert ingestion._embeddings is None

    def test_parse_json_response_valid(self, ingestion: GraphIngestion) -> None:
        """Test parsing valid JSON response."""
        response = '[{"name": "forklift", "type": "VEHICLE"}]'
        result = ingestion._parse_json_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "forklift"

    def test_parse_json_response_with_markdown(self, ingestion: GraphIngestion) -> None:
        """Test parsing JSON with markdown code block."""
        response = '```json\n[{"name": "worker", "type": "PERSON"}]\n```'
        result = ingestion._parse_json_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "worker"

    def test_parse_json_response_with_text(self, ingestion: GraphIngestion) -> None:
        """Test parsing JSON with surrounding text."""
        response = 'Here are the entities:\n[{"name": "pallet", "type": "OBJECT"}]\nDone.'
        result = ingestion._parse_json_response(response)

        assert len(result) == 1
        assert result[0]["name"] == "pallet"

    def test_parse_json_response_invalid(self, ingestion: GraphIngestion) -> None:
        """Test parsing invalid JSON."""
        response = "No JSON here"
        result = ingestion._parse_json_response(response)

        assert result == []

    @pytest.mark.asyncio
    async def test_extract_entities(
        self,
        ingestion: GraphIngestion,
        mock_llm: AsyncMock,
    ) -> None:
        """Test entity extraction."""
        mock_llm.generate.return_value = MagicMock(
            text='[{"name": "forklift", "type": "VEHICLE", "attributes": {"color": "yellow"}}]'
        )

        entities = await ingestion.extract_entities(
            caption="A yellow forklift moves pallets.",
            cv_metadata={"objects": ["forklift"]},
        )

        assert len(entities) == 1
        assert entities[0].name == "forklift"
        assert entities[0].type == "VEHICLE"
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_entities_invalid_type(
        self,
        ingestion: GraphIngestion,
        mock_llm: AsyncMock,
    ) -> None:
        """Test entity extraction with invalid type defaults to OBJECT."""
        mock_llm.generate.return_value = MagicMock(
            text='[{"name": "thing", "type": "INVALID_TYPE"}]'
        )

        entities = await ingestion.extract_entities(caption="A thing appears.")

        assert len(entities) == 1
        assert entities[0].type == "OBJECT"

    @pytest.mark.asyncio
    async def test_extract_events(
        self,
        ingestion: GraphIngestion,
        mock_llm: AsyncMock,
    ) -> None:
        """Test event extraction."""
        mock_llm.generate.return_value = MagicMock(
            text='[{"description": "Forklift moving pallets", "event_type": "OPERATION", "severity": "LOW", "participants": ["forklift", "pallet"]}]'
        )

        entities = [
            ExtractedEntity(name="forklift", type="VEHICLE"),
            ExtractedEntity(name="pallet", type="OBJECT"),
        ]

        events = await ingestion.extract_events(
            caption="A forklift moves pallets.",
            entities=entities,
        )

        assert len(events) == 1
        assert events[0].description == "Forklift moving pallets"
        assert events[0].event_type == "OPERATION"
        assert "forklift" in events[0].participants

    @pytest.mark.asyncio
    async def test_extract_events_invalid_severity(
        self,
        ingestion: GraphIngestion,
        mock_llm: AsyncMock,
    ) -> None:
        """Test event extraction with invalid severity defaults to LOW."""
        mock_llm.generate.return_value = MagicMock(
            text='[{"description": "Something happened", "event_type": "MOVEMENT", "severity": "INVALID"}]'
        )

        events = await ingestion.extract_events(caption="Something happened.", entities=[])

        assert len(events) == 1
        assert events[0].severity == "LOW"

    @pytest.mark.asyncio
    async def test_ingest_chunk(
        self,
        ingestion: GraphIngestion,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test ingesting a chunk."""
        # Mock entity extraction
        mock_llm.generate.side_effect = [
            MagicMock(text='[{"name": "forklift", "type": "VEHICLE"}]'),
            MagicMock(text='[{"description": "Moving", "event_type": "OPERATION", "severity": "LOW", "participants": ["forklift"]}]'),
        ]

        result = await ingestion.ingest_chunk(
            chunk_id="chunk_001",
            stream_id="video_123",
            chunk_idx=0,
            caption="A forklift moves.",
            start_time="00:00:00",
            end_time="00:01:00",
            embedding_id="emb_001",
        )

        assert isinstance(result, IngestionResult)
        assert result.chunk_node_id == "chunk_001"
        assert result.entity_count == 1
        assert result.event_count == 1
        # VideoChunk node + Entity node + Event node
        assert mock_neo4j.create_node.call_count >= 1

    @pytest.mark.asyncio
    async def test_ingest_chunk_no_extraction(
        self,
        ingestion: GraphIngestion,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test ingesting a chunk without extraction."""
        result = await ingestion.ingest_chunk(
            chunk_id="chunk_001",
            stream_id="video_123",
            chunk_idx=0,
            caption="A forklift moves.",
            start_time="00:00:00",
            end_time="00:01:00",
            embedding_id="emb_001",
            extract_entities=False,
            extract_events=False,
        )

        assert result.entity_count == 0
        assert result.event_count == 0
        # Only VideoChunk node created
        mock_neo4j.create_node.assert_called_once()
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_temporal_links(
        self,
        ingestion: GraphIngestion,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test creating temporal links."""
        mock_neo4j.execute_query.return_value = [{"created": 5}]

        count = await ingestion.create_temporal_links("video_123")

        assert count == 5
        mock_neo4j.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_entities_across_chunks_no_embeddings(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test entity linking without embeddings client."""
        ingestion = GraphIngestion(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
            embeddings=None,
        )

        count = await ingestion.link_entities_across_chunks("video_123")

        assert count == 0

    @pytest.mark.asyncio
    async def test_ingest_batch(
        self,
        ingestion: GraphIngestion,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test batch ingestion."""
        # Mock for each chunk's entity and event extraction
        mock_llm.generate.side_effect = [
            MagicMock(text='[{"name": "forklift", "type": "VEHICLE"}]'),
            MagicMock(text='[]'),
            MagicMock(text='[{"name": "worker", "type": "PERSON"}]'),
            MagicMock(text='[]'),
        ]

        # Mock execute_query to return empty list (no existing entity) for entity checks,
        # and {"created": 1} for temporal links
        def mock_query_side_effect(query: str, params: dict = None) -> list:
            if "MERGE" in query and "FOLLOWS" in query:
                return [{"created": 1}]
            return []

        mock_neo4j.execute_query.side_effect = mock_query_side_effect

        chunks = [
            {
                "chunk_id": "chunk_001",
                "chunk_idx": 0,
                "caption": "A forklift appears.",
                "start_time": "00:00:00",
                "end_time": "00:01:00",
                "embedding_id": "emb_001",
            },
            {
                "chunk_id": "chunk_002",
                "chunk_idx": 1,
                "caption": "A worker appears.",
                "start_time": "00:01:00",
                "end_time": "00:02:00",
                "embedding_id": "emb_002",
            },
        ]

        result = await ingestion.ingest_batch(chunks, "video_123")

        assert result["chunks_processed"] == 2
        assert result["temporal_links"] == 1

    def test_clear_cache(self, ingestion: GraphIngestion) -> None:
        """Test clearing entity cache."""
        ingestion._entity_cache["stream_1"] = {"forklift:VEHICLE": "id_1"}
        ingestion._entity_cache["stream_2"] = {"worker:PERSON": "id_2"}

        ingestion.clear_cache("stream_1")
        assert "stream_1" not in ingestion._entity_cache
        assert "stream_2" in ingestion._entity_cache

        ingestion.clear_cache()
        assert ingestion._entity_cache == {}


class TestGraphIngestionEntityDeduplication:
    """Tests for entity deduplication in GraphIngestion."""

    @pytest.fixture
    def mock_neo4j(self) -> AsyncMock:
        """Create mock Neo4j client."""
        mock = AsyncMock()
        mock.create_node = AsyncMock(return_value=MagicMock(id="node_001"))
        mock.create_relationship = AsyncMock(return_value=MagicMock(id="rel_001"))
        mock.execute_query = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM."""
        mock = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_entity_cache_hit(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test entity is retrieved from cache on second occurrence."""
        ingestion = GraphIngestion(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
        )

        # Pre-populate cache
        ingestion._entity_cache["video_123"] = {"forklift:VEHICLE": "existing_id"}

        entity = ExtractedEntity(name="forklift", type="VEHICLE")
        entity_id = await ingestion._get_or_create_entity(
            entity, "video_123", "00:01:00"
        )

        assert entity_id == "existing_id"
        # Should update last_seen, not create new node
        mock_neo4j.execute_query.assert_called()
        mock_neo4j.create_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_entity_database_hit(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test entity is retrieved from database if not in cache."""
        mock_neo4j.execute_query.return_value = [{"id": "db_entity_id"}]

        ingestion = GraphIngestion(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
        )

        entity = ExtractedEntity(name="forklift", type="VEHICLE")
        entity_id = await ingestion._get_or_create_entity(
            entity, "video_123", "00:01:00"
        )

        assert entity_id == "db_entity_id"
        # Should be added to cache
        assert "forklift:VEHICLE" in ingestion._entity_cache.get("video_123", {})
