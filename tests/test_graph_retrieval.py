"""Unit tests for Graph Retrieval."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db.graph_retrieval import (
    EntityTimeline,
    GraphRetrieval,
    TemporalContext,
)


class TestTemporalContext:
    """Tests for TemporalContext dataclass."""

    def test_create_context(self) -> None:
        """Test creating a TemporalContext."""
        context = TemporalContext(
            chunks=[{"id": "chunk_001"}],
            entities=[{"name": "forklift"}],
            events=[{"description": "Moving"}],
            relationships=[{"type": "FOLLOWS"}],
        )

        assert len(context.chunks) == 1
        assert len(context.entities) == 1
        assert len(context.events) == 1
        assert len(context.relationships) == 1

    def test_default_values(self) -> None:
        """Test default empty lists."""
        context = TemporalContext()

        assert context.chunks == []
        assert context.entities == []
        assert context.events == []
        assert context.relationships == []


class TestEntityTimeline:
    """Tests for EntityTimeline dataclass."""

    def test_create_timeline(self) -> None:
        """Test creating an EntityTimeline."""
        timeline = EntityTimeline(
            entity={"name": "forklift", "type": "VEHICLE"},
            appearances=[{"chunk_idx": 0}, {"chunk_idx": 1}],
            interactions=[{"entity": {"name": "worker"}}],
            events=[{"description": "Moving pallets"}],
        )

        assert timeline.entity["name"] == "forklift"
        assert len(timeline.appearances) == 2
        assert len(timeline.interactions) == 1
        assert len(timeline.events) == 1


class TestGraphRetrieval:
    """Tests for GraphRetrieval class."""

    @pytest.fixture
    def mock_neo4j(self) -> AsyncMock:
        """Create mock Neo4j client."""
        mock = AsyncMock()
        mock.execute_query = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_embeddings(self) -> AsyncMock:
        """Create mock embeddings."""
        mock = AsyncMock()
        mock.embed_text = AsyncMock(return_value=[0.1] * 768)
        return mock

    @pytest.fixture
    def retrieval(
        self,
        mock_neo4j: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> GraphRetrieval:
        """Create GraphRetrieval instance."""
        return GraphRetrieval(
            neo4j_client=mock_neo4j,
            embeddings=mock_embeddings,
        )

    def test_init(self, mock_neo4j: AsyncMock) -> None:
        """Test GraphRetrieval initialization."""
        retrieval = GraphRetrieval(neo4j_client=mock_neo4j)

        assert retrieval._neo4j == mock_neo4j
        assert retrieval._embeddings is None

    @pytest.mark.asyncio
    async def test_get_temporal_context(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting temporal context."""
        # Mock finding center chunk
        mock_neo4j.execute_query.side_effect = [
            [{"idx": 5}],  # Center chunk
            [{"chunk": {"id": "chunk_004"}}, {"chunk": {"id": "chunk_005"}}, {"chunk": {"id": "chunk_006"}}],  # Chunks
            [{"entity": {"name": "forklift"}}],  # Entities
            [{"event": {"description": "Moving"}}],  # Events
            [{"from_idx": 4, "to_idx": 5}, {"from_idx": 5, "to_idx": 6}],  # Relationships
        ]

        context = await retrieval.get_temporal_context(
            stream_id="video_123",
            timestamp="00:05:30",
            window_before=2,
            window_after=2,
        )

        assert isinstance(context, TemporalContext)
        assert len(context.chunks) == 3
        assert len(context.entities) == 1
        assert len(context.events) == 1
        assert len(context.relationships) == 2

    @pytest.mark.asyncio
    async def test_get_temporal_context_no_match(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting temporal context with no matching chunk."""
        mock_neo4j.execute_query.side_effect = [
            [],  # No center chunk found
            [],  # Fallback also returns nothing
        ]

        context = await retrieval.get_temporal_context(
            stream_id="video_123",
            timestamp="99:99:99",
        )

        assert context.chunks == []
        assert context.entities == []

    @pytest.mark.asyncio
    async def test_get_entity_timeline(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting entity timeline."""
        mock_neo4j.execute_query.side_effect = [
            [{"entity": {"id": "entity_001", "name": "forklift", "type": "VEHICLE"}}],  # Entity
            [{"chunk": {"chunk_idx": 0}}, {"chunk": {"chunk_idx": 2}}],  # Appearances
            [],  # No interactions
            [{"event": {"description": "Moving pallets"}}],  # Events
        ]

        timeline = await retrieval.get_entity_timeline(
            entity_name="forklift",
            stream_id="video_123",
        )

        assert isinstance(timeline, EntityTimeline)
        assert timeline.entity["name"] == "forklift"
        assert len(timeline.appearances) == 2
        assert len(timeline.events) == 1

    @pytest.mark.asyncio
    async def test_get_entity_timeline_not_found(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting timeline for non-existent entity."""
        mock_neo4j.execute_query.return_value = []

        timeline = await retrieval.get_entity_timeline(entity_name="nonexistent")

        assert timeline.entity == {}
        assert timeline.appearances == []

    @pytest.mark.asyncio
    async def test_get_related_events(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting related events."""
        mock_neo4j.execute_query.side_effect = [
            # All events
            [
                {"event": {"id": "ev_001", "description": "Safety incident occurred", "event_type": "SAFETY_INCIDENT"}},
                {"event": {"id": "ev_002", "description": "Forklift moving", "event_type": "OPERATION"}},
            ],
            # Context for first event
            [{"chunk": {"id": "chunk_001"}}],
            # Participants for first event
            [{"name": "worker", "type": "PERSON"}],
        ]

        events = await retrieval.get_related_events(
            query="safety incident",
            stream_id="video_123",
            top_k=5,
        )

        assert len(events) == 1
        assert "safety" in events[0]["description"].lower()

    @pytest.mark.asyncio
    async def test_traverse_from_chunk(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test traversing from a chunk."""
        mock_neo4j.execute_query.return_value = [
            {
                "nodes": [
                    {"id": "chunk_001", "labels": ["VideoChunk"]},
                    {"id": "entity_001", "labels": ["Entity"]},
                ],
                "rels": [{"type": "CONTAINS", "props": {}}],
            }
        ]

        result = await retrieval.traverse_from_chunk(
            chunk_id="chunk_001",
            max_depth=2,
            rel_types=["CONTAINS", "FOLLOWS"],
        )

        assert result["start_chunk"] == "chunk_001"
        assert len(result["nodes"]) == 2
        assert len(result["relationships"]) == 1

    @pytest.mark.asyncio
    async def test_find_entity_interactions(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test finding entity interactions."""
        mock_neo4j.execute_query.return_value = [
            {
                "event": {"id": "ev_001", "description": "Worker near forklift"},
                "entity1_name": "worker",
                "entity2_name": "forklift",
            }
        ]

        interactions = await retrieval.find_entity_interactions(
            entity1="worker",
            entity2="forklift",
            stream_id="video_123",
        )

        assert len(interactions) == 1
        assert interactions[0]["entity1"] == "worker"
        assert interactions[0]["entity2"] == "forklift"

    @pytest.mark.asyncio
    async def test_get_event_context(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting event context."""
        mock_neo4j.execute_query.side_effect = [
            [{"event": {"id": "ev_001", "description": "Safety incident"}}],  # Event
            [{"entity": {"name": "worker"}, "role": "participant"}],  # Participants
            [{"chunk": {"id": "chunk_001"}}],  # Chunk
            [{"event": {"id": "ev_002", "description": "Related event"}}],  # Related
        ]

        context = await retrieval.get_event_context("ev_001")

        assert context["id"] == "ev_001"
        assert len(context["participants_detail"]) == 1
        assert context["chunk"]["id"] == "chunk_001"
        assert len(context["related_events"]) == 1

    @pytest.mark.asyncio
    async def test_get_event_context_not_found(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting context for non-existent event."""
        mock_neo4j.execute_query.return_value = []

        context = await retrieval.get_event_context("nonexistent")

        assert context == {}

    @pytest.mark.asyncio
    async def test_get_chunks_by_entity(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting chunks by entity."""
        mock_neo4j.execute_query.return_value = [
            {"chunk": {"id": "chunk_001"}, "entity_name": "forklift"},
            {"chunk": {"id": "chunk_003"}, "entity_name": "forklift"},
        ]

        chunks = await retrieval.get_chunks_by_entity(
            entity_name="forklift",
            stream_id="video_123",
            limit=10,
        )

        assert len(chunks) == 2
        assert chunks[0]["matched_entity"] == "forklift"

    @pytest.mark.asyncio
    async def test_get_chunks_by_event_type(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting chunks by event type."""
        mock_neo4j.execute_query.return_value = [
            {
                "chunk": {"id": "chunk_005"},
                "event": {"event_type": "SAFETY_INCIDENT"},
            }
        ]

        chunks = await retrieval.get_chunks_by_event_type(
            event_type="SAFETY_INCIDENT",
            stream_id="video_123",
        )

        assert len(chunks) == 1
        assert chunks[0]["event"]["event_type"] == "SAFETY_INCIDENT"

    @pytest.mark.asyncio
    async def test_get_all_entities(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting all entities."""
        mock_neo4j.execute_query.return_value = [
            {"entity": {"name": "forklift", "type": "VEHICLE"}},
            {"entity": {"name": "worker", "type": "PERSON"}},
        ]

        entities = await retrieval.get_all_entities("video_123")

        assert len(entities) == 2

    @pytest.mark.asyncio
    async def test_get_all_entities_filtered(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting entities filtered by type."""
        mock_neo4j.execute_query.return_value = [
            {"entity": {"name": "forklift", "type": "VEHICLE"}},
        ]

        entities = await retrieval.get_all_entities("video_123", entity_type="VEHICLE")

        assert len(entities) == 1
        # Verify the query included type filter
        call_args = mock_neo4j.execute_query.call_args
        assert "type" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_all_events(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting all events."""
        mock_neo4j.execute_query.return_value = [
            {"event": {"description": "Event 1", "event_type": "OPERATION"}},
            {"event": {"description": "Event 2", "event_type": "MOVEMENT"}},
        ]

        events = await retrieval.get_all_events("video_123")

        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_all_events_filtered(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting events filtered by type and severity."""
        mock_neo4j.execute_query.return_value = [
            {"event": {"event_type": "SAFETY_INCIDENT", "severity": "HIGH"}},
        ]

        events = await retrieval.get_all_events(
            "video_123",
            event_type="SAFETY_INCIDENT",
            severity="HIGH",
        )

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_get_stream_summary(
        self,
        retrieval: GraphRetrieval,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting stream summary."""
        mock_neo4j.execute_query.side_effect = [
            [{"count": 10}],  # Chunk count
            [{"type": "VEHICLE", "count": 3}, {"type": "PERSON", "count": 5}],  # Entity counts
            [{"type": "OPERATION", "count": 4}, {"type": "MOVEMENT", "count": 2}],  # Event counts
            [{"severity": "LOW", "count": 5}, {"severity": "HIGH", "count": 1}],  # Severity
        ]

        summary = await retrieval.get_stream_summary("video_123")

        assert summary["stream_id"] == "video_123"
        assert summary["chunk_count"] == 10
        assert summary["total_entities"] == 8
        assert summary["total_events"] == 6
        assert summary["entity_counts"]["VEHICLE"] == 3
        assert summary["severity_distribution"]["HIGH"] == 1
