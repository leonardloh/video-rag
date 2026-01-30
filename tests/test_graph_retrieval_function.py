"""Unit tests for GraphRetrievalFunction RAG wrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db.graph_retrieval import EntityTimeline, TemporalContext
from src.rag.functions import (
    FunctionConfig,
    FunctionStatus,
    GraphRetrievalFunction,
    GraphRetrievalInput,
    GraphRetrievalOutput,
    RetrievalType,
)


class TestRetrievalType:
    """Tests for RetrievalType enum."""

    def test_retrieval_types(self) -> None:
        """Test all retrieval types exist."""
        assert RetrievalType.TEMPORAL.value == "temporal"
        assert RetrievalType.ENTITY.value == "entity"
        assert RetrievalType.EVENT.value == "event"
        assert RetrievalType.TRAVERSE.value == "traverse"
        assert RetrievalType.INTERACTION.value == "interaction"


class TestGraphRetrievalInput:
    """Tests for GraphRetrievalInput dataclass."""

    def test_create_input(self) -> None:
        """Test creating GraphRetrievalInput."""
        input_data = GraphRetrievalInput(
            query="What happened with the forklift?",
            stream_id="video_123",
            retrieval_type=RetrievalType.EVENT,
            top_k=10,
        )

        assert input_data.query == "What happened with the forklift?"
        assert input_data.stream_id == "video_123"
        assert input_data.retrieval_type == RetrievalType.EVENT
        assert input_data.top_k == 10

    def test_default_values(self) -> None:
        """Test default values."""
        input_data = GraphRetrievalInput(
            query="Test query",
            stream_id="video_123",
        )

        assert input_data.retrieval_type == RetrievalType.EVENT
        assert input_data.top_k == 5
        assert input_data.timestamp is None
        assert input_data.entity_name is None
        assert input_data.entity2_name is None
        assert input_data.chunk_id is None


class TestGraphRetrievalOutput:
    """Tests for GraphRetrievalOutput dataclass."""

    def test_create_output(self) -> None:
        """Test creating GraphRetrievalOutput."""
        output = GraphRetrievalOutput(
            query="What happened?",
            context="## Events\n- Forklift moved",
            chunks=[{"id": "chunk_001"}],
            entities=[{"name": "forklift"}],
            events=[{"description": "Moving"}],
            answer="A forklift moved pallets.",
        )

        assert output.query == "What happened?"
        assert "Events" in output.context
        assert len(output.chunks) == 1
        assert len(output.entities) == 1
        assert len(output.events) == 1
        assert output.answer is not None

    def test_default_values(self) -> None:
        """Test default values."""
        output = GraphRetrievalOutput(
            query="Test",
            context="Test context",
        )

        assert output.chunks == []
        assert output.entities == []
        assert output.events == []
        assert output.answer is None


class TestGraphRetrievalFunction:
    """Tests for GraphRetrievalFunction class."""

    @pytest.fixture
    def mock_neo4j(self) -> AsyncMock:
        """Create mock Neo4j client."""
        mock = AsyncMock()
        mock.execute_query = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM."""
        mock = AsyncMock()
        mock.generate = AsyncMock(return_value=MagicMock(text="Generated answer"))
        return mock

    @pytest.fixture
    def mock_embeddings(self) -> AsyncMock:
        """Create mock embeddings."""
        mock = AsyncMock()
        mock.embed_query = AsyncMock(return_value=[0.1] * 768)
        return mock

    @pytest.fixture
    def function(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> GraphRetrievalFunction:
        """Create GraphRetrievalFunction instance."""
        return GraphRetrievalFunction(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
            embeddings=mock_embeddings,
        )

    def test_init(
        self,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test GraphRetrievalFunction initialization."""
        func = GraphRetrievalFunction(
            neo4j_client=mock_neo4j,
        )

        assert func.name == "graph_retrieval"
        assert func.top_k == 5
        assert func.generate_answer is False
        assert func.chat_history == []

    def test_init_with_config(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test initialization with custom config."""
        config = FunctionConfig(name="custom_retrieval", enabled=True)
        func = GraphRetrievalFunction(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
            config=config,
            top_k=10,
            generate_answer=True,
            chat_history=True,
        )

        assert func.name == "custom_retrieval"
        assert func.top_k == 10
        assert func.generate_answer is True

    def test_configure(self, function: GraphRetrievalFunction) -> None:
        """Test configure method."""
        function.configure(
            top_k=20,
            generate_answer=True,
            chat_history=True,
        )

        assert function.top_k == 20
        assert function.generate_answer is True

    @pytest.mark.asyncio
    async def test_execute_event_retrieval(
        self,
        function: GraphRetrievalFunction,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test event retrieval."""
        # Mock event query results
        mock_neo4j.execute_query.side_effect = [
            # Events query
            [
                {
                    "event": {
                        "id": "event_001",
                        "description": "Forklift moving pallets",
                        "event_type": "OPERATION",
                        "severity": "LOW",
                        "start_time": "00:00:05",
                        "end_time": "00:00:15",
                    }
                }
            ],
            # Chunk context query
            [
                {
                    "chunk": {
                        "id": "chunk_001",
                        "caption": "A forklift moves pallets",
                        "start_time": "00:00:00",
                        "end_time": "00:01:00",
                    }
                }
            ],
            # Participants query
            [{"name": "forklift", "type": "VEHICLE"}],
        ]

        input_data = GraphRetrievalInput(
            query="forklift moving",
            stream_id="video_123",
            retrieval_type=RetrievalType.EVENT,
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert "forklift" in result.output.context.lower() or "moving" in result.output.context.lower()
        assert function.status == FunctionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_temporal_retrieval(
        self,
        function: GraphRetrievalFunction,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test temporal retrieval."""
        # Mock temporal queries
        mock_neo4j.execute_query.side_effect = [
            # Center chunk query
            [{"idx": 2}],
            # Chunks in window
            [
                {"chunk": {"id": "chunk_001", "chunk_idx": 1, "start_time": "00:01:00", "end_time": "00:02:00", "caption": "First chunk"}},
                {"chunk": {"id": "chunk_002", "chunk_idx": 2, "start_time": "00:02:00", "end_time": "00:03:00", "caption": "Center chunk"}},
                {"chunk": {"id": "chunk_003", "chunk_idx": 3, "start_time": "00:03:00", "end_time": "00:04:00", "caption": "Third chunk"}},
            ],
            # Entities query
            [{"entity": {"name": "forklift", "type": "VEHICLE"}}],
            # Events query
            [],
            # Relationships query
            [{"from_idx": 1, "to_idx": 2}, {"from_idx": 2, "to_idx": 3}],
        ]

        input_data = GraphRetrievalInput(
            query="What happened around 00:02:30?",
            stream_id="video_123",
            retrieval_type=RetrievalType.TEMPORAL,
            timestamp="00:02:30",
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert len(result.output.chunks) > 0

    @pytest.mark.asyncio
    async def test_execute_entity_retrieval(
        self,
        function: GraphRetrievalFunction,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test entity timeline retrieval."""
        # Mock entity queries
        mock_neo4j.execute_query.side_effect = [
            # Find entity
            [
                {
                    "entity": {
                        "id": "entity_001",
                        "name": "forklift",
                        "type": "VEHICLE",
                        "first_seen": "00:00:00",
                        "last_seen": "00:05:00",
                        "occurrence_count": 5,
                    }
                }
            ],
            # Appearances
            [
                {"chunk": {"id": "chunk_001", "chunk_idx": 0, "start_time": "00:00:00", "end_time": "00:01:00", "caption": "Forklift appears"}},
            ],
            # Interactions
            [],
            # Events
            [{"event": {"description": "Moving pallets", "event_type": "OPERATION"}}],
        ]

        input_data = GraphRetrievalInput(
            query="Tell me about the forklift",
            stream_id="video_123",
            retrieval_type=RetrievalType.ENTITY,
            entity_name="forklift",
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert "forklift" in result.output.context.lower()

    @pytest.mark.asyncio
    async def test_execute_traverse_retrieval(
        self,
        function: GraphRetrievalFunction,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test graph traversal retrieval."""
        # Mock traverse query
        mock_neo4j.execute_query.return_value = [
            {
                "nodes": [
                    {"id": "chunk_001", "labels": ["VideoChunk"], "caption": "Test caption"},
                    {"id": "entity_001", "labels": ["Entity"], "name": "forklift", "type": "VEHICLE"},
                ],
                "rels": [{"type": "CONTAINS", "props": {}}],
            }
        ]

        input_data = GraphRetrievalInput(
            query="What's connected to this chunk?",
            stream_id="video_123",
            retrieval_type=RetrievalType.TRAVERSE,
            chunk_id="chunk_001",
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_execute_traverse_no_chunk_id(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test traverse retrieval without chunk_id."""
        input_data = GraphRetrievalInput(
            query="Traverse",
            stream_id="video_123",
            retrieval_type=RetrievalType.TRAVERSE,
            chunk_id=None,
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert "No chunk_id" in result.output.context

    @pytest.mark.asyncio
    async def test_execute_interaction_retrieval(
        self,
        function: GraphRetrievalFunction,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test entity interaction retrieval."""
        mock_neo4j.execute_query.return_value = [
            {
                "event": {
                    "id": "event_001",
                    "description": "Forklift picks up pallet",
                    "event_type": "INTERACTION",
                    "severity": "LOW",
                    "start_time": "00:00:10",
                },
                "entity1_name": "forklift",
                "entity2_name": "pallet",
            }
        ]

        input_data = GraphRetrievalInput(
            query="How did forklift interact with pallet?",
            stream_id="video_123",
            retrieval_type=RetrievalType.INTERACTION,
            entity_name="forklift",
            entity2_name="pallet",
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert "forklift" in result.output.context.lower()

    @pytest.mark.asyncio
    async def test_execute_interaction_missing_entities(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test interaction retrieval with missing entity names."""
        input_data = GraphRetrievalInput(
            query="Interactions",
            stream_id="video_123",
            retrieval_type=RetrievalType.INTERACTION,
            entity_name="forklift",
            entity2_name=None,
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert "Two entity names required" in result.output.context

    @pytest.mark.asyncio
    async def test_execute_with_answer_generation(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        """Test retrieval with answer generation."""
        function = GraphRetrievalFunction(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
            embeddings=mock_embeddings,
            generate_answer=True,
        )

        # Mock event query
        mock_neo4j.execute_query.return_value = []

        input_data = GraphRetrievalInput(
            query="What happened?",
            stream_id="video_123",
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert result.output.answer == "Generated answer"
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_chat_history(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test retrieval with chat history."""
        function = GraphRetrievalFunction(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
            generate_answer=True,
            chat_history=True,
        )

        mock_neo4j.execute_query.return_value = []

        # First query
        input_data1 = GraphRetrievalInput(
            query="What happened?",
            stream_id="video_123",
        )
        await function.execute(input_data1)

        # Second query
        input_data2 = GraphRetrievalInput(
            query="Tell me more",
            stream_id="video_123",
        )
        await function.execute(input_data2)

        # Check history
        history = function.chat_history
        assert len(history) == 4  # 2 user + 2 assistant messages

    @pytest.mark.asyncio
    async def test_execute_failure(
        self,
        function: GraphRetrievalFunction,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test handling execution failure."""
        mock_neo4j.execute_query.side_effect = Exception("Database error")

        input_data = GraphRetrievalInput(
            query="Test query",
            stream_id="video_123",
        )

        result = await function.execute(input_data)

        assert result.success is False
        assert result.error is not None
        assert "Database error" in result.error
        assert function.status == FunctionStatus.FAILED

    @pytest.mark.asyncio
    async def test_reset(self, function: GraphRetrievalFunction) -> None:
        """Test reset method."""
        function._history = [{"role": "user", "content": "test"}]
        function._set_status(FunctionStatus.COMPLETED)

        await function.reset()

        assert function.status == FunctionStatus.IDLE
        assert function._history == []

    def test_clear_history(self, function: GraphRetrievalFunction) -> None:
        """Test clearing chat history."""
        function._history = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]

        function.clear_history()

        assert function._history == []

    @pytest.mark.asyncio
    async def test_get_stream_summary(
        self,
        function: GraphRetrievalFunction,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test getting stream summary."""
        mock_neo4j.execute_query.side_effect = [
            # Chunk count
            [{"count": 10}],
            # Entity counts
            [{"type": "PERSON", "count": 5}, {"type": "VEHICLE", "count": 3}],
            # Event counts
            [{"type": "OPERATION", "count": 8}],
            # Severity counts
            [{"severity": "LOW", "count": 6}, {"severity": "HIGH", "count": 2}],
        ]

        summary = await function.get_stream_summary("video_123")

        assert summary["stream_id"] == "video_123"
        assert summary["chunk_count"] == 10
        assert summary["total_entities"] == 8
        assert summary["total_events"] == 8

    def test_properties(self, function: GraphRetrievalFunction) -> None:
        """Test property accessors."""
        assert function.top_k == 5
        assert function.generate_answer is False
        assert function.chat_history == []

    @pytest.mark.asyncio
    async def test_callable(
        self,
        function: GraphRetrievalFunction,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test function is callable."""
        mock_neo4j.execute_query.return_value = []

        input_data = GraphRetrievalInput(
            query="Test",
            stream_id="video_123",
        )

        # Call function directly
        result = await function(input_data)

        assert result.success is True


class TestGraphRetrievalFunctionFormatters:
    """Tests for context formatting methods."""

    @pytest.fixture
    def mock_neo4j(self) -> AsyncMock:
        """Create mock Neo4j client."""
        return AsyncMock()

    @pytest.fixture
    def function(self, mock_neo4j: AsyncMock) -> GraphRetrievalFunction:
        """Create GraphRetrievalFunction instance."""
        return GraphRetrievalFunction(neo4j_client=mock_neo4j)

    def test_format_temporal_context_empty(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test formatting empty temporal context."""
        context = MagicMock()
        context.chunks = []
        context.entities = []
        context.events = []

        result = function._format_temporal_context(context)

        assert result == "No temporal context found."

    def test_format_temporal_context_with_data(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test formatting temporal context with data."""
        context = MagicMock()
        context.chunks = [
            {"start_time": "00:00:00", "end_time": "00:01:00", "caption": "First chunk"}
        ]
        context.entities = [{"name": "forklift", "type": "VEHICLE"}]
        context.events = [
            {"description": "Moving", "event_type": "OPERATION", "severity": "LOW"}
        ]

        result = function._format_temporal_context(context)

        assert "Video Chunks" in result
        assert "Entities Present" in result
        assert "Events" in result
        assert "forklift" in result

    def test_format_entity_timeline_empty(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test formatting empty entity timeline."""
        timeline = MagicMock()
        timeline.entity = {}

        result = function._format_entity_timeline(timeline)

        assert result == "Entity not found."

    def test_format_entity_timeline_with_data(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test formatting entity timeline with data."""
        timeline = MagicMock()
        timeline.entity = {
            "name": "forklift",
            "type": "VEHICLE",
            "first_seen": "00:00:00",
            "last_seen": "00:05:00",
            "occurrence_count": 5,
        }
        timeline.appearances = [
            {"start_time": "00:00:00", "end_time": "00:01:00", "caption": "Forklift appears"}
        ]
        timeline.events = [{"description": "Moving", "event_type": "OPERATION"}]

        result = function._format_entity_timeline(timeline)

        assert "forklift" in result
        assert "VEHICLE" in result
        assert "Appearances" in result
        assert "Events Participated In" in result

    def test_format_events_context_empty(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test formatting empty events."""
        result = function._format_events_context([])

        assert result == "No related events found."

    def test_format_events_context_with_data(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test formatting events with data."""
        events = [
            {
                "description": "Forklift moving",
                "event_type": "OPERATION",
                "severity": "LOW",
                "start_time": "00:00:05",
                "end_time": "00:00:15",
                "chunk_context": {"caption": "A forklift moves"},
                "participants_detail": [{"name": "forklift"}],
            }
        ]

        result = function._format_events_context(events)

        assert "Related Events" in result
        assert "OPERATION" in result
        assert "Forklift moving" in result

    def test_format_subgraph_context_empty(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test formatting empty subgraph."""
        result = function._format_subgraph_context({"nodes": [], "depth": 2})

        assert result == "No graph context found."

    def test_format_interactions_context_empty(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test formatting empty interactions."""
        result = function._format_interactions_context([], "entity1", "entity2")

        assert "No interactions found" in result

    def test_format_interactions_context_with_data(
        self,
        function: GraphRetrievalFunction,
    ) -> None:
        """Test formatting interactions with data."""
        interactions = [
            {
                "description": "Forklift picks up pallet",
                "event_type": "INTERACTION",
                "severity": "LOW",
                "start_time": "00:00:10",
            }
        ]

        result = function._format_interactions_context(
            interactions, "forklift", "pallet"
        )

        assert "Interactions" in result
        assert "forklift" in result
        assert "pallet" in result
