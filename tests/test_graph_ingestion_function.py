"""Unit tests for GraphIngestionFunction RAG wrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.graph_ingestion import IngestionResult
from src.rag.functions import (
    FunctionConfig,
    FunctionStatus,
    GraphIngestionBatchInput,
    GraphIngestionBatchOutput,
    GraphIngestionFunction,
    GraphIngestionInput,
    GraphIngestionOutput,
)


class TestGraphIngestionInput:
    """Tests for GraphIngestionInput dataclass."""

    def test_create_input(self) -> None:
        """Test creating GraphIngestionInput."""
        input_data = GraphIngestionInput(
            chunk_id="chunk_001",
            stream_id="video_123",
            chunk_idx=0,
            caption="A forklift moves pallets.",
            start_time="00:00:00",
            end_time="00:01:00",
            embedding_id="emb_001",
        )

        assert input_data.chunk_id == "chunk_001"
        assert input_data.stream_id == "video_123"
        assert input_data.chunk_idx == 0
        assert input_data.caption == "A forklift moves pallets."

    def test_default_values(self) -> None:
        """Test default values."""
        input_data = GraphIngestionInput(
            chunk_id="chunk_001",
            stream_id="video_123",
            chunk_idx=0,
            caption="Test caption",
            start_time="00:00:00",
            end_time="00:01:00",
        )

        assert input_data.embedding_id == ""
        assert input_data.cv_metadata is None


class TestGraphIngestionBatchInput:
    """Tests for GraphIngestionBatchInput dataclass."""

    def test_create_batch_input(self) -> None:
        """Test creating GraphIngestionBatchInput."""
        chunks = [
            {
                "chunk_id": "chunk_001",
                "chunk_idx": 0,
                "caption": "First chunk",
                "start_time": "00:00:00",
                "end_time": "00:01:00",
            },
            {
                "chunk_id": "chunk_002",
                "chunk_idx": 1,
                "caption": "Second chunk",
                "start_time": "00:01:00",
                "end_time": "00:02:00",
            },
        ]

        input_data = GraphIngestionBatchInput(
            chunks=chunks,
            stream_id="video_123",
        )

        assert len(input_data.chunks) == 2
        assert input_data.stream_id == "video_123"


class TestGraphIngestionOutput:
    """Tests for GraphIngestionOutput dataclass."""

    def test_create_output(self) -> None:
        """Test creating GraphIngestionOutput."""
        output = GraphIngestionOutput(
            chunk_node_id="chunk_001",
            entity_count=3,
            event_count=2,
            relationship_count=5,
            entities=["forklift", "worker", "pallet"],
            events=["Moving pallets"],
        )

        assert output.chunk_node_id == "chunk_001"
        assert output.entity_count == 3
        assert output.event_count == 2
        assert output.relationship_count == 5

    def test_default_values(self) -> None:
        """Test default values."""
        output = GraphIngestionOutput(
            chunk_node_id="chunk_001",
            entity_count=0,
            event_count=0,
            relationship_count=0,
        )

        assert output.entities == []
        assert output.events == []


class TestGraphIngestionFunction:
    """Tests for GraphIngestionFunction class."""

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
        mock.generate = AsyncMock(return_value=MagicMock(text="[]"))
        return mock

    @pytest.fixture
    def mock_embeddings(self) -> AsyncMock:
        """Create mock embeddings."""
        mock = AsyncMock()
        mock.embed_documents = AsyncMock(return_value=[[0.1] * 768])
        mock.cosine_similarity = MagicMock(return_value=0.9)
        return mock

    @pytest.fixture
    def function(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> GraphIngestionFunction:
        """Create GraphIngestionFunction instance."""
        return GraphIngestionFunction(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
            embeddings=mock_embeddings,
        )

    def test_init(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test GraphIngestionFunction initialization."""
        func = GraphIngestionFunction(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
        )

        assert func.name == "graph_ingestion"
        assert func.extract_entities is True
        assert func.extract_events is True
        assert func.link_entities is True
        assert func.batch_size == 1

    def test_init_with_config(
        self,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test initialization with custom config."""
        config = FunctionConfig(name="custom_ingestion", enabled=True)
        func = GraphIngestionFunction(
            neo4j_client=mock_neo4j,
            llm=mock_llm,
            config=config,
            extract_entities=False,
            batch_size=5,
        )

        assert func.name == "custom_ingestion"
        assert func.extract_entities is False
        assert func.batch_size == 5

    def test_configure(self, function: GraphIngestionFunction) -> None:
        """Test configure method."""
        function.configure(
            extract_entities=False,
            extract_events=False,
            link_entities=False,
            batch_size=10,
        )

        assert function.extract_entities is False
        assert function.extract_events is False
        assert function.link_entities is False
        assert function.batch_size == 10

    @pytest.mark.asyncio
    async def test_execute_single(
        self,
        function: GraphIngestionFunction,
        mock_llm: AsyncMock,
    ) -> None:
        """Test executing single chunk ingestion."""
        # Mock LLM responses for entity and event extraction
        mock_llm.generate.side_effect = [
            MagicMock(text='[{"name": "forklift", "type": "VEHICLE"}]'),
            MagicMock(text='[{"description": "Moving", "event_type": "OPERATION", "severity": "LOW"}]'),
        ]

        input_data = GraphIngestionInput(
            chunk_id="chunk_001",
            stream_id="video_123",
            chunk_idx=0,
            caption="A forklift moves pallets.",
            start_time="00:00:00",
            end_time="00:01:00",
            embedding_id="emb_001",
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert isinstance(result.output, GraphIngestionOutput)
        assert result.output.chunk_node_id == "chunk_001"
        assert function.status == FunctionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_single_no_extraction(
        self,
        function: GraphIngestionFunction,
        mock_llm: AsyncMock,
    ) -> None:
        """Test executing without entity/event extraction."""
        input_data = GraphIngestionInput(
            chunk_id="chunk_001",
            stream_id="video_123",
            chunk_idx=0,
            caption="A forklift moves pallets.",
            start_time="00:00:00",
            end_time="00:01:00",
        )

        result = await function.execute(
            input_data,
            extract_entities=False,
            extract_events=False,
        )

        assert result.success is True
        assert result.output is not None
        assert result.output.entity_count == 0
        assert result.output.event_count == 0
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_batch(
        self,
        function: GraphIngestionFunction,
        mock_neo4j: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test executing batch ingestion."""
        # Mock LLM responses
        mock_llm.generate.side_effect = [
            MagicMock(text='[{"name": "forklift", "type": "VEHICLE"}]'),
            MagicMock(text='[]'),
            MagicMock(text='[{"name": "worker", "type": "PERSON"}]'),
            MagicMock(text='[]'),
        ]

        # Mock temporal links
        def mock_query_side_effect(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
            },
            {
                "chunk_id": "chunk_002",
                "chunk_idx": 1,
                "caption": "A worker appears.",
                "start_time": "00:01:00",
                "end_time": "00:02:00",
            },
        ]

        input_data = GraphIngestionBatchInput(
            chunks=chunks,
            stream_id="video_123",
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert isinstance(result.output, GraphIngestionBatchOutput)
        assert result.output.chunks_processed == 2
        assert result.output.temporal_links == 1

    @pytest.mark.asyncio
    async def test_execute_failure(
        self,
        function: GraphIngestionFunction,
        mock_llm: AsyncMock,
    ) -> None:
        """Test handling execution failure."""
        mock_llm.generate.side_effect = Exception("LLM error")

        input_data = GraphIngestionInput(
            chunk_id="chunk_001",
            stream_id="video_123",
            chunk_idx=0,
            caption="Test caption",
            start_time="00:00:00",
            end_time="00:01:00",
        )

        result = await function.execute(input_data)

        assert result.success is False
        assert result.error is not None
        assert "LLM error" in result.error
        assert function.status == FunctionStatus.FAILED

    @pytest.mark.asyncio
    async def test_reset(self, function: GraphIngestionFunction) -> None:
        """Test reset method."""
        # Set some state
        function._graph_ingestion._entity_cache["video_123"] = {"test": "value"}
        function._set_status(FunctionStatus.COMPLETED)

        await function.reset()

        assert function.status == FunctionStatus.IDLE
        assert function._graph_ingestion._entity_cache == {}

    @pytest.mark.asyncio
    async def test_clear_stream(
        self,
        function: GraphIngestionFunction,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test clearing stream data."""
        function._graph_ingestion._entity_cache["video_123"] = {"test": "value"}

        await function.clear_stream("video_123")

        assert "video_123" not in function._graph_ingestion._entity_cache
        mock_neo4j.execute_query.assert_called_once()

    def test_properties(self, function: GraphIngestionFunction) -> None:
        """Test property accessors."""
        assert function.extract_entities is True
        assert function.extract_events is True
        assert function.link_entities is True
        assert function.batch_size == 1

    @pytest.mark.asyncio
    async def test_callable(
        self,
        function: GraphIngestionFunction,
        mock_llm: AsyncMock,
    ) -> None:
        """Test function is callable."""
        mock_llm.generate.return_value = MagicMock(text="[]")

        input_data = GraphIngestionInput(
            chunk_id="chunk_001",
            stream_id="video_123",
            chunk_idx=0,
            caption="Test",
            start_time="00:00:00",
            end_time="00:01:00",
        )

        # Call function directly
        result = await function(input_data, extract_entities=False, extract_events=False)

        assert result.success is True
