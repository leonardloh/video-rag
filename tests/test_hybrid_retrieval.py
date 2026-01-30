"""Unit tests for Hybrid Retrieval."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.hybrid_retrieval import (
    HybridConfig,
    HybridRetriever,
    RetrievalMode,
    RetrievalResult,
)


class TestRetrievalMode:
    """Tests for RetrievalMode enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert RetrievalMode.VECTOR_ONLY.value == "vector_only"
        assert RetrievalMode.GRAPH_ONLY.value == "graph_only"
        assert RetrievalMode.HYBRID.value == "hybrid"


class TestHybridConfig:
    """Tests for HybridConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = HybridConfig()
        assert config.mode == RetrievalMode.HYBRID
        assert config.vector_weight == 0.6
        assert config.graph_weight == 0.4
        assert config.top_k == 5
        assert config.rerank is True
        assert config.temporal_boost == 1.2
        assert config.entity_boost == 1.1

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = HybridConfig(
            mode=RetrievalMode.VECTOR_ONLY,
            vector_weight=0.8,
            graph_weight=0.2,
            top_k=10,
            rerank=False,
        )
        assert config.mode == RetrievalMode.VECTOR_ONLY
        assert config.vector_weight == 0.8
        assert config.top_k == 10
        assert config.rerank is False


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a RetrievalResult."""
        result = RetrievalResult(
            chunk_id="chunk_001",
            text="Test caption",
            score=0.95,
            source="vector",
            metadata={"chunk_idx": 0},
            entities=["forklift"],
            events=["moving"],
        )

        assert result.chunk_id == "chunk_001"
        assert result.text == "Test caption"
        assert result.score == 0.95
        assert result.source == "vector"
        assert "forklift" in result.entities

    def test_default_values(self) -> None:
        """Test default values."""
        result = RetrievalResult(
            chunk_id="chunk_001",
            text="Test",
            score=0.5,
            source="graph",
        )

        assert result.metadata == {}
        assert result.entities == []
        assert result.events == []


class TestHybridRetriever:
    """Tests for HybridRetriever class."""

    @pytest.fixture
    def mock_milvus(self) -> AsyncMock:
        """Create mock Milvus client."""
        mock = AsyncMock()
        mock.search_by_stream = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_neo4j(self) -> AsyncMock:
        """Create mock Neo4j client."""
        mock = AsyncMock()
        mock.execute_query = AsyncMock(return_value=[])
        return mock

    @pytest.fixture
    def mock_embeddings(self) -> AsyncMock:
        """Create mock embeddings client."""
        mock = AsyncMock()
        mock.embed_query = AsyncMock(return_value=[0.1] * 768)
        return mock

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM client."""
        mock = AsyncMock()
        mock.generate = AsyncMock(return_value=MagicMock(text="0, 1, 2"))
        return mock

    @pytest.fixture
    def retriever(
        self,
        mock_milvus: AsyncMock,
        mock_neo4j: AsyncMock,
        mock_embeddings: AsyncMock,
        mock_llm: AsyncMock,
    ) -> HybridRetriever:
        """Create HybridRetriever instance."""
        return HybridRetriever(
            milvus_client=mock_milvus,
            neo4j_client=mock_neo4j,
            embeddings=mock_embeddings,
            llm=mock_llm,
            config=HybridConfig(rerank=False),  # Disable rerank for simpler tests
        )

    def test_init(self, mock_milvus: AsyncMock, mock_neo4j: AsyncMock) -> None:
        """Test HybridRetriever initialization."""
        retriever = HybridRetriever(
            milvus_client=mock_milvus,
            neo4j_client=mock_neo4j,
        )

        assert retriever._milvus == mock_milvus
        assert retriever._neo4j == mock_neo4j
        assert retriever._config.mode == RetrievalMode.HYBRID

    def test_init_with_config(self, mock_milvus: AsyncMock) -> None:
        """Test initialization with custom config."""
        config = HybridConfig(mode=RetrievalMode.VECTOR_ONLY, top_k=10)
        retriever = HybridRetriever(
            milvus_client=mock_milvus,
            config=config,
        )

        assert retriever._config.mode == RetrievalMode.VECTOR_ONLY
        assert retriever._config.top_k == 10

    @pytest.mark.asyncio
    async def test_retrieve_vector_only(
        self,
        mock_milvus: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        """Test vector-only retrieval."""
        # Mock vector search results
        mock_doc = MagicMock()
        mock_doc.id = "chunk_001"
        mock_doc.text = "Forklift moving"
        mock_doc.chunk_idx = 0
        mock_doc.start_time = "00:00:00"
        mock_doc.end_time = "00:01:00"
        mock_doc.stream_id = "video_123"

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.score = 0.95

        mock_milvus.search_by_stream.return_value = [mock_result]

        retriever = HybridRetriever(
            milvus_client=mock_milvus,
            embeddings=mock_embeddings,
            config=HybridConfig(mode=RetrievalMode.VECTOR_ONLY, rerank=False),
        )

        results = await retriever.retrieve("forklift", "video_123")

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_001"
        assert results[0].source == "vector"
        mock_milvus.search_by_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_graph_only(
        self,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test graph-only retrieval."""
        # Mock graph search results
        mock_neo4j.execute_query.side_effect = [
            # Entity query results
            [{
                "chunk_id": "chunk_001",
                "text": "Forklift moving",
                "chunk_idx": 0,
                "start_time": "00:00:00",
                "end_time": "00:01:00",
                "entities": ["forklift"],
            }],
            # Event query results
            [],
        ]

        retriever = HybridRetriever(
            neo4j_client=mock_neo4j,
            config=HybridConfig(mode=RetrievalMode.GRAPH_ONLY, rerank=False),
        )

        results = await retriever.retrieve("forklift", "video_123")

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_001"
        assert results[0].source == "graph"
        assert "forklift" in results[0].entities

    @pytest.mark.asyncio
    async def test_retrieve_hybrid(
        self,
        retriever: HybridRetriever,
        mock_milvus: AsyncMock,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test hybrid retrieval."""
        # Mock vector results
        mock_doc = MagicMock()
        mock_doc.id = "chunk_001"
        mock_doc.text = "Forklift moving"
        mock_doc.chunk_idx = 0
        mock_doc.start_time = "00:00:00"
        mock_doc.end_time = "00:01:00"
        mock_doc.stream_id = "video_123"

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.score = 0.9

        mock_milvus.search_by_stream.return_value = [mock_result]

        # Mock graph results
        mock_neo4j.execute_query.side_effect = [
            # Entity query
            [{
                "chunk_id": "chunk_002",
                "text": "Worker walking",
                "chunk_idx": 1,
                "start_time": "00:01:00",
                "end_time": "00:02:00",
                "entities": ["worker"],
            }],
            # Event query
            [],
        ]

        results = await retriever.retrieve("forklift worker", "video_123")

        # Should have results from both sources
        assert len(results) >= 1
        mock_milvus.search_by_stream.assert_called_once()
        mock_neo4j.execute_query.assert_called()

    @pytest.mark.asyncio
    async def test_retrieve_no_clients(self) -> None:
        """Test retrieval with no clients configured."""
        retriever = HybridRetriever()

        results = await retriever.retrieve("test", "video_123")

        assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_failure(
        self,
        mock_milvus: AsyncMock,
        mock_embeddings: AsyncMock,
    ) -> None:
        """Test handling of vector search failure."""
        mock_embeddings.embed_query.side_effect = Exception("API Error")

        retriever = HybridRetriever(
            milvus_client=mock_milvus,
            embeddings=mock_embeddings,
            config=HybridConfig(mode=RetrievalMode.VECTOR_ONLY, rerank=False),
        )

        results = await retriever.retrieve("test", "video_123")

        assert results == []

    @pytest.mark.asyncio
    async def test_graph_search_failure(
        self,
        mock_neo4j: AsyncMock,
    ) -> None:
        """Test handling of graph search failure."""
        mock_neo4j.execute_query.side_effect = Exception("DB Error")

        retriever = HybridRetriever(
            neo4j_client=mock_neo4j,
            config=HybridConfig(mode=RetrievalMode.GRAPH_ONLY, rerank=False),
        )

        results = await retriever.retrieve("test", "video_123")

        assert results == []

    def test_merge_results(self, retriever: HybridRetriever) -> None:
        """Test merging vector and graph results."""
        vector_results = [
            RetrievalResult(
                chunk_id="chunk_001",
                text="Text 1",
                score=0.9,
                source="vector",
                metadata={"chunk_idx": 0},
            ),
        ]

        graph_results = [
            RetrievalResult(
                chunk_id="chunk_001",  # Same chunk
                text="Text 1",
                score=0.8,
                source="graph",
                metadata={"chunk_idx": 0},
                entities=["forklift"],
            ),
            RetrievalResult(
                chunk_id="chunk_002",  # Different chunk
                text="Text 2",
                score=0.7,
                source="graph",
                metadata={"chunk_idx": 1},
            ),
        ]

        merged = retriever._merge_results(vector_results, graph_results, top_k=5)

        assert len(merged) == 2

        # chunk_001 should have combined score and be marked as "both"
        chunk_001 = next(r for r in merged if r.chunk_id == "chunk_001")
        assert chunk_001.source == "both"
        assert chunk_001.score > 0.5  # Combined weighted score
        assert "forklift" in chunk_001.entities

    def test_apply_temporal_boost(self, retriever: HybridRetriever) -> None:
        """Test temporal boost for adjacent chunks."""
        results = [
            RetrievalResult(
                chunk_id="chunk_001",
                text="Text 1",
                score=0.9,
                source="vector",
                metadata={"chunk_idx": 0},
            ),
            RetrievalResult(
                chunk_id="chunk_002",
                text="Text 2",
                score=0.5,
                source="vector",
                metadata={"chunk_idx": 1},  # Adjacent to chunk_001
            ),
            RetrievalResult(
                chunk_id="chunk_010",
                text="Text 10",
                score=0.6,
                source="vector",
                metadata={"chunk_idx": 10},  # Not adjacent
            ),
        ]

        boosted = retriever._apply_temporal_boost(results)

        # chunk_002 should be boosted because it's adjacent to chunk_001
        chunk_002 = next(r for r in boosted if r.chunk_id == "chunk_002")
        assert chunk_002.score > 0.5  # Should be boosted

    @pytest.mark.asyncio
    async def test_rerank(
        self,
        mock_milvus: AsyncMock,
        mock_embeddings: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Test LLM reranking."""
        # Mock LLM to reverse order
        mock_llm.generate.return_value = MagicMock(text="2, 1, 0")

        retriever = HybridRetriever(
            milvus_client=mock_milvus,
            embeddings=mock_embeddings,
            llm=mock_llm,
            config=HybridConfig(rerank=True),
        )

        results = [
            RetrievalResult(chunk_id="chunk_0", text="Text 0", score=0.9, source="vector"),
            RetrievalResult(chunk_id="chunk_1", text="Text 1", score=0.8, source="vector"),
            RetrievalResult(chunk_id="chunk_2", text="Text 2", score=0.7, source="vector"),
        ]

        reranked = await retriever._rerank("test query", results)

        # Order should be reversed
        assert reranked[0].chunk_id == "chunk_2"
        assert reranked[1].chunk_id == "chunk_1"
        assert reranked[2].chunk_id == "chunk_0"

    @pytest.mark.asyncio
    async def test_rerank_failure(
        self,
        mock_llm: AsyncMock,
    ) -> None:
        """Test reranking failure handling."""
        mock_llm.generate.side_effect = Exception("LLM Error")

        retriever = HybridRetriever(
            llm=mock_llm,
            config=HybridConfig(rerank=True),
        )

        results = [
            RetrievalResult(chunk_id="chunk_0", text="Text 0", score=0.9, source="vector"),
            RetrievalResult(chunk_id="chunk_1", text="Text 1", score=0.8, source="vector"),
        ]

        reranked = await retriever._rerank("test query", results)

        # Should return original order on failure
        assert reranked[0].chunk_id == "chunk_0"
        assert reranked[1].chunk_id == "chunk_1"

    @pytest.mark.asyncio
    async def test_get_context_for_query(
        self,
        retriever: HybridRetriever,
        mock_milvus: AsyncMock,
    ) -> None:
        """Test getting formatted context string."""
        # Mock vector results
        mock_doc = MagicMock()
        mock_doc.id = "chunk_001"
        mock_doc.text = "Forklift moving pallets"
        mock_doc.chunk_idx = 0
        mock_doc.start_time = "00:00:00"
        mock_doc.end_time = "00:01:00"
        mock_doc.stream_id = "video_123"

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.score = 0.9

        mock_milvus.search_by_stream.return_value = [mock_result]

        context = await retriever.get_context_for_query("forklift", "video_123")

        assert "Forklift moving pallets" in context
        assert "00:00:00" in context

    @pytest.mark.asyncio
    async def test_get_context_for_query_no_results(
        self,
        retriever: HybridRetriever,
        mock_milvus: AsyncMock,
    ) -> None:
        """Test context generation with no results."""
        mock_milvus.search_by_stream.return_value = []

        context = await retriever.get_context_for_query("nonexistent", "video_123")

        assert "No relevant context found" in context

    def test_config_property(self, retriever: HybridRetriever) -> None:
        """Test config property."""
        assert isinstance(retriever.config, HybridConfig)

    def test_update_config(self, retriever: HybridRetriever) -> None:
        """Test updating configuration."""
        retriever.update_config(top_k=20, vector_weight=0.8)

        assert retriever._config.top_k == 20
        assert retriever._config.vector_weight == 0.8
