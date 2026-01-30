"""Graph ingestion function for RAG pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .base import FunctionConfig, FunctionResult, FunctionStatus, RAGFunction

if TYPE_CHECKING:
    from ...db.graph_ingestion import GraphIngestion
    from ...db.neo4j_client import Neo4jClient
    from ...models.gemini.gemini_embeddings import GeminiEmbeddings
    from ...models.gemini.gemini_llm import GeminiLLM

logger = logging.getLogger(__name__)


@dataclass
class GraphIngestionInput:
    """Input for graph ingestion function."""

    chunk_id: str
    stream_id: str
    chunk_idx: int
    caption: str
    start_time: str
    end_time: str
    embedding_id: str = ""
    cv_metadata: Optional[dict[str, Any]] = None


@dataclass
class GraphIngestionBatchInput:
    """Input for batch graph ingestion."""

    chunks: list[dict[str, Any]]
    stream_id: str


@dataclass
class GraphIngestionOutput:
    """Output from graph ingestion function."""

    chunk_node_id: str
    entity_count: int
    event_count: int
    relationship_count: int
    entities: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


@dataclass
class GraphIngestionBatchOutput:
    """Output from batch graph ingestion."""

    stream_id: str
    chunks_processed: int
    total_entities: int
    total_events: int
    total_relationships: int
    temporal_links: int = 0


class GraphIngestionFunction(
    RAGFunction[GraphIngestionInput | GraphIngestionBatchInput, GraphIngestionOutput | GraphIngestionBatchOutput]
):
    """Graph ingestion function for video chunks.

    This function:
    1. Creates VideoChunk nodes in Neo4j
    2. Extracts entities from captions using LLM
    3. Extracts events from captions using LLM
    4. Creates relationships between nodes
    5. Links entities across chunks
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        llm: GeminiLLM,
        embeddings: Optional[GeminiEmbeddings] = None,
        config: FunctionConfig | None = None,
        extract_entities: bool = True,
        extract_events: bool = True,
        link_entities: bool = True,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize graph ingestion function.

        Args:
            neo4j_client: Neo4j database client
            llm: Gemini LLM for entity/event extraction
            embeddings: Optional embedding generator for entity linking
            config: Function configuration
            extract_entities: Whether to extract entities
            extract_events: Whether to extract events
            link_entities: Whether to link entities across chunks
            batch_size: Number of chunks to process before linking
        """
        super().__init__(config or FunctionConfig(name="graph_ingestion"))
        self._neo4j_client = neo4j_client
        self._llm = llm
        self._embeddings = embeddings
        self._extract_entities = extract_entities
        self._extract_events = extract_events
        self._link_entities = link_entities
        self._batch_size = batch_size

        # Lazy import to avoid circular imports
        from ...db.graph_ingestion import GraphIngestion

        self._graph_ingestion = GraphIngestion(
            neo4j_client=neo4j_client,
            llm=llm,
            embeddings=embeddings,
        )

    @property
    def extract_entities(self) -> bool:
        """Get extract_entities setting."""
        return self._extract_entities

    @property
    def extract_events(self) -> bool:
        """Get extract_events setting."""
        return self._extract_events

    @property
    def link_entities(self) -> bool:
        """Get link_entities setting."""
        return self._link_entities

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self._batch_size

    def configure(self, **kwargs: Any) -> None:
        """
        Update function configuration.

        Args:
            **kwargs: Configuration parameters
        """
        super().configure(**kwargs)

        if "extract_entities" in kwargs:
            self._extract_entities = kwargs["extract_entities"]
        if "extract_events" in kwargs:
            self._extract_events = kwargs["extract_events"]
        if "link_entities" in kwargs:
            self._link_entities = kwargs["link_entities"]
        if "batch_size" in kwargs:
            self._batch_size = kwargs["batch_size"]

    async def execute(
        self,
        input_data: GraphIngestionInput | GraphIngestionBatchInput,
        **kwargs: Any,
    ) -> FunctionResult[GraphIngestionOutput | GraphIngestionBatchOutput]:
        """
        Execute graph ingestion.

        Args:
            input_data: Single chunk or batch of chunks to ingest
            **kwargs: Additional parameters

        Returns:
            FunctionResult with ingestion summary
        """
        self._set_status(FunctionStatus.RUNNING)

        # Get parameters from kwargs or use defaults
        extract_entities = kwargs.get("extract_entities", self._extract_entities)
        extract_events = kwargs.get("extract_events", self._extract_events)
        link_entities = kwargs.get("link_entities", self._link_entities)

        try:
            if isinstance(input_data, GraphIngestionBatchInput):
                return await self._execute_batch(
                    input_data,
                    extract_entities=extract_entities,
                    extract_events=extract_events,
                    link_entities=link_entities,
                )
            else:
                return await self._execute_single(
                    input_data,
                    extract_entities=extract_entities,
                    extract_events=extract_events,
                )

        except Exception as e:
            logger.error(f"Graph ingestion failed: {e}")
            self._set_status(FunctionStatus.FAILED)
            return FunctionResult.fail(str(e))

    async def _execute_single(
        self,
        input_data: GraphIngestionInput,
        extract_entities: bool,
        extract_events: bool,
    ) -> FunctionResult[GraphIngestionOutput | GraphIngestionBatchOutput]:
        """Execute single chunk ingestion."""
        result = await self._graph_ingestion.ingest_chunk(
            chunk_id=input_data.chunk_id,
            stream_id=input_data.stream_id,
            chunk_idx=input_data.chunk_idx,
            caption=input_data.caption,
            start_time=input_data.start_time,
            end_time=input_data.end_time,
            embedding_id=input_data.embedding_id,
            cv_metadata=input_data.cv_metadata,
            extract_entities=extract_entities,
            extract_events=extract_events,
        )

        self._set_status(FunctionStatus.COMPLETED)
        return FunctionResult.ok(
            GraphIngestionOutput(
                chunk_node_id=result.chunk_node_id,
                entity_count=result.entity_count,
                event_count=result.event_count,
                relationship_count=result.relationship_count,
                entities=result.entities,
                events=result.events,
            ),
            stream_id=input_data.stream_id,
            chunk_idx=input_data.chunk_idx,
        )

    async def _execute_batch(
        self,
        input_data: GraphIngestionBatchInput,
        extract_entities: bool,
        extract_events: bool,
        link_entities: bool,
    ) -> FunctionResult[GraphIngestionOutput | GraphIngestionBatchOutput]:
        """Execute batch chunk ingestion."""
        result = await self._graph_ingestion.ingest_batch(
            chunks=input_data.chunks,
            stream_id=input_data.stream_id,
            extract_entities=extract_entities,
            extract_events=extract_events,
        )

        # Link entities across chunks if enabled
        entity_links = 0
        if link_entities and self._embeddings:
            entity_links = await self._graph_ingestion.link_entities_across_chunks(
                stream_id=input_data.stream_id,
            )

        self._set_status(FunctionStatus.COMPLETED)
        return FunctionResult.ok(
            GraphIngestionBatchOutput(
                stream_id=result["stream_id"],
                chunks_processed=result["chunks_processed"],
                total_entities=result["total_entities"],
                total_events=result["total_events"],
                total_relationships=result["total_relationships"] + entity_links,
                temporal_links=result["temporal_links"],
            ),
            entity_links_created=entity_links,
        )

    async def reset(self) -> None:
        """Reset function state and clear entity cache."""
        await super().reset()
        self._graph_ingestion.clear_cache()

    async def clear_stream(self, stream_id: str) -> None:
        """
        Clear all graph data for a stream.

        Args:
            stream_id: Stream to clear
        """
        # Clear cache
        self._graph_ingestion.clear_cache(stream_id)

        # Delete all nodes for this stream
        await self._neo4j_client.execute_query(
            """
            MATCH (n)
            WHERE n.stream_id = $stream_id
            DETACH DELETE n
            """,
            {"stream_id": stream_id},
        )

        logger.info(f"Cleared graph data for stream: {stream_id}")
