"""Hybrid retrieval combining vector and graph search for CA-RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from db.milvus_client import MilvusClient
    from db.neo4j_client import Neo4jClient
    from models.gemini.gemini_embeddings import GeminiEmbeddings
    from models.gemini.gemini_llm import GeminiLLM

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Retrieval strategy mode."""

    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""

    mode: RetrievalMode = RetrievalMode.HYBRID
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    top_k: int = 5
    rerank: bool = True
    temporal_boost: float = 1.2
    entity_boost: float = 1.1


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""

    chunk_id: str
    text: str
    score: float
    source: str  # "vector", "graph", or "both"
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


class HybridRetriever:
    """Combined vector and graph retrieval for CA-RAG."""

    def __init__(
        self,
        milvus_client: Optional[MilvusClient] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        embeddings: Optional[GeminiEmbeddings] = None,
        llm: Optional[GeminiLLM] = None,
        config: Optional[HybridConfig] = None,
    ) -> None:
        """
        Initialize hybrid retriever.

        Args:
            milvus_client: Milvus vector database client
            neo4j_client: Neo4j graph database client
            embeddings: Embedding generator for query processing
            llm: LLM for optional reranking
            config: Retrieval configuration
        """
        self._milvus = milvus_client
        self._neo4j = neo4j_client
        self._embeddings = embeddings
        self._llm = llm
        self._config = config or HybridConfig()

    async def retrieve(
        self,
        query: str,
        stream_id: str,
        top_k: Optional[int] = None,
        mode: Optional[RetrievalMode] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid approach.

        Args:
            query: Natural language query
            stream_id: Stream to search
            top_k: Number of results (overrides config)
            mode: Retrieval mode (overrides config)

        Returns:
            List of retrieval results sorted by relevance
        """
        top_k = top_k or self._config.top_k
        mode = mode or self._config.mode

        vector_results: list[RetrievalResult] = []
        graph_results: list[RetrievalResult] = []

        # Vector search
        if mode in (RetrievalMode.VECTOR_ONLY, RetrievalMode.HYBRID):
            vector_results = await self._vector_search(query, stream_id, top_k * 2)

        # Graph search
        if mode in (RetrievalMode.GRAPH_ONLY, RetrievalMode.HYBRID):
            graph_results = await self._graph_search(query, stream_id, top_k * 2)

        # Merge results
        if mode == RetrievalMode.VECTOR_ONLY:
            merged = vector_results[:top_k]
        elif mode == RetrievalMode.GRAPH_ONLY:
            merged = graph_results[:top_k]
        else:
            merged = self._merge_results(vector_results, graph_results, top_k)

        # Apply temporal boost
        merged = self._apply_temporal_boost(merged)

        # Optional reranking
        if self._config.rerank and self._llm and len(merged) > 1:
            merged = await self._rerank(query, merged)

        return merged[:top_k]

    async def _vector_search(
        self,
        query: str,
        stream_id: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Search Milvus with query embedding."""
        if not self._milvus or not self._embeddings:
            logger.warning("Vector search unavailable: missing milvus or embeddings client")
            return []

        try:
            # Generate query embedding
            query_embedding = await self._embeddings.embed_query(query)

            # Search Milvus
            results = await self._milvus.search_by_stream(
                query_embedding=query_embedding,
                stream_id=stream_id,
                top_k=top_k,
            )

            return [
                RetrievalResult(
                    chunk_id=r.document.id,
                    text=r.document.text,
                    score=r.score,
                    source="vector",
                    metadata={
                        "chunk_idx": r.document.chunk_idx,
                        "start_time": r.document.start_time,
                        "end_time": r.document.end_time,
                        "stream_id": r.document.stream_id,
                    },
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _graph_search(
        self,
        query: str,
        stream_id: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Search Neo4j for entities and events related to query."""
        if not self._neo4j:
            logger.warning("Graph search unavailable: missing neo4j client")
            return []

        try:
            results: list[RetrievalResult] = []

            # Extract keywords from query for entity/event matching
            keywords = query.lower().split()

            # Search for entities mentioned in query
            entity_query = """
            MATCH (e:Entity {stream_id: $stream_id})<-[:CONTAINS]-(c:VideoChunk)
            WHERE any(kw IN $keywords WHERE toLower(e.name) CONTAINS kw)
            RETURN c.id as chunk_id, c.caption as text, c.chunk_idx as chunk_idx,
                   c.start_time as start_time, c.end_time as end_time,
                   collect(e.name) as entities
            ORDER BY c.chunk_idx
            LIMIT $limit
            """

            entity_results = await self._neo4j.execute_query(
                entity_query,
                {"stream_id": stream_id, "keywords": keywords, "limit": top_k},
            )

            for r in entity_results:
                # Score based on number of matching entities
                entity_count = len(r.get("entities", []))
                score = min(1.0, 0.5 + entity_count * 0.1)

                results.append(
                    RetrievalResult(
                        chunk_id=r["chunk_id"],
                        text=r.get("text", ""),
                        score=score,
                        source="graph",
                        metadata={
                            "chunk_idx": r.get("chunk_idx", 0),
                            "start_time": r.get("start_time", ""),
                            "end_time": r.get("end_time", ""),
                            "stream_id": stream_id,
                        },
                        entities=r.get("entities", []),
                    )
                )

            # Search for events related to query
            event_query = """
            MATCH (ev:Event {stream_id: $stream_id})-[:OCCURS_IN]->(c:VideoChunk)
            WHERE any(kw IN $keywords WHERE toLower(ev.description) CONTAINS kw
                  OR toLower(ev.event_type) CONTAINS kw)
            RETURN c.id as chunk_id, c.caption as text, c.chunk_idx as chunk_idx,
                   c.start_time as start_time, c.end_time as end_time,
                   collect(ev.description) as events
            ORDER BY c.chunk_idx
            LIMIT $limit
            """

            event_results = await self._neo4j.execute_query(
                event_query,
                {"stream_id": stream_id, "keywords": keywords, "limit": top_k},
            )

            for r in event_results:
                # Check if already in results
                existing = next(
                    (res for res in results if res.chunk_id == r["chunk_id"]),
                    None,
                )

                if existing:
                    # Boost existing result
                    existing.score = min(1.0, existing.score + 0.2)
                    existing.events = r.get("events", [])
                else:
                    event_count = len(r.get("events", []))
                    score = min(1.0, 0.4 + event_count * 0.15)

                    results.append(
                        RetrievalResult(
                            chunk_id=r["chunk_id"],
                            text=r.get("text", ""),
                            score=score,
                            source="graph",
                            metadata={
                                "chunk_idx": r.get("chunk_idx", 0),
                                "start_time": r.get("start_time", ""),
                                "end_time": r.get("end_time", ""),
                                "stream_id": stream_id,
                            },
                            events=r.get("events", []),
                        )
                    )

            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    def _merge_results(
        self,
        vector_results: list[RetrievalResult],
        graph_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Merge and rank results from vector and graph search."""
        # Create lookup by chunk_id
        merged: dict[str, RetrievalResult] = {}

        # Add vector results with weight
        for r in vector_results:
            weighted_score = r.score * self._config.vector_weight
            merged[r.chunk_id] = RetrievalResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=weighted_score,
                source="vector",
                metadata=r.metadata,
                entities=r.entities,
                events=r.events,
            )

        # Add/merge graph results with weight
        for r in graph_results:
            weighted_score = r.score * self._config.graph_weight

            if r.chunk_id in merged:
                # Combine scores and mark as both
                existing = merged[r.chunk_id]
                existing.score += weighted_score
                existing.source = "both"
                existing.entities = list(set(existing.entities + r.entities))
                existing.events = list(set(existing.events + r.events))
            else:
                merged[r.chunk_id] = RetrievalResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=weighted_score,
                    source="graph",
                    metadata=r.metadata,
                    entities=r.entities,
                    events=r.events,
                )

        # Sort by combined score
        results = list(merged.values())
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def _apply_temporal_boost(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Apply temporal boost for adjacent chunks."""
        if len(results) < 2 or self._config.temporal_boost <= 1.0:
            return results

        # Get chunk indices
        chunk_indices = {
            r.chunk_id: r.metadata.get("chunk_idx", -1) for r in results
        }

        # Find adjacent pairs and boost
        for i, result in enumerate(results):
            idx = chunk_indices.get(result.chunk_id, -1)
            if idx < 0:
                continue

            # Check if adjacent to any higher-ranked result
            for j in range(i):
                other_idx = chunk_indices.get(results[j].chunk_id, -1)
                if other_idx >= 0 and abs(idx - other_idx) == 1:
                    # Apply temporal boost
                    result.score *= self._config.temporal_boost
                    break

        # Re-sort after boost
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    async def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Rerank results using LLM."""
        if not self._llm or len(results) <= 1:
            return results

        try:
            # Build reranking prompt
            context_parts = []
            for i, r in enumerate(results):
                context_parts.append(f"[{i}] {r.text[:500]}")

            context = "\n\n".join(context_parts)

            prompt = f"""Given the query: "{query}"

Rank the following passages by relevance (most relevant first).
Return only the indices in order, separated by commas.

Passages:
{context}

Ranking (indices only):"""

            response = await self._llm.generate(prompt)

            # Parse ranking
            try:
                indices_str = response.text.strip()
                # Handle various formats: "0, 1, 2" or "0,1,2" or "[0, 1, 2]"
                indices_str = indices_str.strip("[]")
                indices = [int(x.strip()) for x in indices_str.split(",")]

                # Reorder results
                reranked = []
                seen = set()
                for idx in indices:
                    if 0 <= idx < len(results) and idx not in seen:
                        reranked.append(results[idx])
                        seen.add(idx)

                # Add any missing results at the end
                for i, r in enumerate(results):
                    if i not in seen:
                        reranked.append(r)

                return reranked

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse reranking response: {e}")
                return results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    async def get_context_for_query(
        self,
        query: str,
        stream_id: str,
        max_tokens: int = 8000,
    ) -> str:
        """
        Get formatted context string for RAG generation.

        Args:
            query: User query
            stream_id: Stream to search
            max_tokens: Maximum tokens in context

        Returns:
            Formatted context string
        """
        results = await self.retrieve(query, stream_id)

        if not results:
            return "No relevant context found."

        # Build context string
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough estimate

        for r in results:
            # Format with timestamp
            start = r.metadata.get("start_time", "")
            end = r.metadata.get("end_time", "")
            timestamp = f"[{start} - {end}]" if start and end else ""

            # Add entity/event info if available
            extra_info = ""
            if r.entities:
                extra_info += f" Entities: {', '.join(r.entities[:5])}"
            if r.events:
                extra_info += f" Events: {', '.join(r.events[:3])}"

            part = f"{timestamp}\n{r.text}{extra_info}"

            if total_chars + len(part) > max_chars:
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n\n".join(context_parts)

    @property
    def config(self) -> HybridConfig:
        """Get current configuration."""
        return self._config

    def update_config(self, **kwargs: Any) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
