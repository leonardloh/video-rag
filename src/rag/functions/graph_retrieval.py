"""Graph retrieval function for RAG pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from .base import FunctionConfig, FunctionResult, FunctionStatus, RAGFunction

if TYPE_CHECKING:
    from ...db.graph_retrieval import GraphRetrieval
    from ...db.neo4j_client import Neo4jClient
    from ...models.gemini.gemini_embeddings import GeminiEmbeddings
    from ...models.gemini.gemini_llm import GeminiLLM

logger = logging.getLogger(__name__)


class RetrievalType(Enum):
    """Type of graph retrieval to perform."""

    TEMPORAL = "temporal"  # Get temporal context around timestamp
    ENTITY = "entity"  # Get entity timeline
    EVENT = "event"  # Get related events
    TRAVERSE = "traverse"  # Traverse from chunk
    INTERACTION = "interaction"  # Find entity interactions


@dataclass
class GraphRetrievalInput:
    """Input for graph retrieval function."""

    query: str
    stream_id: str
    retrieval_type: RetrievalType = RetrievalType.EVENT
    timestamp: Optional[str] = None  # For temporal retrieval
    entity_name: Optional[str] = None  # For entity retrieval
    entity2_name: Optional[str] = None  # For interaction retrieval
    chunk_id: Optional[str] = None  # For traverse retrieval
    top_k: int = 5


@dataclass
class GraphRetrievalOutput:
    """Output from graph retrieval function."""

    query: str
    context: str  # Formatted context string
    chunks: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    answer: Optional[str] = None  # LLM-generated answer if requested


# Default prompt for generating answers from graph context
DEFAULT_ANSWER_PROMPT = """Based on the following video context from a knowledge graph, answer the user's question.

Context:
{context}

Question: {query}

Provide a detailed answer based only on the information in the context. If the context doesn't contain enough information to answer the question, say so.
"""


class GraphRetrievalFunction(RAGFunction[GraphRetrievalInput, GraphRetrievalOutput]):
    """Graph retrieval function for video context.

    This function:
    1. Queries the Neo4j graph for relevant context
    2. Supports multiple retrieval types (temporal, entity, event, etc.)
    3. Formats context for LLM consumption
    4. Optionally generates answers using LLM
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        llm: Optional[GeminiLLM] = None,
        embeddings: Optional[GeminiEmbeddings] = None,
        config: FunctionConfig | None = None,
        top_k: int = 5,
        generate_answer: bool = False,
        answer_prompt: Optional[str] = None,
        chat_history: bool = False,
    ) -> None:
        """
        Initialize graph retrieval function.

        Args:
            neo4j_client: Neo4j database client
            llm: Optional LLM for answer generation
            embeddings: Optional embedding generator
            config: Function configuration
            top_k: Default number of results to return
            generate_answer: Whether to generate LLM answer
            answer_prompt: Custom prompt for answer generation
            chat_history: Whether to maintain chat history
        """
        super().__init__(config or FunctionConfig(name="graph_retrieval"))
        self._neo4j_client = neo4j_client
        self._llm = llm
        self._embeddings = embeddings
        self._top_k = top_k
        self._generate_answer = generate_answer
        self._answer_prompt = answer_prompt or DEFAULT_ANSWER_PROMPT
        self._chat_history = chat_history
        self._history: list[dict[str, str]] = []

        # Lazy import to avoid circular imports
        from ...db.graph_retrieval import GraphRetrieval

        self._graph_retrieval = GraphRetrieval(
            neo4j_client=neo4j_client,
            embeddings=embeddings,
        )

    @property
    def top_k(self) -> int:
        """Get top_k setting."""
        return self._top_k

    @property
    def generate_answer(self) -> bool:
        """Get generate_answer setting."""
        return self._generate_answer

    @property
    def chat_history(self) -> list[dict[str, str]]:
        """Get chat history."""
        return self._history.copy()

    def configure(self, **kwargs: Any) -> None:
        """
        Update function configuration.

        Args:
            **kwargs: Configuration parameters
        """
        super().configure(**kwargs)

        if "top_k" in kwargs:
            self._top_k = kwargs["top_k"]
        if "generate_answer" in kwargs:
            self._generate_answer = kwargs["generate_answer"]
        if "answer_prompt" in kwargs:
            self._answer_prompt = kwargs["answer_prompt"]
        if "chat_history" in kwargs:
            self._chat_history = kwargs["chat_history"]

    async def execute(
        self,
        input_data: GraphRetrievalInput,
        **kwargs: Any,
    ) -> FunctionResult[GraphRetrievalOutput]:
        """
        Execute graph retrieval.

        Args:
            input_data: Retrieval query parameters
            **kwargs: Additional parameters

        Returns:
            FunctionResult with retrieved context
        """
        self._set_status(FunctionStatus.RUNNING)

        # Get parameters from kwargs or use defaults
        top_k = kwargs.get("top_k", input_data.top_k or self._top_k)
        generate_answer = kwargs.get("generate_answer", self._generate_answer)

        try:
            # Execute retrieval based on type
            if input_data.retrieval_type == RetrievalType.TEMPORAL:
                output = await self._retrieve_temporal(input_data, top_k)
            elif input_data.retrieval_type == RetrievalType.ENTITY:
                output = await self._retrieve_entity(input_data)
            elif input_data.retrieval_type == RetrievalType.EVENT:
                output = await self._retrieve_events(input_data, top_k)
            elif input_data.retrieval_type == RetrievalType.TRAVERSE:
                output = await self._retrieve_traverse(input_data)
            elif input_data.retrieval_type == RetrievalType.INTERACTION:
                output = await self._retrieve_interaction(input_data)
            else:
                # Default to event retrieval
                output = await self._retrieve_events(input_data, top_k)

            # Generate answer if requested
            if generate_answer and self._llm and output.context:
                answer = await self._generate_answer_from_context(
                    query=input_data.query,
                    context=output.context,
                )
                output.answer = answer

                # Update chat history
                if self._chat_history:
                    self._history.append({
                        "role": "user",
                        "content": input_data.query,
                    })
                    self._history.append({
                        "role": "assistant",
                        "content": answer,
                    })

            self._set_status(FunctionStatus.COMPLETED)
            return FunctionResult.ok(
                output,
                stream_id=input_data.stream_id,
                retrieval_type=input_data.retrieval_type.value,
            )

        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            self._set_status(FunctionStatus.FAILED)
            return FunctionResult.fail(str(e))

    async def _retrieve_temporal(
        self,
        input_data: GraphRetrievalInput,
        top_k: int,
    ) -> GraphRetrievalOutput:
        """Retrieve temporal context around a timestamp."""
        timestamp = input_data.timestamp or "00:00:00"

        context = await self._graph_retrieval.get_temporal_context(
            stream_id=input_data.stream_id,
            timestamp=timestamp,
            window_before=top_k // 2,
            window_after=top_k // 2,
        )

        # Format context string
        context_str = self._format_temporal_context(context)

        return GraphRetrievalOutput(
            query=input_data.query,
            context=context_str,
            chunks=context.chunks,
            entities=context.entities,
            events=context.events,
        )

    async def _retrieve_entity(
        self,
        input_data: GraphRetrievalInput,
    ) -> GraphRetrievalOutput:
        """Retrieve entity timeline."""
        entity_name = input_data.entity_name or input_data.query

        timeline = await self._graph_retrieval.get_entity_timeline(
            entity_name=entity_name,
            stream_id=input_data.stream_id,
        )

        # Format context string
        context_str = self._format_entity_timeline(timeline)

        return GraphRetrievalOutput(
            query=input_data.query,
            context=context_str,
            chunks=timeline.appearances,
            entities=[timeline.entity] if timeline.entity else [],
            events=timeline.events,
        )

    async def _retrieve_events(
        self,
        input_data: GraphRetrievalInput,
        top_k: int,
    ) -> GraphRetrievalOutput:
        """Retrieve events related to query."""
        events = await self._graph_retrieval.get_related_events(
            query=input_data.query,
            stream_id=input_data.stream_id,
            top_k=top_k,
        )

        # Extract chunks from events
        chunks = []
        for event in events:
            chunk_context = event.get("chunk_context")
            if chunk_context and chunk_context not in chunks:
                chunks.append(chunk_context)

        # Format context string
        context_str = self._format_events_context(events)

        return GraphRetrievalOutput(
            query=input_data.query,
            context=context_str,
            chunks=chunks,
            events=events,
        )

    async def _retrieve_traverse(
        self,
        input_data: GraphRetrievalInput,
    ) -> GraphRetrievalOutput:
        """Traverse graph from a chunk."""
        chunk_id = input_data.chunk_id
        if not chunk_id:
            return GraphRetrievalOutput(
                query=input_data.query,
                context="No chunk_id provided for traversal.",
            )

        subgraph = await self._graph_retrieval.traverse_from_chunk(
            chunk_id=chunk_id,
            max_depth=2,
        )

        # Extract nodes by type
        chunks = []
        entities = []
        events = []

        for node in subgraph.get("nodes", []):
            labels = node.get("labels", [])
            if "VideoChunk" in labels:
                chunks.append(node)
            elif "Entity" in labels:
                entities.append(node)
            elif "Event" in labels:
                events.append(node)

        # Format context string
        context_str = self._format_subgraph_context(subgraph)

        return GraphRetrievalOutput(
            query=input_data.query,
            context=context_str,
            chunks=chunks,
            entities=entities,
            events=events,
        )

    async def _retrieve_interaction(
        self,
        input_data: GraphRetrievalInput,
    ) -> GraphRetrievalOutput:
        """Retrieve interactions between entities."""
        entity1 = input_data.entity_name or ""
        entity2 = input_data.entity2_name or ""

        if not entity1 or not entity2:
            return GraphRetrievalOutput(
                query=input_data.query,
                context="Two entity names required for interaction retrieval.",
            )

        interactions = await self._graph_retrieval.find_entity_interactions(
            entity1=entity1,
            entity2=entity2,
            stream_id=input_data.stream_id,
        )

        # Format context string
        context_str = self._format_interactions_context(interactions, entity1, entity2)

        return GraphRetrievalOutput(
            query=input_data.query,
            context=context_str,
            events=interactions,
        )

    def _format_temporal_context(self, context: Any) -> str:
        """Format temporal context as string."""
        parts = []

        if context.chunks:
            parts.append("## Video Chunks (Temporal Sequence)")
            for chunk in context.chunks:
                start = chunk.get("start_time", "")
                end = chunk.get("end_time", "")
                caption = chunk.get("caption", "")[:500]
                parts.append(f"\n[{start} - {end}]\n{caption}")

        if context.entities:
            parts.append("\n\n## Entities Present")
            for entity in context.entities:
                name = entity.get("name", "")
                etype = entity.get("type", "")
                parts.append(f"- {name} ({etype})")

        if context.events:
            parts.append("\n\n## Events")
            for event in context.events:
                desc = event.get("description", "")
                etype = event.get("event_type", "")
                severity = event.get("severity", "")
                parts.append(f"- [{etype}/{severity}] {desc}")

        return "\n".join(parts) if parts else "No temporal context found."

    def _format_entity_timeline(self, timeline: Any) -> str:
        """Format entity timeline as string."""
        if not timeline.entity:
            return "Entity not found."

        parts = []
        entity = timeline.entity
        parts.append(f"## Entity: {entity.get('name', '')} ({entity.get('type', '')})")
        parts.append(f"First seen: {entity.get('first_seen', '')}")
        parts.append(f"Last seen: {entity.get('last_seen', '')}")
        parts.append(f"Occurrences: {entity.get('occurrence_count', 0)}")

        if timeline.appearances:
            parts.append("\n### Appearances")
            for chunk in timeline.appearances:
                start = chunk.get("start_time", "")
                end = chunk.get("end_time", "")
                caption = chunk.get("caption", "")[:300]
                parts.append(f"\n[{start} - {end}]\n{caption}")

        if timeline.events:
            parts.append("\n### Events Participated In")
            for event in timeline.events:
                desc = event.get("description", "")
                etype = event.get("event_type", "")
                parts.append(f"- [{etype}] {desc}")

        return "\n".join(parts)

    def _format_events_context(self, events: list[dict[str, Any]]) -> str:
        """Format events as context string."""
        if not events:
            return "No related events found."

        parts = ["## Related Events"]

        for event in events:
            desc = event.get("description", "")
            etype = event.get("event_type", "")
            severity = event.get("severity", "")
            start = event.get("start_time", "")
            end = event.get("end_time", "")

            parts.append(f"\n### [{etype}] {desc}")
            parts.append(f"Severity: {severity}")
            parts.append(f"Time: {start} - {end}")

            # Add chunk context
            chunk = event.get("chunk_context")
            if chunk:
                caption = chunk.get("caption", "")[:300]
                parts.append(f"Context: {caption}")

            # Add participants
            participants = event.get("participants_detail", [])
            if participants:
                participant_names = [p.get("name", "") for p in participants]
                parts.append(f"Participants: {', '.join(participant_names)}")

        return "\n".join(parts)

    def _format_subgraph_context(self, subgraph: dict[str, Any]) -> str:
        """Format subgraph as context string."""
        nodes = subgraph.get("nodes", [])
        if not nodes:
            return "No graph context found."

        parts = [f"## Graph Context (depth={subgraph.get('depth', 0)})"]

        # Group by type
        chunks = []
        entities = []
        events = []

        for node in nodes:
            labels = node.get("labels", [])
            if "VideoChunk" in labels:
                chunks.append(node)
            elif "Entity" in labels:
                entities.append(node)
            elif "Event" in labels:
                events.append(node)

        if chunks:
            parts.append("\n### Video Chunks")
            for chunk in sorted(chunks, key=lambda x: x.get("chunk_idx", 0)):
                start = chunk.get("start_time", "")
                end = chunk.get("end_time", "")
                caption = chunk.get("caption", "")[:300]
                parts.append(f"\n[{start} - {end}]\n{caption}")

        if entities:
            parts.append("\n### Entities")
            for entity in entities:
                name = entity.get("name", "")
                etype = entity.get("type", "")
                parts.append(f"- {name} ({etype})")

        if events:
            parts.append("\n### Events")
            for event in events:
                desc = event.get("description", "")
                etype = event.get("event_type", "")
                parts.append(f"- [{etype}] {desc}")

        return "\n".join(parts)

    def _format_interactions_context(
        self,
        interactions: list[dict[str, Any]],
        entity1: str,
        entity2: str,
    ) -> str:
        """Format entity interactions as context string."""
        if not interactions:
            return f"No interactions found between '{entity1}' and '{entity2}'."

        parts = [f"## Interactions: {entity1} <-> {entity2}"]

        for interaction in interactions:
            desc = interaction.get("description", "")
            etype = interaction.get("event_type", "")
            severity = interaction.get("severity", "")
            start = interaction.get("start_time", "")

            parts.append(f"\n### [{etype}] {desc}")
            parts.append(f"Severity: {severity}")
            parts.append(f"Time: {start}")

        return "\n".join(parts)

    async def _generate_answer_from_context(
        self,
        query: str,
        context: str,
    ) -> str:
        """Generate answer using LLM."""
        if not self._llm:
            return ""

        # Include chat history if enabled
        history_context = ""
        if self._chat_history and self._history:
            history_parts = []
            for msg in self._history[-6:]:  # Last 3 exchanges
                role = msg["role"]
                content = msg["content"]
                history_parts.append(f"{role.title()}: {content}")
            history_context = "\n\nPrevious conversation:\n" + "\n".join(history_parts)

        prompt = self._answer_prompt.format(
            context=context + history_context,
            query=query,
        )

        result = await self._llm.generate(prompt)
        return result.text

    async def reset(self) -> None:
        """Reset function state and clear chat history."""
        await super().reset()
        self._history.clear()

    def clear_history(self) -> None:
        """Clear chat history."""
        self._history.clear()

    async def get_stream_summary(self, stream_id: str) -> dict[str, Any]:
        """
        Get summary of graph data for a stream.

        Args:
            stream_id: Stream to summarize

        Returns:
            Summary statistics
        """
        return await self._graph_retrieval.get_stream_summary(stream_id)
