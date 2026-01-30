"""Graph retrieval for querying Neo4j graph database."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from db.neo4j_client import Neo4jClient
    from models.gemini.gemini_embeddings import GeminiEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class TemporalContext:
    """Context from temporal graph traversal."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EntityTimeline:
    """Timeline of an entity's appearances."""

    entity: dict[str, Any]
    appearances: list[dict[str, Any]] = field(default_factory=list)
    interactions: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class GraphRetrieval:
    """Handles retrieval from the graph database."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        embeddings: Optional[GeminiEmbeddings] = None,
    ) -> None:
        """
        Initialize graph retrieval.

        Args:
            neo4j_client: Neo4j database client
            embeddings: Optional embedding generator for query processing
        """
        self._neo4j = neo4j_client
        self._embeddings = embeddings

    async def get_temporal_context(
        self,
        stream_id: str,
        timestamp: str,
        window_before: int = 2,
        window_after: int = 2,
    ) -> TemporalContext:
        """
        Get temporal context around a timestamp.

        Args:
            stream_id: Stream to query
            timestamp: Center timestamp (HH:MM:SS)
            window_before: Number of chunks before
            window_after: Number of chunks after

        Returns:
            Temporal context with chunks, entities, events
        """
        # Find the chunk containing the timestamp
        center_query = """
        MATCH (c:VideoChunk {stream_id: $stream_id})
        WHERE c.start_time <= $timestamp AND c.end_time >= $timestamp
        RETURN c.chunk_idx as idx
        LIMIT 1
        """

        results = await self._neo4j.execute_query(
            center_query, {"stream_id": stream_id, "timestamp": timestamp}
        )

        if not results:
            # If no exact match, find closest chunk
            fallback_query = """
            MATCH (c:VideoChunk {stream_id: $stream_id})
            RETURN c.chunk_idx as idx
            ORDER BY abs(c.chunk_idx - 0)
            LIMIT 1
            """
            results = await self._neo4j.execute_query(
                fallback_query, {"stream_id": stream_id}
            )
            if not results:
                return TemporalContext()

        center_idx = results[0]["idx"]
        min_idx = max(0, center_idx - window_before)
        max_idx = center_idx + window_after

        # Get chunks in window
        chunks_query = """
        MATCH (c:VideoChunk {stream_id: $stream_id})
        WHERE c.chunk_idx >= $min_idx AND c.chunk_idx <= $max_idx
        RETURN c {.*} as chunk
        ORDER BY c.chunk_idx
        """

        chunk_results = await self._neo4j.execute_query(
            chunks_query,
            {"stream_id": stream_id, "min_idx": min_idx, "max_idx": max_idx},
        )
        chunks = [r["chunk"] for r in chunk_results]

        # Get entities in these chunks
        entities_query = """
        MATCH (c:VideoChunk {stream_id: $stream_id})-[:CONTAINS]->(e:Entity)
        WHERE c.chunk_idx >= $min_idx AND c.chunk_idx <= $max_idx
        RETURN DISTINCT e {.*} as entity
        """

        entity_results = await self._neo4j.execute_query(
            entities_query,
            {"stream_id": stream_id, "min_idx": min_idx, "max_idx": max_idx},
        )
        entities = [r["entity"] for r in entity_results]

        # Get events in these chunks
        events_query = """
        MATCH (ev:Event)-[:OCCURS_IN]->(c:VideoChunk {stream_id: $stream_id})
        WHERE c.chunk_idx >= $min_idx AND c.chunk_idx <= $max_idx
        RETURN ev {.*} as event
        """

        event_results = await self._neo4j.execute_query(
            events_query,
            {"stream_id": stream_id, "min_idx": min_idx, "max_idx": max_idx},
        )
        events = [r["event"] for r in event_results]

        # Get relationships between chunks
        rel_query = """
        MATCH (c1:VideoChunk {stream_id: $stream_id})-[r:FOLLOWS]->(c2:VideoChunk)
        WHERE c1.chunk_idx >= $min_idx AND c1.chunk_idx < $max_idx
        RETURN c1.chunk_idx as from_idx, c2.chunk_idx as to_idx
        """

        rel_results = await self._neo4j.execute_query(
            rel_query,
            {"stream_id": stream_id, "min_idx": min_idx, "max_idx": max_idx},
        )
        relationships = [
            {"type": "FOLLOWS", "from": r["from_idx"], "to": r["to_idx"]}
            for r in rel_results
        ]

        return TemporalContext(
            chunks=chunks,
            entities=entities,
            events=events,
            relationships=relationships,
        )

    async def get_entity_timeline(
        self,
        entity_name: str,
        stream_id: Optional[str] = None,
    ) -> EntityTimeline:
        """
        Get timeline of an entity's appearances.

        Args:
            entity_name: Entity to track
            stream_id: Optional stream filter

        Returns:
            Entity timeline with all appearances
        """
        # Find entity
        if stream_id:
            entity_query = """
            MATCH (e:Entity {stream_id: $stream_id})
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e {.*} as entity
            LIMIT 1
            """
            params: dict[str, Any] = {"stream_id": stream_id, "name": entity_name}
        else:
            entity_query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e {.*} as entity
            LIMIT 1
            """
            params = {"name": entity_name}

        entity_results = await self._neo4j.execute_query(entity_query, params)

        if not entity_results:
            return EntityTimeline(entity={})

        entity = entity_results[0]["entity"]
        entity_id = entity.get("id")

        # Get appearances (chunks containing this entity)
        appearances_query = """
        MATCH (c:VideoChunk)-[:CONTAINS]->(e:Entity {id: $entity_id})
        RETURN c {.*} as chunk
        ORDER BY c.chunk_idx
        """

        appearance_results = await self._neo4j.execute_query(
            appearances_query, {"entity_id": entity_id}
        )
        appearances = [r["chunk"] for r in appearance_results]

        # Get interactions with other entities
        interactions_query = """
        MATCH (e1:Entity {id: $entity_id})-[r:INTERACTS_WITH]-(e2:Entity)
        RETURN e2 {.*} as other_entity, r {.*} as interaction
        """

        interaction_results = await self._neo4j.execute_query(
            interactions_query, {"entity_id": entity_id}
        )
        interactions = [
            {
                "entity": r["other_entity"],
                "interaction": r["interaction"],
            }
            for r in interaction_results
        ]

        # Get events this entity participated in
        events_query = """
        MATCH (e:Entity {id: $entity_id})-[:PARTICIPATES_IN]->(ev:Event)
        RETURN ev {.*} as event
        ORDER BY ev.start_time
        """

        event_results = await self._neo4j.execute_query(
            events_query, {"entity_id": entity_id}
        )
        events = [r["event"] for r in event_results]

        return EntityTimeline(
            entity=entity,
            appearances=appearances,
            interactions=interactions,
            events=events,
        )

    async def get_related_events(
        self,
        query: str,
        stream_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Get events related to a query.

        Args:
            query: Natural language query
            stream_id: Stream to search
            top_k: Maximum events to return

        Returns:
            Related events with context
        """
        # Simple keyword-based search for now
        # Could be enhanced with embedding similarity
        keywords = query.lower().split()

        # Search events by description
        events_query = """
        MATCH (ev:Event {stream_id: $stream_id})
        RETURN ev {.*} as event
        """

        results = await self._neo4j.execute_query(
            events_query, {"stream_id": stream_id}
        )

        # Score events by keyword match
        scored_events = []
        for r in results:
            event = r["event"]
            description = event.get("description", "").lower()
            event_type = event.get("event_type", "").lower()

            score = sum(
                1 for kw in keywords if kw in description or kw in event_type
            )

            if score > 0:
                scored_events.append((score, event))

        # Sort by score and return top_k
        scored_events.sort(key=lambda x: x[0], reverse=True)

        events = []
        for _, event in scored_events[:top_k]:
            # Get chunk context for each event
            context_query = """
            MATCH (ev:Event {id: $event_id})-[:OCCURS_IN]->(c:VideoChunk)
            RETURN c {.*} as chunk
            """

            context_results = await self._neo4j.execute_query(
                context_query, {"event_id": event.get("id")}
            )

            event["chunk_context"] = (
                context_results[0]["chunk"] if context_results else None
            )

            # Get participants
            participants_query = """
            MATCH (e:Entity)-[:PARTICIPATES_IN]->(ev:Event {id: $event_id})
            RETURN e.name as name, e.type as type
            """

            participant_results = await self._neo4j.execute_query(
                participants_query, {"event_id": event.get("id")}
            )
            event["participants_detail"] = participant_results

            events.append(event)

        return events

    async def traverse_from_chunk(
        self,
        chunk_id: str,
        max_depth: int = 2,
        rel_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Traverse graph from a starting chunk.

        Args:
            chunk_id: Starting chunk
            max_depth: Maximum traversal depth
            rel_types: Relationship types to follow

        Returns:
            Subgraph with all reached nodes
        """
        if rel_types:
            rel_filter = "|".join(rel_types)
            query = f"""
            MATCH path = (start:VideoChunk {{id: $chunk_id}})-[:{rel_filter}*1..{max_depth}]-(end)
            RETURN
                [n in nodes(path) | n {{.*, labels: labels(n)}}] as nodes,
                [r in relationships(path) | {{type: type(r), props: properties(r)}}] as rels
            """
        else:
            query = f"""
            MATCH path = (start:VideoChunk {{id: $chunk_id}})-[*1..{max_depth}]-(end)
            RETURN
                [n in nodes(path) | n {{.*, labels: labels(n)}}] as nodes,
                [r in relationships(path) | {{type: type(r), props: properties(r)}}] as rels
            """

        results = await self._neo4j.execute_query(query, {"chunk_id": chunk_id})

        # Deduplicate nodes and relationships
        seen_nodes: dict[str, dict[str, Any]] = {}
        all_rels: list[dict[str, Any]] = []

        for r in results:
            for node in r["nodes"]:
                node_id = node.get("id", "")
                if node_id and node_id not in seen_nodes:
                    seen_nodes[node_id] = node

            all_rels.extend(r["rels"])

        return {
            "start_chunk": chunk_id,
            "nodes": list(seen_nodes.values()),
            "relationships": all_rels,
            "depth": max_depth,
        }

    async def find_entity_interactions(
        self,
        entity1: str,
        entity2: str,
        stream_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Find interactions between two entities.

        Args:
            entity1: First entity name
            entity2: Second entity name
            stream_id: Optional stream filter

        Returns:
            List of interaction events
        """
        if stream_id:
            query = """
            MATCH (e1:Entity {stream_id: $stream_id})-[:PARTICIPATES_IN]->(ev:Event)<-[:PARTICIPATES_IN]-(e2:Entity {stream_id: $stream_id})
            WHERE toLower(e1.name) CONTAINS toLower($entity1)
              AND toLower(e2.name) CONTAINS toLower($entity2)
            RETURN ev {.*} as event, e1.name as entity1_name, e2.name as entity2_name
            """
            params: dict[str, Any] = {
                "stream_id": stream_id,
                "entity1": entity1,
                "entity2": entity2,
            }
        else:
            query = """
            MATCH (e1:Entity)-[:PARTICIPATES_IN]->(ev:Event)<-[:PARTICIPATES_IN]-(e2:Entity)
            WHERE toLower(e1.name) CONTAINS toLower($entity1)
              AND toLower(e2.name) CONTAINS toLower($entity2)
            RETURN ev {.*} as event, e1.name as entity1_name, e2.name as entity2_name
            """
            params = {"entity1": entity1, "entity2": entity2}

        results = await self._neo4j.execute_query(query, params)

        interactions = []
        for r in results:
            interaction = r["event"]
            interaction["entity1"] = r["entity1_name"]
            interaction["entity2"] = r["entity2_name"]
            interactions.append(interaction)

        return interactions

    async def get_event_context(self, event_id: str) -> dict[str, Any]:
        """
        Get full context around an event.

        Returns:
            Event with participants, chunks, and related events
        """
        # Get event details
        event_query = """
        MATCH (ev:Event {id: $event_id})
        RETURN ev {.*} as event
        """

        event_results = await self._neo4j.execute_query(
            event_query, {"event_id": event_id}
        )

        if not event_results:
            return {}

        event = event_results[0]["event"]

        # Get participants
        participants_query = """
        MATCH (e:Entity)-[r:PARTICIPATES_IN]->(ev:Event {id: $event_id})
        RETURN e {.*} as entity, r.role as role
        """

        participant_results = await self._neo4j.execute_query(
            participants_query, {"event_id": event_id}
        )
        event["participants_detail"] = [
            {"entity": r["entity"], "role": r["role"]} for r in participant_results
        ]

        # Get chunk where event occurs
        chunk_query = """
        MATCH (ev:Event {id: $event_id})-[:OCCURS_IN]->(c:VideoChunk)
        RETURN c {.*} as chunk
        """

        chunk_results = await self._neo4j.execute_query(
            chunk_query, {"event_id": event_id}
        )
        event["chunk"] = chunk_results[0]["chunk"] if chunk_results else None

        # Get related events (same chunk or same participants)
        related_query = """
        MATCH (ev:Event {id: $event_id})-[:OCCURS_IN]->(c:VideoChunk)<-[:OCCURS_IN]-(related:Event)
        WHERE related.id <> $event_id
        RETURN related {.*} as event
        UNION
        MATCH (e:Entity)-[:PARTICIPATES_IN]->(ev:Event {id: $event_id})
        MATCH (e)-[:PARTICIPATES_IN]->(related:Event)
        WHERE related.id <> $event_id
        RETURN related {.*} as event
        """

        related_results = await self._neo4j.execute_query(
            related_query, {"event_id": event_id}
        )

        # Deduplicate related events
        seen_ids: set[str] = set()
        related_events = []
        for r in related_results:
            rel_event = r["event"]
            rel_id = rel_event.get("id")
            if rel_id and rel_id not in seen_ids:
                seen_ids.add(rel_id)
                related_events.append(rel_event)

        event["related_events"] = related_events

        return event

    async def get_chunks_by_entity(
        self,
        entity_name: str,
        stream_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get chunks containing a specific entity."""
        query = """
        MATCH (c:VideoChunk {stream_id: $stream_id})-[:CONTAINS]->(e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($entity_name)
        RETURN c {.*} as chunk, e.name as entity_name
        ORDER BY c.chunk_idx
        LIMIT $limit
        """

        results = await self._neo4j.execute_query(
            query,
            {"stream_id": stream_id, "entity_name": entity_name, "limit": limit},
        )

        return [
            {"chunk": r["chunk"], "matched_entity": r["entity_name"]}
            for r in results
        ]

    async def get_chunks_by_event_type(
        self,
        event_type: str,
        stream_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get chunks containing a specific event type."""
        query = """
        MATCH (ev:Event {stream_id: $stream_id, event_type: $event_type})-[:OCCURS_IN]->(c:VideoChunk)
        RETURN c {.*} as chunk, ev {.*} as event
        ORDER BY c.chunk_idx
        LIMIT $limit
        """

        results = await self._neo4j.execute_query(
            query,
            {"stream_id": stream_id, "event_type": event_type.upper(), "limit": limit},
        )

        return [{"chunk": r["chunk"], "event": r["event"]} for r in results]

    async def get_all_entities(
        self,
        stream_id: str,
        entity_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get all entities for a stream, optionally filtered by type."""
        if entity_type:
            query = """
            MATCH (e:Entity {stream_id: $stream_id, type: $type})
            RETURN e {.*} as entity
            ORDER BY e.occurrence_count DESC
            """
            params: dict[str, Any] = {"stream_id": stream_id, "type": entity_type.upper()}
        else:
            query = """
            MATCH (e:Entity {stream_id: $stream_id})
            RETURN e {.*} as entity
            ORDER BY e.occurrence_count DESC
            """
            params = {"stream_id": stream_id}

        results = await self._neo4j.execute_query(query, params)
        return [r["entity"] for r in results]

    async def get_all_events(
        self,
        stream_id: str,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get all events for a stream, optionally filtered."""
        conditions = ["ev.stream_id = $stream_id"]
        params: dict[str, Any] = {"stream_id": stream_id}

        if event_type:
            conditions.append("ev.event_type = $event_type")
            params["event_type"] = event_type.upper()

        if severity:
            conditions.append("ev.severity = $severity")
            params["severity"] = severity.upper()

        where_clause = " AND ".join(conditions)

        query = f"""
        MATCH (ev:Event)
        WHERE {where_clause}
        RETURN ev {{.*}} as event
        ORDER BY ev.start_time
        """

        results = await self._neo4j.execute_query(query, params)
        return [r["event"] for r in results]

    async def get_stream_summary(self, stream_id: str) -> dict[str, Any]:
        """Get summary statistics for a stream's graph data."""
        # Count chunks
        chunk_count_query = """
        MATCH (c:VideoChunk {stream_id: $stream_id})
        RETURN count(c) as count
        """
        chunk_results = await self._neo4j.execute_query(
            chunk_count_query, {"stream_id": stream_id}
        )
        chunk_count = chunk_results[0]["count"] if chunk_results else 0

        # Count entities by type
        entity_query = """
        MATCH (e:Entity {stream_id: $stream_id})
        RETURN e.type as type, count(e) as count
        """
        entity_results = await self._neo4j.execute_query(
            entity_query, {"stream_id": stream_id}
        )
        entity_counts = {r["type"]: r["count"] for r in entity_results}

        # Count events by type
        event_query = """
        MATCH (ev:Event {stream_id: $stream_id})
        RETURN ev.event_type as type, count(ev) as count
        """
        event_results = await self._neo4j.execute_query(
            event_query, {"stream_id": stream_id}
        )
        event_counts = {r["type"]: r["count"] for r in event_results}

        # Count events by severity
        severity_query = """
        MATCH (ev:Event {stream_id: $stream_id})
        RETURN ev.severity as severity, count(ev) as count
        """
        severity_results = await self._neo4j.execute_query(
            severity_query, {"stream_id": stream_id}
        )
        severity_counts = {r["severity"]: r["count"] for r in severity_results}

        return {
            "stream_id": stream_id,
            "chunk_count": chunk_count,
            "entity_counts": entity_counts,
            "total_entities": sum(entity_counts.values()),
            "event_counts": event_counts,
            "total_events": sum(event_counts.values()),
            "severity_distribution": severity_counts,
        }
