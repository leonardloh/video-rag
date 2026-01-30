"""Graph ingestion for entity and event extraction into Neo4j."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from db.neo4j_client import Neo4jClient
    from models.gemini.gemini_embeddings import GeminiEmbeddings
    from models.gemini.gemini_llm import GeminiLLM

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity extracted from caption text."""

    name: str
    type: str  # PERSON, VEHICLE, OBJECT, LOCATION
    confidence: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedEvent:
    """An event extracted from caption text."""

    description: str
    event_type: str  # MOVEMENT, INTERACTION, SAFETY_INCIDENT, OPERATION, ANOMALY
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    participants: list[str] = field(default_factory=list)
    confidence: float = 1.0
    start_time: str = ""
    end_time: str = ""


@dataclass
class IngestionResult:
    """Result of ingesting a chunk into the graph."""

    chunk_node_id: str
    entity_count: int
    event_count: int
    relationship_count: int
    entities: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


# Default prompts for entity and event extraction
DEFAULT_ENTITY_PROMPT = """Extract all entities from the following video caption. For each entity, provide:
- name: The entity identifier (e.g., "forklift", "worker in blue", "pallet")
- type: One of PERSON, VEHICLE, OBJECT, LOCATION
- attributes: Any descriptive attributes (color, size, position)

Caption:
{caption}

CV Metadata (if available):
{cv_metadata}

Return as JSON array:
[
  {{"name": "...", "type": "...", "attributes": {{...}}}},
  ...
]
"""

DEFAULT_EVENT_PROMPT = """Extract all events/actions from the following video caption. For each event:
- description: What happened
- event_type: Category (MOVEMENT, INTERACTION, SAFETY_INCIDENT, OPERATION, ANOMALY)
- severity: LOW, MEDIUM, HIGH, or CRITICAL
- participants: List of entity names involved
- start_time: When event started (from caption timestamps)
- end_time: When event ended

Caption:
{caption}

Entities present:
{entities}

Return as JSON array:
[
  {{
    "description": "...",
    "event_type": "...",
    "severity": "...",
    "participants": ["...", "..."],
    "start_time": "HH:MM:SS",
    "end_time": "HH:MM:SS"
  }},
  ...
]
"""


class GraphIngestion:
    """Handles ingestion of video data into the graph database."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        llm: GeminiLLM,
        embeddings: Optional[GeminiEmbeddings] = None,
        entity_prompt: Optional[str] = None,
        event_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize graph ingestion.

        Args:
            neo4j_client: Neo4j database client
            llm: LLM for entity/event extraction
            embeddings: Optional embedding generator for entity matching
            entity_prompt: Custom prompt for entity extraction
            event_prompt: Custom prompt for event extraction
        """
        self._neo4j = neo4j_client
        self._llm = llm
        self._embeddings = embeddings
        self._entity_prompt = entity_prompt or DEFAULT_ENTITY_PROMPT
        self._event_prompt = event_prompt or DEFAULT_EVENT_PROMPT

        # Cache for entity deduplication within a stream
        self._entity_cache: dict[str, dict[str, str]] = {}  # stream_id -> {name: node_id}

    def _parse_json_response(self, text: str) -> list[dict[str, Any]]:
        """Parse JSON array from LLM response, handling common issues."""
        # Try to find JSON array in response
        text = text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        # Find the JSON array
        start_idx = text.find("[")
        end_idx = text.rfind("]") + 1

        if start_idx == -1 or end_idx == 0:
            logger.warning("No JSON array found in response")
            return []

        json_str = text[start_idx:end_idx]

        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return []

    async def extract_entities(
        self,
        caption: str,
        cv_metadata: Optional[dict[str, Any]] = None,
    ) -> list[ExtractedEntity]:
        """
        Extract entities from caption using LLM.

        Args:
            caption: VLM-generated caption text
            cv_metadata: Optional CV detection metadata

        Returns:
            List of extracted entities
        """
        cv_meta_str = json.dumps(cv_metadata) if cv_metadata else "None"

        prompt = self._entity_prompt.format(
            caption=caption,
            cv_metadata=cv_meta_str,
        )

        result = await self._llm.generate(prompt)
        parsed = self._parse_json_response(result.text)

        entities = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            name = item.get("name", "").strip()
            entity_type = item.get("type", "OBJECT").upper()

            if not name:
                continue

            # Validate entity type
            valid_types = {"PERSON", "VEHICLE", "OBJECT", "LOCATION"}
            if entity_type not in valid_types:
                entity_type = "OBJECT"

            entities.append(
                ExtractedEntity(
                    name=name,
                    type=entity_type,
                    attributes=item.get("attributes", {}),
                )
            )

        return entities

    async def extract_events(
        self,
        caption: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEvent]:
        """
        Extract events from caption using LLM.

        Args:
            caption: VLM-generated caption text
            entities: Previously extracted entities

        Returns:
            List of extracted events
        """
        entities_str = ", ".join([f"{e.name} ({e.type})" for e in entities])

        prompt = self._event_prompt.format(
            caption=caption,
            entities=entities_str,
        )

        result = await self._llm.generate(prompt)
        parsed = self._parse_json_response(result.text)

        events = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            description = item.get("description", "").strip()
            event_type = item.get("event_type", "OPERATION").upper()

            if not description:
                continue

            # Validate event type
            valid_types = {"MOVEMENT", "INTERACTION", "SAFETY_INCIDENT", "OPERATION", "ANOMALY"}
            if event_type not in valid_types:
                event_type = "OPERATION"

            # Validate severity
            severity = item.get("severity", "LOW").upper()
            valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
            if severity not in valid_severities:
                severity = "LOW"

            events.append(
                ExtractedEvent(
                    description=description,
                    event_type=event_type,
                    severity=severity,
                    participants=item.get("participants", []),
                    start_time=item.get("start_time", ""),
                    end_time=item.get("end_time", ""),
                )
            )

        return events

    async def _get_or_create_entity(
        self,
        entity: ExtractedEntity,
        stream_id: str,
        chunk_time: str,
    ) -> str:
        """Get existing entity node or create new one."""
        # Check cache first
        if stream_id not in self._entity_cache:
            self._entity_cache[stream_id] = {}

        cache_key = f"{entity.name.lower()}:{entity.type}"
        if cache_key in self._entity_cache[stream_id]:
            # Update last_seen time
            entity_id = self._entity_cache[stream_id][cache_key]
            await self._neo4j.execute_query(
                """
                MATCH (e:Entity {id: $entity_id})
                SET e.last_seen = $time, e.occurrence_count = e.occurrence_count + 1
                """,
                {"entity_id": entity_id, "time": chunk_time},
            )
            return entity_id

        # Check database for existing entity
        results = await self._neo4j.execute_query(
            """
            MATCH (e:Entity {name: $name, type: $type, stream_id: $stream_id})
            RETURN e.id as id
            """,
            {"name": entity.name, "type": entity.type, "stream_id": stream_id},
        )

        if results:
            entity_id = results[0]["id"]
            self._entity_cache[stream_id][cache_key] = entity_id
            # Update last_seen
            await self._neo4j.execute_query(
                """
                MATCH (e:Entity {id: $entity_id})
                SET e.last_seen = $time, e.occurrence_count = e.occurrence_count + 1
                """,
                {"entity_id": entity_id, "time": chunk_time},
            )
            return entity_id

        # Create new entity
        entity_id = str(uuid.uuid4())
        await self._neo4j.create_node(
            labels=["Entity"],
            properties={
                "id": entity_id,
                "entity_id": entity_id,
                "name": entity.name,
                "type": entity.type,
                "stream_id": stream_id,
                "first_seen": chunk_time,
                "last_seen": chunk_time,
                "occurrence_count": 1,
                "attributes": json.dumps(entity.attributes),
            },
        )

        self._entity_cache[stream_id][cache_key] = entity_id
        return entity_id

    async def ingest_chunk(
        self,
        chunk_id: str,
        stream_id: str,
        chunk_idx: int,
        caption: str,
        start_time: str,
        end_time: str,
        embedding_id: str,
        cv_metadata: Optional[dict[str, Any]] = None,
        extract_entities: bool = True,
        extract_events: bool = True,
    ) -> IngestionResult:
        """
        Ingest a video chunk into the graph.

        Creates:
        - VideoChunk node
        - Entity nodes (with deduplication)
        - Event nodes
        - All relationships

        Args:
            chunk_id: Unique chunk identifier
            stream_id: Parent stream ID
            chunk_idx: Chunk sequence number
            caption: VLM caption text
            start_time: Chunk start timestamp
            end_time: Chunk end timestamp
            embedding_id: Reference to Milvus document
            cv_metadata: Optional CV detection data
            extract_entities: Whether to extract entities
            extract_events: Whether to extract events

        Returns:
            Summary of created nodes and relationships
        """
        result = IngestionResult(
            chunk_node_id=chunk_id,
            entity_count=0,
            event_count=0,
            relationship_count=0,
        )

        # Create VideoChunk node
        await self._neo4j.create_node(
            labels=["VideoChunk"],
            properties={
                "id": chunk_id,
                "chunk_id": chunk_id,
                "stream_id": stream_id,
                "chunk_idx": chunk_idx,
                "start_time": start_time,
                "end_time": end_time,
                "caption": caption[:10000],  # Truncate very long captions
                "embedding_id": embedding_id,
            },
        )

        if not extract_entities and not extract_events:
            return result

        # Extract entities
        entities: list[ExtractedEntity] = []
        if extract_entities:
            entities = await self.extract_entities(caption, cv_metadata)
            result.entity_count = len(entities)

            # Create entity nodes and relationships
            for entity in entities:
                entity_id = await self._get_or_create_entity(
                    entity, stream_id, start_time
                )
                result.entities.append(entity.name)

                # Create CONTAINS relationship
                await self._neo4j.create_relationship(
                    start_node_id=chunk_id,
                    end_node_id=entity_id,
                    rel_type="CONTAINS",
                    properties={
                        "confidence": entity.confidence,
                    },
                )
                result.relationship_count += 1

        # Extract events
        if extract_events and entities:
            events = await self.extract_events(caption, entities)
            result.event_count = len(events)

            for event in events:
                event_id = str(uuid.uuid4())

                # Create Event node
                await self._neo4j.create_node(
                    labels=["Event"],
                    properties={
                        "id": event_id,
                        "event_id": event_id,
                        "description": event.description,
                        "event_type": event.event_type,
                        "severity": event.severity,
                        "start_time": event.start_time or start_time,
                        "end_time": event.end_time or end_time,
                        "stream_id": stream_id,
                        "confidence": event.confidence,
                    },
                )
                result.events.append(event.description[:50])

                # Create OCCURS_IN relationship
                await self._neo4j.create_relationship(
                    start_node_id=event_id,
                    end_node_id=chunk_id,
                    rel_type="OCCURS_IN",
                )
                result.relationship_count += 1

                # Create PARTICIPATES_IN relationships for entities
                for participant_name in event.participants:
                    # Find entity by name
                    cache_key_prefix = f"{participant_name.lower()}:"
                    for cache_key, entity_id in self._entity_cache.get(stream_id, {}).items():
                        if cache_key.startswith(cache_key_prefix):
                            await self._neo4j.create_relationship(
                                start_node_id=entity_id,
                                end_node_id=event_id,
                                rel_type="PARTICIPATES_IN",
                                properties={"role": "participant"},
                            )
                            result.relationship_count += 1
                            break

        return result

    async def create_temporal_links(self, stream_id: str) -> int:
        """
        Create FOLLOWS relationships between sequential chunks.

        Args:
            stream_id: Stream to process

        Returns:
            Number of relationships created
        """
        query = """
        MATCH (c1:VideoChunk {stream_id: $stream_id})
        MATCH (c2:VideoChunk {stream_id: $stream_id})
        WHERE c2.chunk_idx = c1.chunk_idx + 1
        MERGE (c1)-[r:FOLLOWS]->(c2)
        RETURN count(r) as created
        """

        results = await self._neo4j.execute_query(query, {"stream_id": stream_id})
        return results[0]["created"] if results else 0

    async def link_entities_across_chunks(
        self,
        stream_id: str,
        similarity_threshold: float = 0.85,
    ) -> int:
        """
        Link same entities across different chunks using embeddings.

        Args:
            stream_id: Stream to process
            similarity_threshold: Minimum similarity for SAME_AS

        Returns:
            Number of SAME_AS relationships created
        """
        if not self._embeddings:
            logger.warning("No embeddings client provided, skipping entity linking")
            return 0

        # Get all entities for the stream
        results = await self._neo4j.execute_query(
            """
            MATCH (e:Entity {stream_id: $stream_id})
            RETURN e.id as id, e.name as name, e.type as type
            """,
            {"stream_id": stream_id},
        )

        if len(results) < 2:
            return 0

        # Generate embeddings for entity names
        entity_names = [r["name"] for r in results]
        embeddings = await self._embeddings.embed_documents(entity_names)

        # Find similar entities
        links_created = 0
        for i, emb1 in enumerate(embeddings):
            for j, emb2 in enumerate(embeddings):
                if i >= j:
                    continue

                # Skip if same type (likely same entity)
                if results[i]["type"] == results[j]["type"]:
                    similarity = self._embeddings.cosine_similarity(emb1, emb2)
                    if similarity >= similarity_threshold:
                        # Create SAME_AS relationship
                        try:
                            await self._neo4j.create_relationship(
                                start_node_id=results[i]["id"],
                                end_node_id=results[j]["id"],
                                rel_type="SAME_AS",
                                properties={"confidence": similarity},
                            )
                            links_created += 1
                        except Exception as e:
                            logger.debug(f"Failed to create SAME_AS link: {e}")

        return links_created

    async def ingest_batch(
        self,
        chunks: list[dict[str, Any]],
        stream_id: str,
        extract_entities: bool = True,
        extract_events: bool = True,
    ) -> dict[str, Any]:
        """
        Batch ingest multiple chunks.

        Args:
            chunks: List of chunk data dictionaries with keys:
                - chunk_id, chunk_idx, caption, start_time, end_time, embedding_id
            stream_id: Parent stream ID
            extract_entities: Whether to extract entities
            extract_events: Whether to extract events

        Returns:
            Summary of all ingested data
        """
        total_entities = 0
        total_events = 0
        total_relationships = 0
        chunk_results = []

        for chunk in chunks:
            result = await self.ingest_chunk(
                chunk_id=chunk["chunk_id"],
                stream_id=stream_id,
                chunk_idx=chunk["chunk_idx"],
                caption=chunk["caption"],
                start_time=chunk["start_time"],
                end_time=chunk["end_time"],
                embedding_id=chunk.get("embedding_id", ""),
                cv_metadata=chunk.get("cv_metadata"),
                extract_entities=extract_entities,
                extract_events=extract_events,
            )

            total_entities += result.entity_count
            total_events += result.event_count
            total_relationships += result.relationship_count
            chunk_results.append(result)

        # Create temporal links after all chunks are ingested
        temporal_links = await self.create_temporal_links(stream_id)
        total_relationships += temporal_links

        return {
            "stream_id": stream_id,
            "chunks_processed": len(chunks),
            "total_entities": total_entities,
            "total_events": total_events,
            "total_relationships": total_relationships,
            "temporal_links": temporal_links,
            "chunk_results": chunk_results,
        }

    def clear_cache(self, stream_id: Optional[str] = None) -> None:
        """Clear entity cache for a stream or all streams."""
        if stream_id:
            self._entity_cache.pop(stream_id, None)
        else:
            self._entity_cache.clear()
