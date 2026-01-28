# Neo4j Graph Database Specification

## Overview

Neo4j serves as the graph database for the Video Search and Summarization (VSS) PoC, enabling temporal relationships and entity tracking across video segments. It works in conjunction with Milvus vector database to provide a hybrid retrieval system for Context-Aware RAG (CA-RAG).

## Component Locations

```
./src/db/neo4j_client.py      # Neo4j database client
./src/db/graph_ingestion.py   # Entity/event extraction and graph population
./src/db/graph_retrieval.py   # Graph traversal and context retrieval
```

---

## Graph Schema

### Node Types

```cypher
// VideoChunk - represents a processed video segment
(:VideoChunk {
    chunk_id: STRING,        // Unique identifier
    stream_id: STRING,       // Parent video stream
    chunk_idx: INTEGER,      // Chunk sequence number
    start_time: STRING,      // HH:MM:SS format
    end_time: STRING,        // HH:MM:SS format
    start_pts: INTEGER,      // Presentation timestamp (ns)
    end_pts: INTEGER,        // Presentation timestamp (ns)
    caption: STRING,         // VLM-generated caption
    embedding_id: STRING,    // Reference to Milvus document
    created_at: INTEGER      // Unix timestamp
})

// Entity - extracted entity from captions
(:Entity {
    entity_id: STRING,       // Unique identifier
    name: STRING,            // Entity name (e.g., "forklift", "worker")
    type: STRING,            // Entity type (PERSON, VEHICLE, OBJECT, LOCATION)
    first_seen: STRING,      // First appearance timestamp
    last_seen: STRING,       // Last appearance timestamp
    occurrence_count: INTEGER
})

// Event - detected event or action
(:Event {
    event_id: STRING,        // Unique identifier
    description: STRING,     // Event description
    event_type: STRING,      // Event category
    start_time: STRING,      // Event start timestamp
    end_time: STRING,        // Event end timestamp
    severity: STRING,        // LOW, MEDIUM, HIGH, CRITICAL
    confidence: FLOAT        // Detection confidence
})
```

### Relationship Types

```cypher
// Temporal sequence between chunks
(:VideoChunk)-[:FOLLOWS]->(:VideoChunk)

// Entity appears in chunk
(:VideoChunk)-[:CONTAINS {
    confidence: FLOAT,
    bbox: STRING,            // Bounding box if available
    first_frame: INTEGER,
    last_frame: INTEGER
}]->(:Entity)

// Entity participates in event
(:Entity)-[:PARTICIPATES_IN {
    role: STRING             // e.g., "actor", "subject", "object"
}]->(:Event)

// Event occurs in chunk
(:Event)-[:OCCURS_IN]->(:VideoChunk)

// Entity interactions
(:Entity)-[:INTERACTS_WITH {
    interaction_type: STRING,  // e.g., "near", "collides", "carries"
    timestamp: STRING
}]->(:Entity)

// Entity tracking across chunks
(:Entity)-[:SAME_AS {
    confidence: FLOAT
}]->(:Entity)
```

---

## Configuration

### Neo4jConfig Dataclass

```python
from dataclasses import dataclass


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    host: str = "localhost"
    port: int = 7687
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
```

### Environment Variables

```bash
# Neo4j Configuration
NEO4J_HOST=localhost
NEO4J_BOLT_PORT=7687
NEO4J_HTTP_PORT=7474
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### YAML Configuration

```yaml
# config/config.yaml
rag:
  graph_db:
    enabled: true
    type: neo4j
    host: !ENV ${NEO4J_HOST:localhost}
    port: !ENV ${NEO4J_BOLT_PORT:7687}
    username: !ENV ${NEO4J_USERNAME:neo4j}
    password: !ENV ${NEO4J_PASSWORD}

tools:
  graph_db:
    type: neo4j
    params:
      host: !ENV ${NEO4J_HOST:localhost}
      port: !ENV ${NEO4J_BOLT_PORT:7687}
      username: !ENV ${NEO4J_USERNAME:neo4j}
      password: !ENV ${NEO4J_PASSWORD}
    tools:
      embedding: gemini_embedding
```

---

## Data Classes

### GraphNode

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class GraphNode:
    """A node in the graph."""
    id: str
    labels: list[str]
    properties: dict[str, Any]
```

### GraphRelationship

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphRelationship:
    """A relationship between nodes."""
    id: str
    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
```

---

## Neo4jClient API

```python
from dataclasses import dataclass, field
from typing import Optional, List, Any


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    host: str = "localhost"
    port: int = 7687
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"


@dataclass
class GraphNode:
    """A node in the graph."""
    id: str
    labels: list[str]
    properties: dict[str, Any]


@dataclass
class GraphRelationship:
    """A relationship between nodes."""
    id: str
    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


class Neo4jClient:
    """Client for Neo4j graph database operations."""

    def __init__(self, config: Neo4jConfig):
        """
        Initialize Neo4j client.

        Args:
            config: Neo4j connection configuration
        """
        pass

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        pass

    async def close(self) -> None:
        """Close connection to Neo4j."""
        pass

    async def execute_query(
        self,
        query: str,
        parameters: Optional[dict] = None
    ) -> list[dict]:
        """
        Execute a Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Query results as list of dictionaries
        """
        pass

    async def create_node(
        self,
        labels: list[str],
        properties: dict[str, Any]
    ) -> GraphNode:
        """
        Create a new node.

        Args:
            labels: Node labels (e.g., ["VideoChunk"])
            properties: Node properties

        Returns:
            Created node
        """
        pass

    async def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        rel_type: str,
        properties: Optional[dict] = None
    ) -> GraphRelationship:
        """
        Create a relationship between nodes.

        Args:
            start_node_id: Source node ID
            end_node_id: Target node ID
            rel_type: Relationship type (e.g., "FOLLOWS")
            properties: Relationship properties

        Returns:
            Created relationship
        """
        pass

    async def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID."""
        pass

    async def find_nodes(
        self,
        label: str,
        properties: Optional[dict] = None,
        limit: int = 100
    ) -> list[GraphNode]:
        """
        Find nodes by label and properties.

        Args:
            label: Node label to search
            properties: Property filters
            limit: Maximum results

        Returns:
            Matching nodes
        """
        pass

    async def get_neighbors(
        self,
        node_id: str,
        rel_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 100
    ) -> list[tuple[GraphRelationship, GraphNode]]:
        """
        Get neighboring nodes.

        Args:
            node_id: Starting node ID
            rel_type: Filter by relationship type
            direction: "in", "out", or "both"
            limit: Maximum results

        Returns:
            List of (relationship, neighbor_node) tuples
        """
        pass

    async def delete_node(self, node_id: str, detach: bool = True) -> bool:
        """
        Delete a node.

        Args:
            node_id: Node to delete
            detach: Also delete relationships

        Returns:
            True if deleted
        """
        pass

    async def delete_by_stream(self, stream_id: str) -> int:
        """Delete all nodes for a stream. Returns count deleted."""
        pass

    async def clear_database(self) -> int:
        """Clear all nodes and relationships. Returns count deleted."""
        pass

    async def create_indexes(self) -> None:
        """Create recommended indexes for performance."""
        pass
```

### Usage Examples

```python
# Initialize client
config = Neo4jConfig(
    host="localhost",
    port=7687,
    username="neo4j",
    password="your_password"
)
client = Neo4jClient(config)
await client.connect()

# Create a VideoChunk node
chunk_node = await client.create_node(
    labels=["VideoChunk"],
    properties={
        "chunk_id": "chunk_001",
        "stream_id": "video_123",
        "chunk_idx": 0,
        "start_time": "00:00:00",
        "end_time": "00:01:00",
        "caption": "A forklift moves across the warehouse floor."
    }
)

# Create an Entity node
entity_node = await client.create_node(
    labels=["Entity"],
    properties={
        "entity_id": "entity_001",
        "name": "forklift",
        "type": "VEHICLE",
        "first_seen": "00:00:05",
        "last_seen": "00:00:45"
    }
)

# Create CONTAINS relationship
relationship = await client.create_relationship(
    start_node_id=chunk_node.id,
    end_node_id=entity_node.id,
    rel_type="CONTAINS",
    properties={"confidence": 0.95}
)

# Find all entities in a chunk
neighbors = await client.get_neighbors(
    node_id=chunk_node.id,
    rel_type="CONTAINS",
    direction="out"
)

# Execute custom Cypher query
results = await client.execute_query(
    """
    MATCH (c:VideoChunk)-[:CONTAINS]->(e:Entity)
    WHERE c.stream_id = $stream_id
    RETURN c, e
    """,
    parameters={"stream_id": "video_123"}
)

# Cleanup
await client.close()
```

---

## GraphIngestion API

### Data Classes

```python
from dataclasses import dataclass, field


@dataclass
class ExtractedEntity:
    """An entity extracted from caption text."""
    name: str
    type: str  # PERSON, VEHICLE, OBJECT, LOCATION
    confidence: float
    attributes: dict = field(default_factory=dict)


@dataclass
class ExtractedEvent:
    """An event extracted from caption text."""
    description: str
    event_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    participants: list[str]  # Entity names
    confidence: float
```

### GraphIngestion Class

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedEntity:
    """An entity extracted from caption text."""
    name: str
    type: str  # PERSON, VEHICLE, OBJECT, LOCATION
    confidence: float
    attributes: dict = field(default_factory=dict)


@dataclass
class ExtractedEvent:
    """An event extracted from caption text."""
    description: str
    event_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    participants: list[str]  # Entity names
    confidence: float


class GraphIngestion:
    """Handles ingestion of video data into the graph database."""

    def __init__(
        self,
        neo4j_client: "Neo4jClient",
        llm: "GeminiLLM",
        embeddings: "GeminiEmbeddings"
    ):
        """
        Initialize graph ingestion.

        Args:
            neo4j_client: Neo4j database client
            llm: LLM for entity/event extraction
            embeddings: Embedding generator for entity matching
        """
        pass

    async def extract_entities(
        self,
        caption: str,
        cv_metadata: Optional[dict] = None
    ) -> list[ExtractedEntity]:
        """
        Extract entities from caption using LLM.

        Args:
            caption: VLM-generated caption text
            cv_metadata: Optional CV detection metadata

        Returns:
            List of extracted entities
        """
        pass

    async def extract_events(
        self,
        caption: str,
        entities: list[ExtractedEntity]
    ) -> list[ExtractedEvent]:
        """
        Extract events from caption using LLM.

        Args:
            caption: VLM-generated caption text
            entities: Previously extracted entities

        Returns:
            List of extracted events
        """
        pass

    async def ingest_chunk(
        self,
        chunk_id: str,
        stream_id: str,
        chunk_idx: int,
        caption: str,
        start_time: str,
        end_time: str,
        embedding_id: str,
        cv_metadata: Optional[dict] = None
    ) -> dict:
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

        Returns:
            Summary of created nodes and relationships
        """
        pass

    async def create_temporal_links(
        self,
        stream_id: str
    ) -> int:
        """
        Create FOLLOWS relationships between sequential chunks.

        Args:
            stream_id: Stream to process

        Returns:
            Number of relationships created
        """
        pass

    async def link_entities_across_chunks(
        self,
        stream_id: str,
        similarity_threshold: float = 0.85
    ) -> int:
        """
        Link same entities across different chunks using embeddings.

        Args:
            stream_id: Stream to process
            similarity_threshold: Minimum similarity for SAME_AS

        Returns:
            Number of SAME_AS relationships created
        """
        pass

    async def ingest_batch(
        self,
        chunks: list[dict],
        stream_id: str
    ) -> dict:
        """
        Batch ingest multiple chunks.

        Args:
            chunks: List of chunk data dictionaries
            stream_id: Parent stream ID

        Returns:
            Summary of all ingested data
        """
        pass
```

### Usage Examples

```python
# Initialize ingestion
ingestion = GraphIngestion(
    neo4j_client=neo4j_client,
    llm=gemini_llm,
    embeddings=gemini_embeddings
)

# Extract entities from a caption
caption = """
00:00:05-00:00:15: A forklift carrying pallets moves across the warehouse floor.
00:00:15-00:00:30: A worker in a blue vest approaches the forklift.
"""
entities = await ingestion.extract_entities(caption)
# Returns: [
#   ExtractedEntity(name="forklift", type="VEHICLE", confidence=0.95),
#   ExtractedEntity(name="worker in blue vest", type="PERSON", confidence=0.90),
#   ExtractedEntity(name="pallets", type="OBJECT", confidence=0.85)
# ]

# Extract events
events = await ingestion.extract_events(caption, entities)
# Returns: [
#   ExtractedEvent(
#     description="Forklift transporting pallets",
#     event_type="OPERATION",
#     severity="LOW",
#     participants=["forklift", "pallets"],
#     confidence=0.9
#   )
# ]

# Ingest a complete chunk
result = await ingestion.ingest_chunk(
    chunk_id="chunk_001",
    stream_id="video_123",
    chunk_idx=0,
    caption=caption,
    start_time="00:00:00",
    end_time="00:01:00",
    embedding_id="milvus_doc_001"
)

# Create temporal links after all chunks are ingested
links_created = await ingestion.create_temporal_links("video_123")

# Link entities across chunks
same_as_links = await ingestion.link_entities_across_chunks(
    "video_123",
    similarity_threshold=0.85
)
```

---

## GraphRetrieval API

### Data Classes

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class TemporalContext:
    """Context from temporal graph traversal."""
    chunks: list[dict]
    entities: list[dict]
    events: list[dict]
    relationships: list[dict]


@dataclass
class EntityTimeline:
    """Timeline of an entity's appearances."""
    entity: dict
    appearances: list[dict]  # Chunks where entity appears
    interactions: list[dict]  # Interactions with other entities
    events: list[dict]  # Events entity participated in
```

### GraphRetrieval Class

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class TemporalContext:
    """Context from temporal graph traversal."""
    chunks: list[dict]
    entities: list[dict]
    events: list[dict]
    relationships: list[dict]


@dataclass
class EntityTimeline:
    """Timeline of an entity's appearances."""
    entity: dict
    appearances: list[dict]  # Chunks where entity appears
    interactions: list[dict]  # Interactions with other entities
    events: list[dict]  # Events entity participated in


class GraphRetrieval:
    """Handles retrieval from the graph database."""

    def __init__(
        self,
        neo4j_client: "Neo4jClient",
        embeddings: "GeminiEmbeddings"
    ):
        """
        Initialize graph retrieval.

        Args:
            neo4j_client: Neo4j database client
            embeddings: Embedding generator for query processing
        """
        pass

    async def get_temporal_context(
        self,
        stream_id: str,
        timestamp: str,
        window_before: int = 2,
        window_after: int = 2
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
        pass

    async def get_entity_timeline(
        self,
        entity_name: str,
        stream_id: Optional[str] = None
    ) -> EntityTimeline:
        """
        Get timeline of an entity's appearances.

        Args:
            entity_name: Entity to track
            stream_id: Optional stream filter

        Returns:
            Entity timeline with all appearances
        """
        pass

    async def get_related_events(
        self,
        query: str,
        stream_id: str,
        top_k: int = 5
    ) -> list[dict]:
        """
        Get events related to a query.

        Args:
            query: Natural language query
            stream_id: Stream to search
            top_k: Maximum events to return

        Returns:
            Related events with context
        """
        pass

    async def traverse_from_chunk(
        self,
        chunk_id: str,
        max_depth: int = 2,
        rel_types: Optional[list[str]] = None
    ) -> dict:
        """
        Traverse graph from a starting chunk.

        Args:
            chunk_id: Starting chunk
            max_depth: Maximum traversal depth
            rel_types: Relationship types to follow

        Returns:
            Subgraph with all reached nodes
        """
        pass

    async def find_entity_interactions(
        self,
        entity1: str,
        entity2: str,
        stream_id: Optional[str] = None
    ) -> list[dict]:
        """
        Find interactions between two entities.

        Args:
            entity1: First entity name
            entity2: Second entity name
            stream_id: Optional stream filter

        Returns:
            List of interaction events
        """
        pass

    async def get_event_context(
        self,
        event_id: str
    ) -> dict:
        """
        Get full context around an event.

        Returns:
            Event with participants, chunks, and related events
        """
        pass

    async def get_chunks_by_entity(
        self,
        entity_name: str,
        stream_id: str,
        limit: int = 10
    ) -> list[dict]:
        """Get chunks containing a specific entity."""
        pass

    async def get_chunks_by_event_type(
        self,
        event_type: str,
        stream_id: str,
        limit: int = 10
    ) -> list[dict]:
        """Get chunks containing a specific event type."""
        pass
```

### Usage Examples

```python
# Initialize retrieval
retrieval = GraphRetrieval(
    neo4j_client=neo4j_client,
    embeddings=gemini_embeddings
)

# Get temporal context around a timestamp
context = await retrieval.get_temporal_context(
    stream_id="video_123",
    timestamp="00:05:30",
    window_before=2,
    window_after=2
)
# Returns chunks from 00:03:30 to 00:07:30 with related entities and events

# Track an entity across the video
timeline = await retrieval.get_entity_timeline(
    entity_name="forklift",
    stream_id="video_123"
)
# Returns all appearances, interactions, and events for the forklift

# Find events related to a query
events = await retrieval.get_related_events(
    query="safety incidents",
    stream_id="video_123",
    top_k=5
)

# Traverse graph from a chunk
subgraph = await retrieval.traverse_from_chunk(
    chunk_id="chunk_005",
    max_depth=2,
    rel_types=["CONTAINS", "PARTICIPATES_IN"]
)

# Find interactions between entities
interactions = await retrieval.find_entity_interactions(
    entity1="forklift",
    entity2="worker",
    stream_id="video_123"
)

# Get chunks containing a specific entity
chunks = await retrieval.get_chunks_by_entity(
    entity_name="forklift",
    stream_id="video_123",
    limit=10
)

# Get chunks with specific event types
safety_chunks = await retrieval.get_chunks_by_event_type(
    event_type="SAFETY_INCIDENT",
    stream_id="video_123",
    limit=10
)
```

---

## Prompts for Entity/Event Extraction

### Entity Extraction Prompt

```
# config/prompts/entity_extraction.txt

Extract all entities from the following video caption. For each entity, provide:
- name: The entity identifier (e.g., "forklift", "worker in blue", "pallet")
- type: One of PERSON, VEHICLE, OBJECT, LOCATION
- attributes: Any descriptive attributes (color, size, position)

Caption:
{caption}

CV Metadata (if available):
{cv_metadata}

Return as JSON array:
[
  {"name": "...", "type": "...", "attributes": {...}},
  ...
]
```

### Event Extraction Prompt

```
# config/prompts/event_extraction.txt

Extract all events/actions from the following video caption. For each event:
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
  {
    "description": "...",
    "event_type": "...",
    "severity": "...",
    "participants": ["...", "..."],
    "start_time": "HH:MM:SS",
    "end_time": "HH:MM:SS"
  },
  ...
]
```

---

## Docker Compose Configuration

```yaml
# docker-compose.yaml (Neo4j section)

services:
  neo4j:
    image: neo4j:5.26.4
    container_name: vss-neo4j
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:-password}
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: apoc.*
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 30s
      timeout: 10s
      retries: 5

volumes:
  neo4j_data:
  neo4j_logs:
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_neo4j_client.py
class TestNeo4jClient:
    async def test_create_node(self): pass
    async def test_create_relationship(self): pass
    async def test_traverse_graph(self): pass

# tests/test_graph_ingestion.py
class TestGraphIngestion:
    async def test_extract_entities(self): pass
    async def test_extract_events(self): pass
    async def test_ingest_chunk(self): pass

# tests/test_graph_retrieval.py
class TestGraphRetrieval:
    async def test_temporal_context(self): pass
    async def test_entity_timeline(self): pass
    async def test_find_interactions(self): pass
```

### Integration Tests

```python
# tests/integration/test_neo4j_integration.py
class TestNeo4jIntegration:
    async def test_full_ingestion_flow(self): pass
    async def test_entity_tracking(self): pass
    async def test_temporal_queries(self): pass
```

---

## Dependencies

```
# Neo4j Python driver
neo4j>=5.0.0
```
