"""Video Search and Summarization (VSS) PoC.

A video analysis pipeline using Google Gemini APIs (VLM, LLM, embeddings)
and YOLOv26 for object detection. Processes video files to generate
timestamped captions, extract entities/events, and enable semantic search
via hybrid RAG (Milvus vector DB + Neo4j graph DB).
"""

__version__ = "0.1.0"
