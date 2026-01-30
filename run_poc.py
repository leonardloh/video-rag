#!/usr/bin/env python3
"""Entry point for VSS PoC - Video Search and Summarization."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    from src.config import load_config as _load_config

    return _load_config(config_path)


def load_prompt(prompt_path: str) -> str:
    """Load prompt from file."""
    with open(prompt_path) as f:
        return f.read().strip()


def get_default_prompt() -> str:
    """Get default VLM prompt."""
    prompt_path = Path(__file__).parent / "config" / "prompts" / "caption.txt"
    if prompt_path.exists():
        return load_prompt(str(prompt_path))
    return """Analyze this video segment and provide detailed captions of all events.
For each distinct event or action, provide:
- Start and end timestamps in format <HH:MM:SS>
- Detailed description of what is happening
- Any relevant objects, people, or text visible
Focus on actions, interactions, and notable occurrences."""


def get_chat_prompt() -> str:
    """Get chat system prompt."""
    prompt_path = Path(__file__).parent / "config" / "prompts" / "chat.txt"
    if prompt_path.exists():
        return load_prompt(str(prompt_path))
    return """You are a video analysis assistant. You have access to detailed captions,
object detection data, and a knowledge graph of entities and events from a video.
Answer questions accurately based on the provided context.
If information is not available in the context, say so clearly.
Always reference specific timestamps when relevant."""


async def initialize_components(
    config: dict[str, Any],
    enable_cv: bool = False,
    enable_milvus: bool = False,
    enable_neo4j: bool = False,
) -> dict[str, Any]:
    """Initialize all components."""
    from src.asset_manager import AssetManager
    from src.models.gemini.gemini_embeddings import GeminiEmbeddings
    from src.models.gemini.gemini_file_manager import GeminiFileManager
    from src.models.gemini.gemini_llm import GeminiLLM
    from src.models.gemini.gemini_vlm import GeminiVLM

    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    # Initialize Gemini components
    gemini_config = config.get("gemini", {})

    file_manager = GeminiFileManager(
        api_key=api_key,
        max_concurrent_uploads=config.get("processing", {}).get("max_concurrent_chunks", 3),
    )

    vlm = GeminiVLM(
        api_key=api_key,
        model=gemini_config.get("vlm", {}).get("model", "gemini-2.0-flash"),
    )

    llm = GeminiLLM(
        api_key=api_key,
        model=gemini_config.get("llm", {}).get("model", "gemini-2.0-flash"),
    )

    embeddings = GeminiEmbeddings(
        api_key=api_key,
        model=gemini_config.get("embeddings", {}).get("model", "text-embedding-004"),
    )

    # Initialize asset manager
    asset_manager = AssetManager()

    components: dict[str, Any] = {
        "asset_manager": asset_manager,
        "file_manager": file_manager,
        "vlm": vlm,
        "llm": llm,
        "embeddings": embeddings,
        "yolo_pipeline": None,
        "milvus_client": None,
        "neo4j_client": None,
    }

    # Initialize YOLO pipeline if enabled
    if enable_cv:
        try:
            from src.cv_pipeline.yolo_pipeline import YOLOPipeline

            cv_config = config.get("cv_pipeline", {}).get("yolo", {})
            components["yolo_pipeline"] = YOLOPipeline(
                model=cv_config.get("model", "yolov8n-seg"),
                confidence=cv_config.get("confidence", 0.5),
                device=cv_config.get("device", "cpu"),
            )
            logging.info("YOLO pipeline initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize YOLO pipeline: {e}")

    # Initialize Milvus if enabled
    if enable_milvus:
        try:
            from src.db.milvus_client import MilvusClient, MilvusConfig

            milvus_cfg = config.get("rag", {}).get("vector_db", {})
            milvus_config = MilvusConfig(
                host=milvus_cfg.get("host", os.environ.get("MILVUS_HOST", "localhost")),
                port=milvus_cfg.get("port", int(os.environ.get("MILVUS_PORT", "19530"))),
                collection_name=milvus_cfg.get("collection_name", "vss_poc_captions"),
            )
            milvus_client = MilvusClient(config=milvus_config)
            await milvus_client.connect()
            await milvus_client.ensure_collection()
            components["milvus_client"] = milvus_client
            logging.info("Milvus client initialized and connected")
        except Exception as e:
            logging.warning(f"Failed to initialize Milvus client: {e}")

    # Initialize Neo4j if enabled
    if enable_neo4j:
        try:
            from src.db.neo4j_client import Neo4jClient, Neo4jConfig

            neo4j_cfg = config.get("rag", {}).get("graph_db", {})
            neo4j_config = Neo4jConfig(
                host=neo4j_cfg.get("host", os.environ.get("NEO4J_HOST", "localhost")),
                port=neo4j_cfg.get("port", int(os.environ.get("NEO4J_BOLT_PORT", "7687"))),
                username=neo4j_cfg.get("username", os.environ.get("NEO4J_USERNAME", "neo4j")),
                password=neo4j_cfg.get("password", os.environ.get("NEO4J_PASSWORD", "")),
            )
            neo4j_client = Neo4jClient(config=neo4j_config)
            await neo4j_client.connect()
            await neo4j_client.create_indexes()
            components["neo4j_client"] = neo4j_client
            logging.info("Neo4j client initialized and connected")
        except Exception as e:
            logging.warning(f"Failed to initialize Neo4j client: {e}")

    return components


async def process_video(
    video_path: str,
    config: dict[str, Any],
    prompt: Optional[str] = None,
    enable_cv: bool = False,
    enable_milvus: bool = False,
    enable_neo4j: bool = False,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Process a video file."""
    from src.via_stream_handler import ViaStreamHandler

    # Initialize components
    components = await initialize_components(
        config=config,
        enable_cv=enable_cv,
        enable_milvus=enable_milvus,
        enable_neo4j=enable_neo4j,
    )

    # Create stream handler
    handler = ViaStreamHandler(
        asset_manager=components["asset_manager"],
        gemini_file_manager=components["file_manager"],
        gemini_vlm=components["vlm"],
        gemini_llm=components["llm"],
        gemini_embeddings=components["embeddings"],
        yolo_pipeline=components["yolo_pipeline"],
        milvus_client=components["milvus_client"],
        neo4j_client=components["neo4j_client"],
        config=config,
    )

    # Progress callback
    def on_progress(progress: float) -> None:
        print(f"\rProgress: {progress:.1f}%", end="", flush=True)

    # Chunk callback
    def on_chunk_complete(result: Any) -> None:
        print(f"\nChunk {result.chunk.chunkIdx} completed in {result.processing_time:.2f}s")
        if result.error:
            print(f"  Error: {result.error}")

    # Process video
    vlm_prompt = prompt or get_default_prompt()
    processing_config = config.get("processing", {})

    print(f"Processing video: {video_path}")
    print(f"Chunk duration: {processing_config.get('chunk_duration', 60)}s")
    print(f"CV pipeline: {'enabled' if enable_cv else 'disabled'}")
    print()

    request = await handler.process_video(
        file_path=video_path,
        vlm_prompt=vlm_prompt,
        chunk_duration=processing_config.get("chunk_duration", 60),
        chunk_overlap=processing_config.get("chunk_overlap", 2),
        enable_cv_pipeline=enable_cv,
        on_progress=on_progress,
        on_chunk_complete=on_chunk_complete,
    )

    print(f"\n\nProcessing complete: {request.status.value}")
    print(f"Chunks processed: {len(request.processed_chunks)}/{request.chunk_count}")

    # Build result
    result = {
        "request_id": request.request_id,
        "stream_id": request.stream_id,
        "status": request.status.value,
        "chunk_count": request.chunk_count,
        "captions": [
            {
                "chunk_idx": r.chunk.chunkIdx,
                "start_time": r.chunk.start_ntp,
                "end_time": r.chunk.end_ntp,
                "caption": r.vlm_response,
                "cv_metadata": r.cv_metadata,
            }
            for r in request.processed_chunks
            if r.vlm_response
        ],
    }

    # Save result if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"{request.request_id}_captions.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_file}")

    return {"handler": handler, "request": request, "result": result}


async def run_summarization(
    handler: Any,
    request: Any,
    config: dict[str, Any],
) -> str:
    """Generate summary for processed video."""
    print("\nGenerating summary...")

    summary = await handler.summarize(
        request_id=request.request_id,
        batch_size=config.get("rag", {}).get("batch_size", 6),
    )

    print("\n" + "=" * 60)
    print("VIDEO SUMMARY")
    print("=" * 60)
    print(summary)
    print("=" * 60)

    return summary


async def run_chat(
    handler: Any,
    request: Any,
) -> None:
    """Run interactive chat about the video."""
    print("\n" + "=" * 60)
    print("INTERACTIVE CHAT")
    print("Type 'quit' or 'exit' to end the chat")
    print("=" * 60 + "\n")

    chat_history: list[dict[str, str]] = []
    system_prompt = get_chat_prompt()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat...")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            answer = await handler.chat(
                request_id=request.request_id,
                question=question,
                chat_history=chat_history,
                system_prompt=system_prompt,
            )

            print(f"\nAssistant: {answer}\n")

            # Update chat history
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"\nError: {e}\n")


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VSS PoC - Video Search and Summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video and generate captions
  python run_poc.py --video /path/to/video.mp4

  # Process with CV pipeline enabled
  python run_poc.py --video /path/to/video.mp4 --enable-cv

  # Generate summary after processing
  python run_poc.py --video /path/to/video.mp4 --summarize

  # Interactive chat mode
  python run_poc.py --video /path/to/video.mp4 --chat

  # Full pipeline with all features
  python run_poc.py --video /path/to/video.mp4 --enable-cv --summarize --chat

Environment Variables:
  GEMINI_API_KEY     Google AI Studio API key (required)
  MILVUS_HOST        Milvus host (default: localhost)
  MILVUS_PORT        Milvus port (default: 19530)
  NEO4J_HOST         Neo4j host (default: localhost)
  NEO4J_BOLT_PORT    Neo4j Bolt port (default: 7687)
  NEO4J_USERNAME     Neo4j username (default: neo4j)
  NEO4J_PASSWORD     Neo4j password
""",
    )

    parser.add_argument(
        "--video",
        "-v",
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        help="Custom VLM prompt (or path to prompt file)",
    )
    parser.add_argument(
        "--enable-cv",
        action="store_true",
        help="Enable CV pipeline (YOLO object detection)",
    )
    parser.add_argument(
        "--enable-milvus",
        action="store_true",
        help="Enable Milvus vector database",
    )
    parser.add_argument(
        "--enable-neo4j",
        action="store_true",
        help="Enable Neo4j graph database",
    )
    parser.add_argument(
        "--summarize",
        "-s",
        action="store_true",
        help="Generate video summary after processing",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Enter interactive chat mode after processing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        logging.warning(f"Config file not found: {args.config}, using defaults")
        config = {}

    # Load custom prompt if provided
    prompt = None
    if args.prompt:
        prompt_path = Path(args.prompt)
        if prompt_path.exists():
            prompt = load_prompt(str(prompt_path))
        else:
            prompt = args.prompt

    try:
        # Process video
        result = await process_video(
            video_path=str(video_path),
            config=config,
            prompt=prompt,
            enable_cv=args.enable_cv,
            enable_milvus=args.enable_milvus,
            enable_neo4j=args.enable_neo4j,
            output_dir=args.output,
        )

        handler = result["handler"]
        request = result["request"]

        # Generate summary if requested
        if args.summarize:
            summary = await run_summarization(handler, request, config)

            # Save summary
            if args.output:
                summary_file = Path(args.output) / f"{request.request_id}_summary.txt"
                with open(summary_file, "w") as f:
                    f.write(summary)
                print(f"Summary saved to: {summary_file}")

        # Enter chat mode if requested
        if args.chat:
            await run_chat(handler, request)

        # Cleanup
        handler.cleanup_request(request.request_id)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
