"""CV Metadata Fuser for combining VLM captions with CV detection metadata."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from .yolo_pipeline import FrameDetection


@dataclass
class FusedMetadata:
    """Fused metadata from VLM and CV pipelines."""

    caption: str
    cv_summary: str
    class_counts: dict[str, int]
    total_detections: int
    frames_processed: int
    raw_cv_data: Optional[dict] = None


class CVMetadataFuser:
    """Fuse CV detection metadata with VLM captions."""

    def __init__(
        self,
        include_raw_data: bool = False,
        min_confidence: float = 0.5,
    ) -> None:
        """
        Initialize CV metadata fuser.

        Args:
            include_raw_data: Include raw detection data in output
            min_confidence: Minimum confidence for counting detections
        """
        self._include_raw_data = include_raw_data
        self._min_confidence = min_confidence

    def fuse(
        self,
        captions: str,
        detections: list[FrameDetection],
    ) -> str:
        """
        Enrich captions with object detection metadata.

        Args:
            captions: VLM-generated captions
            detections: Frame-by-frame detections

        Returns:
            Enriched captions with object counts and classes
        """
        if not detections:
            return captions

        # Count objects per class across all frames
        class_counts: Counter[str] = Counter()
        total_detections = 0

        for fd in detections:
            for i, name in enumerate(fd.detections.class_names):
                if fd.detections.confidences[i] >= self._min_confidence:
                    class_counts[name] += 1
                    total_detections += 1

        if not class_counts:
            return captions

        # Create metadata summary
        metadata = "\n\n[CV Detection Summary]\n"
        metadata += f"Frames analyzed: {len(detections)}\n"
        metadata += f"Total detections: {total_detections}\n"
        metadata += "Detected objects:\n"

        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            avg_per_frame = count / len(detections)
            metadata += f"  - {class_name}: {count} instances (avg {avg_per_frame:.1f}/frame)\n"

        return captions + metadata

    def fuse_detailed(
        self,
        captions: str,
        detections: list[FrameDetection],
    ) -> FusedMetadata:
        """
        Create detailed fused metadata.

        Args:
            captions: VLM-generated captions
            detections: Frame-by-frame detections

        Returns:
            FusedMetadata with detailed information
        """
        # Count objects per class
        class_counts: Counter[str] = Counter()
        total_detections = 0

        for fd in detections:
            for i, name in enumerate(fd.detections.class_names):
                if fd.detections.confidences[i] >= self._min_confidence:
                    class_counts[name] += 1
                    total_detections += 1

        # Create summary string
        cv_summary = ""
        if class_counts:
            cv_summary = "Detected: " + ", ".join(
                f"{name} ({count})" for name, count in class_counts.most_common()
            )

        # Create raw data if requested
        raw_data: Optional[dict] = None
        if self._include_raw_data:
            raw_data = {
                "frames": [
                    {
                        "frame_idx": fd.frame_idx,
                        "timestamp": fd.timestamp,
                        "objects": [
                            {
                                "class": name,
                                "confidence": float(fd.detections.confidences[i]),
                                "bbox": fd.detections.boxes[i].tolist(),
                            }
                            for i, name in enumerate(fd.detections.class_names)
                            if fd.detections.confidences[i] >= self._min_confidence
                        ],
                    }
                    for fd in detections
                ],
            }

        return FusedMetadata(
            caption=self.fuse(captions, detections),
            cv_summary=cv_summary,
            class_counts=dict(class_counts),
            total_detections=total_detections,
            frames_processed=len(detections),
            raw_cv_data=raw_data,
        )

    def to_json_metadata(
        self,
        detections: list[FrameDetection],
    ) -> str:
        """
        Convert detections to JSON metadata string for storage.

        Args:
            detections: Frame-by-frame detections

        Returns:
            JSON string of detection metadata
        """
        class_counts: Counter[str] = Counter()

        for fd in detections:
            for i, name in enumerate(fd.detections.class_names):
                if fd.detections.confidences[i] >= self._min_confidence:
                    class_counts[name] += 1

        metadata = {
            "class_counts": dict(class_counts),
            "frames_processed": len(detections),
            "total_detections": sum(class_counts.values()),
        }

        return json.dumps(metadata)

    def fuse_from_dict(
        self,
        captions: str,
        cv_metadata: dict[str, Any],
    ) -> str:
        """
        Enrich captions with pre-aggregated CV metadata dictionary.

        This method accepts the aggregated metadata format returned by
        ViaStreamHandler._run_cv_pipeline() instead of raw FrameDetection objects.

        Args:
            captions: VLM-generated captions
            cv_metadata: Pre-aggregated CV metadata with keys:
                - class_counts: dict[str, int] - object counts per class
                - total_frames: int - number of frames processed
                - detections_per_frame: list[dict] - per-frame detection details

        Returns:
            Enriched captions with object counts and classes
        """
        if not cv_metadata:
            return captions

        class_counts = cv_metadata.get("class_counts", {})
        total_frames = cv_metadata.get("total_frames", 0)

        if not class_counts:
            return captions

        # Calculate total detections
        total_detections = sum(class_counts.values())

        # Create metadata summary
        metadata = "\n\n[CV Detection Summary]\n"
        metadata += f"Frames analyzed: {total_frames}\n"
        metadata += f"Total detections: {total_detections}\n"
        metadata += "Detected objects:\n"

        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            if total_frames > 0:
                avg_per_frame = count / total_frames
                metadata += f"  - {class_name}: {count} instances (avg {avg_per_frame:.1f}/frame)\n"
            else:
                metadata += f"  - {class_name}: {count} instances\n"

        return captions + metadata

    @staticmethod
    def from_json_metadata(json_str: str) -> dict[str, Any]:
        """
        Parse JSON metadata string.

        Args:
            json_str: JSON string from to_json_metadata

        Returns:
            Parsed metadata dictionary
        """
        if not json_str:
            return {}
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}
