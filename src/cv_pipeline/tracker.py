"""Object Tracker using ByteTrack for multi-object tracking."""

from __future__ import annotations

from typing import Optional

import numpy as np
import supervision as sv

from .yolo_pipeline import DetectionResult, TrackedObject


class ObjectTracker:
    """Multi-object tracker using ByteTrack."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ) -> None:
        """
        Initialize tracker.

        Args:
            track_thresh: Detection threshold for tracking
            track_buffer: Frames to keep lost tracks
            match_thresh: IoU threshold for matching
        """
        self._track_thresh = track_thresh
        self._track_buffer = track_buffer
        self._match_thresh = match_thresh

        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
        )
        self._frame_idx = 0
        self._track_ages: dict[int, int] = {}  # track_id -> first_seen_frame
        self._track_hits: dict[int, int] = {}  # track_id -> hit count

    def update(
        self,
        detections: DetectionResult,
        frame_idx: Optional[int] = None,
    ) -> list[TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: Detection result from YOLO
            frame_idx: Optional frame index

        Returns:
            List of tracked objects with IDs
        """
        if frame_idx is not None:
            self._frame_idx = frame_idx
        else:
            self._frame_idx += 1

        # Handle empty detections
        if detections.num_detections == 0:
            return []

        # Convert to supervision format
        sv_detections = sv.Detections(
            xyxy=detections.boxes,
            confidence=detections.confidences,
            class_id=detections.class_ids,
            mask=detections.masks,
        )

        # Update tracker
        tracked = self._tracker.update_with_detections(sv_detections)

        # Convert to TrackedObject list
        tracked_objects: list[TrackedObject] = []

        if tracked.tracker_id is None or len(tracked) == 0:
            return tracked_objects

        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])

            # Update track statistics
            if track_id not in self._track_ages:
                self._track_ages[track_id] = self._frame_idx
                self._track_hits[track_id] = 0
            self._track_hits[track_id] += 1

            # Get class name
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            class_name = ""
            if i < len(detections.class_names):
                class_name = detections.class_names[i]

            # Get mask if available
            mask = None
            if tracked.mask is not None and i < len(tracked.mask):
                mask = tracked.mask[i]

            tracked_objects.append(TrackedObject(
                track_id=track_id,
                box=tracked.xyxy[i],
                mask=mask,
                class_id=class_id,
                class_name=class_name,
                confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                age=self._frame_idx - self._track_ages[track_id],
                hits=self._track_hits[track_id],
            ))

        return tracked_objects

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self._track_thresh,
            lost_track_buffer=self._track_buffer,
            minimum_matching_threshold=self._match_thresh,
        )
        self._frame_idx = 0
        self._track_ages.clear()
        self._track_hits.clear()

    @property
    def frame_idx(self) -> int:
        """Get current frame index."""
        return self._frame_idx

    @property
    def active_tracks(self) -> int:
        """Get number of active tracks."""
        return len(self._track_ages)
