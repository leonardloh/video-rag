"""Chunk metadata for video processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkInfo:
    """Metadata for a video chunk."""

    chunkIdx: int  # Chunk index within the video
    streamId: str  # Video stream identifier
    file: str  # Path to the chunk file
    start_ntp: str  # Start timestamp in HH:MM:SS format
    end_ntp: str  # End timestamp in HH:MM:SS format
    start_pts: int = 0  # Presentation timestamp start (nanoseconds)
    end_pts: int = 0  # Presentation timestamp end (nanoseconds)
    duration: Optional[float] = None  # Duration in seconds

    @property
    def start_seconds(self) -> float:
        """Convert start_ntp to seconds."""
        return self._timestamp_to_seconds(self.start_ntp)

    @property
    def end_seconds(self) -> float:
        """Convert end_ntp to seconds."""
        return self._timestamp_to_seconds(self.end_ntp)

    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        if self.duration is not None:
            return self.duration
        return self.end_seconds - self.start_seconds

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """Convert HH:MM:SS or HH:MM:SS.mmm to seconds."""
        parts = timestamp.split(":")
        if len(parts) != 3:
            return 0.0

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])

        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def seconds_to_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def seconds_to_pts(seconds: float) -> int:
        """Convert seconds to presentation timestamp (nanoseconds)."""
        return int(seconds * 1_000_000_000)

    @staticmethod
    def pts_to_seconds(pts: int) -> float:
        """Convert presentation timestamp (nanoseconds) to seconds."""
        return pts / 1_000_000_000

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "chunkIdx": self.chunkIdx,
            "streamId": self.streamId,
            "file": self.file,
            "start_ntp": self.start_ntp,
            "end_ntp": self.end_ntp,
            "start_pts": self.start_pts,
            "end_pts": self.end_pts,
            "duration": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChunkInfo:
        """Create from dictionary."""
        return cls(
            chunkIdx=data["chunkIdx"],
            streamId=data["streamId"],
            file=data["file"],
            start_ntp=data["start_ntp"],
            end_ntp=data["end_ntp"],
            start_pts=data.get("start_pts", 0),
            end_pts=data.get("end_pts", 0),
            duration=data.get("duration"),
        )
