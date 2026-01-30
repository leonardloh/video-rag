"""Unit tests for Object Tracker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.cv_pipeline.tracker import ObjectTracker
from src.cv_pipeline.yolo_pipeline import DetectionResult, TrackedObject


class TestObjectTracker:
    """Tests for ObjectTracker class."""

    @pytest.fixture
    def mock_bytetrack(self) -> MagicMock:
        """Create a mock ByteTrack tracker."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def sample_detection(self) -> DetectionResult:
        """Create a sample detection result."""
        return DetectionResult(
            boxes=np.array([[0, 0, 100, 100], [50, 50, 150, 150]]),
            masks=None,
            class_ids=np.array([0, 1]),
            class_names=["person", "car"],
            confidences=np.array([0.9, 0.8]),
        )

    @pytest.fixture
    def empty_detection(self) -> DetectionResult:
        """Create an empty detection result."""
        return DetectionResult(
            boxes=np.array([]).reshape(0, 4),
            masks=None,
            class_ids=np.array([], dtype=np.int64),
            class_names=[],
            confidences=np.array([]),
        )

    @patch("src.cv_pipeline.tracker.sv.ByteTrack")
    def test_init(self, mock_bytetrack_class: MagicMock) -> None:
        """Test ObjectTracker initialization."""
        tracker = ObjectTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
        )

        assert tracker._track_thresh == 0.5
        assert tracker._track_buffer == 30
        assert tracker._match_thresh == 0.8
        assert tracker._frame_idx == 0
        mock_bytetrack_class.assert_called_once()

    @patch("src.cv_pipeline.tracker.sv.ByteTrack")
    def test_update_empty_detections(
        self,
        mock_bytetrack_class: MagicMock,
        empty_detection: DetectionResult,
    ) -> None:
        """Test update with empty detections."""
        tracker = ObjectTracker()
        result = tracker.update(empty_detection)

        assert result == []

    @patch("src.cv_pipeline.tracker.sv.ByteTrack")
    def test_update_with_detections(
        self,
        mock_bytetrack_class: MagicMock,
        sample_detection: DetectionResult,
    ) -> None:
        """Test update with valid detections."""
        # Setup mock tracker response
        mock_tracker = MagicMock()
        mock_tracked = MagicMock()
        mock_tracked.tracker_id = np.array([1, 2])
        mock_tracked.xyxy = np.array([[0, 0, 100, 100], [50, 50, 150, 150]])
        mock_tracked.confidence = np.array([0.9, 0.8])
        mock_tracked.class_id = np.array([0, 1])
        mock_tracked.mask = None
        mock_tracked.__len__ = MagicMock(return_value=2)
        mock_tracker.update_with_detections.return_value = mock_tracked
        mock_bytetrack_class.return_value = mock_tracker

        tracker = ObjectTracker()
        result = tracker.update(sample_detection)

        assert len(result) == 2
        assert all(isinstance(obj, TrackedObject) for obj in result)
        assert result[0].track_id == 1
        assert result[1].track_id == 2

    @patch("src.cv_pipeline.tracker.sv.ByteTrack")
    def test_update_increments_frame_idx(
        self,
        mock_bytetrack_class: MagicMock,
        empty_detection: DetectionResult,
    ) -> None:
        """Test that update increments frame index."""
        tracker = ObjectTracker()

        assert tracker._frame_idx == 0
        tracker.update(empty_detection)
        assert tracker._frame_idx == 1
        tracker.update(empty_detection)
        assert tracker._frame_idx == 2

    @patch("src.cv_pipeline.tracker.sv.ByteTrack")
    def test_update_with_explicit_frame_idx(
        self,
        mock_bytetrack_class: MagicMock,
        empty_detection: DetectionResult,
    ) -> None:
        """Test update with explicit frame index."""
        tracker = ObjectTracker()

        tracker.update(empty_detection, frame_idx=10)
        assert tracker._frame_idx == 10

    @patch("src.cv_pipeline.tracker.sv.ByteTrack")
    def test_reset(self, mock_bytetrack_class: MagicMock) -> None:
        """Test tracker reset."""
        tracker = ObjectTracker()
        tracker._frame_idx = 100
        tracker._track_ages[1] = 50
        tracker._track_hits[1] = 10

        tracker.reset()

        assert tracker._frame_idx == 0
        assert len(tracker._track_ages) == 0
        assert len(tracker._track_hits) == 0
        # ByteTrack should be recreated
        assert mock_bytetrack_class.call_count == 2

    @patch("src.cv_pipeline.tracker.sv.ByteTrack")
    def test_frame_idx_property(self, mock_bytetrack_class: MagicMock) -> None:
        """Test frame_idx property."""
        tracker = ObjectTracker()
        tracker._frame_idx = 42

        assert tracker.frame_idx == 42

    @patch("src.cv_pipeline.tracker.sv.ByteTrack")
    def test_active_tracks_property(self, mock_bytetrack_class: MagicMock) -> None:
        """Test active_tracks property."""
        tracker = ObjectTracker()
        tracker._track_ages = {1: 0, 2: 5, 3: 10}

        assert tracker.active_tracks == 3

    @patch("src.cv_pipeline.tracker.sv.ByteTrack")
    def test_track_age_calculation(
        self,
        mock_bytetrack_class: MagicMock,
        sample_detection: DetectionResult,
    ) -> None:
        """Test that track age is calculated correctly."""
        # Setup mock
        mock_tracker = MagicMock()
        mock_tracked = MagicMock()
        mock_tracked.tracker_id = np.array([1])
        mock_tracked.xyxy = np.array([[0, 0, 100, 100]])
        mock_tracked.confidence = np.array([0.9])
        mock_tracked.class_id = np.array([0])
        mock_tracked.mask = None
        mock_tracked.__len__ = MagicMock(return_value=1)
        mock_tracker.update_with_detections.return_value = mock_tracked
        mock_bytetrack_class.return_value = mock_tracker

        tracker = ObjectTracker()

        # First detection at frame 0
        result1 = tracker.update(sample_detection, frame_idx=0)
        assert result1[0].age == 0

        # Same track at frame 5
        result2 = tracker.update(sample_detection, frame_idx=5)
        assert result2[0].age == 5

    @patch("src.cv_pipeline.tracker.sv.ByteTrack")
    def test_track_hits_increment(
        self,
        mock_bytetrack_class: MagicMock,
        sample_detection: DetectionResult,
    ) -> None:
        """Test that track hits are incremented correctly."""
        # Setup mock
        mock_tracker = MagicMock()
        mock_tracked = MagicMock()
        mock_tracked.tracker_id = np.array([1])
        mock_tracked.xyxy = np.array([[0, 0, 100, 100]])
        mock_tracked.confidence = np.array([0.9])
        mock_tracked.class_id = np.array([0])
        mock_tracked.mask = None
        mock_tracked.__len__ = MagicMock(return_value=1)
        mock_tracker.update_with_detections.return_value = mock_tracked
        mock_bytetrack_class.return_value = mock_tracker

        tracker = ObjectTracker()

        result1 = tracker.update(sample_detection)
        assert result1[0].hits == 1

        result2 = tracker.update(sample_detection)
        assert result2[0].hits == 2

        result3 = tracker.update(sample_detection)
        assert result3[0].hits == 3
