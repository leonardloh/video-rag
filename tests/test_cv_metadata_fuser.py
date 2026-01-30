"""Unit tests for CV Metadata Fuser."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.cv_pipeline.cv_metadata_fuser import CVMetadataFuser, FusedMetadata
from src.cv_pipeline.yolo_pipeline import DetectionResult, FrameDetection


class TestCVMetadataFuser:
    """Tests for CVMetadataFuser class."""

    @pytest.fixture
    def fuser(self) -> CVMetadataFuser:
        """Create a CVMetadataFuser instance."""
        return CVMetadataFuser(min_confidence=0.5)

    @pytest.fixture
    def sample_detections(self) -> list[FrameDetection]:
        """Create sample frame detections."""
        return [
            FrameDetection(
                frame_idx=0,
                timestamp=0.0,
                detections=DetectionResult(
                    boxes=np.array([[0, 0, 100, 100], [50, 50, 150, 150]]),
                    masks=None,
                    class_ids=np.array([0, 1]),
                    class_names=["person", "car"],
                    confidences=np.array([0.9, 0.8]),
                ),
            ),
            FrameDetection(
                frame_idx=1,
                timestamp=0.1,
                detections=DetectionResult(
                    boxes=np.array([[10, 10, 110, 110]]),
                    masks=None,
                    class_ids=np.array([0]),
                    class_names=["person"],
                    confidences=np.array([0.85]),
                ),
            ),
        ]

    @pytest.fixture
    def empty_detections(self) -> list[FrameDetection]:
        """Create empty frame detections."""
        return [
            FrameDetection(
                frame_idx=0,
                timestamp=0.0,
                detections=DetectionResult(
                    boxes=np.array([]).reshape(0, 4),
                    masks=None,
                    class_ids=np.array([], dtype=np.int64),
                    class_names=[],
                    confidences=np.array([]),
                ),
            ),
        ]

    def test_init(self) -> None:
        """Test CVMetadataFuser initialization."""
        fuser = CVMetadataFuser(include_raw_data=True, min_confidence=0.7)
        assert fuser._include_raw_data is True
        assert fuser._min_confidence == 0.7

    def test_fuse_empty_detections(
        self,
        fuser: CVMetadataFuser,
    ) -> None:
        """Test fuse with empty detections list."""
        caption = "A person walks in the park."
        result = fuser.fuse(caption, [])

        assert result == caption

    def test_fuse_with_detections(
        self,
        fuser: CVMetadataFuser,
        sample_detections: list[FrameDetection],
    ) -> None:
        """Test fuse with valid detections."""
        caption = "A person walks in the park."
        result = fuser.fuse(caption, sample_detections)

        assert caption in result
        assert "[CV Detection Summary]" in result
        assert "person" in result
        assert "car" in result
        assert "Frames analyzed: 2" in result

    def test_fuse_filters_low_confidence(
        self,
        fuser: CVMetadataFuser,
    ) -> None:
        """Test that fuse filters low confidence detections."""
        detections = [
            FrameDetection(
                frame_idx=0,
                timestamp=0.0,
                detections=DetectionResult(
                    boxes=np.array([[0, 0, 100, 100], [50, 50, 150, 150]]),
                    masks=None,
                    class_ids=np.array([0, 1]),
                    class_names=["person", "car"],
                    confidences=np.array([0.9, 0.3]),  # car below threshold
                ),
            ),
        ]

        caption = "Test caption"
        result = fuser.fuse(caption, detections)

        assert "person" in result
        # Car should not be in result due to low confidence
        assert "car: 1" not in result

    def test_fuse_detailed(
        self,
        fuser: CVMetadataFuser,
        sample_detections: list[FrameDetection],
    ) -> None:
        """Test fuse_detailed method."""
        caption = "A person walks in the park."
        result = fuser.fuse_detailed(caption, sample_detections)

        assert isinstance(result, FusedMetadata)
        assert caption in result.caption
        assert result.class_counts["person"] == 2
        assert result.class_counts["car"] == 1
        assert result.total_detections == 3
        assert result.frames_processed == 2
        assert "person" in result.cv_summary

    def test_fuse_detailed_with_raw_data(
        self,
        sample_detections: list[FrameDetection],
    ) -> None:
        """Test fuse_detailed with raw data included."""
        fuser = CVMetadataFuser(include_raw_data=True)
        caption = "Test caption"
        result = fuser.fuse_detailed(caption, sample_detections)

        assert result.raw_cv_data is not None
        assert "frames" in result.raw_cv_data
        assert len(result.raw_cv_data["frames"]) == 2

    def test_fuse_detailed_empty_detections(
        self,
        fuser: CVMetadataFuser,
        empty_detections: list[FrameDetection],
    ) -> None:
        """Test fuse_detailed with empty detections."""
        caption = "Test caption"
        result = fuser.fuse_detailed(caption, empty_detections)

        assert isinstance(result, FusedMetadata)
        assert result.total_detections == 0
        assert result.class_counts == {}
        assert result.cv_summary == ""

    def test_to_json_metadata(
        self,
        fuser: CVMetadataFuser,
        sample_detections: list[FrameDetection],
    ) -> None:
        """Test to_json_metadata method."""
        json_str = fuser.to_json_metadata(sample_detections)

        # Should be valid JSON
        data = json.loads(json_str)
        assert "class_counts" in data
        assert "frames_processed" in data
        assert "total_detections" in data
        assert data["class_counts"]["person"] == 2
        assert data["class_counts"]["car"] == 1

    def test_to_json_metadata_empty(
        self,
        fuser: CVMetadataFuser,
    ) -> None:
        """Test to_json_metadata with empty detections."""
        json_str = fuser.to_json_metadata([])

        data = json.loads(json_str)
        assert data["class_counts"] == {}
        assert data["frames_processed"] == 0
        assert data["total_detections"] == 0

    def test_from_json_metadata(self) -> None:
        """Test from_json_metadata static method."""
        json_str = '{"class_counts": {"person": 5}, "frames_processed": 10}'
        result = CVMetadataFuser.from_json_metadata(json_str)

        assert result["class_counts"]["person"] == 5
        assert result["frames_processed"] == 10

    def test_from_json_metadata_empty(self) -> None:
        """Test from_json_metadata with empty string."""
        result = CVMetadataFuser.from_json_metadata("")
        assert result == {}

    def test_from_json_metadata_invalid(self) -> None:
        """Test from_json_metadata with invalid JSON."""
        result = CVMetadataFuser.from_json_metadata("not valid json")
        assert result == {}

    def test_fuse_calculates_average_per_frame(
        self,
        fuser: CVMetadataFuser,
        sample_detections: list[FrameDetection],
    ) -> None:
        """Test that fuse calculates average detections per frame."""
        caption = "Test"
        result = fuser.fuse(caption, sample_detections)

        # person appears 2 times in 2 frames = 1.0 avg
        assert "1.0/frame" in result

    def test_fuse_from_dict_empty_metadata(
        self,
        fuser: CVMetadataFuser,
    ) -> None:
        """Test fuse_from_dict with empty metadata."""
        caption = "A person walks in the park."
        result = fuser.fuse_from_dict(caption, {})

        assert result == caption

    def test_fuse_from_dict_with_metadata(
        self,
        fuser: CVMetadataFuser,
    ) -> None:
        """Test fuse_from_dict with valid pre-aggregated metadata."""
        caption = "A person walks in the park."
        cv_metadata = {
            "class_counts": {"person": 5, "car": 2},
            "total_frames": 10,
            "detections_per_frame": [],
        }
        result = fuser.fuse_from_dict(caption, cv_metadata)

        assert caption in result
        assert "[CV Detection Summary]" in result
        assert "person" in result
        assert "car" in result
        assert "Frames analyzed: 10" in result
        assert "Total detections: 7" in result

    def test_fuse_from_dict_calculates_average(
        self,
        fuser: CVMetadataFuser,
    ) -> None:
        """Test fuse_from_dict calculates average per frame."""
        caption = "Test"
        cv_metadata = {
            "class_counts": {"person": 10},
            "total_frames": 5,
        }
        result = fuser.fuse_from_dict(caption, cv_metadata)

        # 10 detections / 5 frames = 2.0 avg
        assert "2.0/frame" in result

    def test_fuse_from_dict_empty_class_counts(
        self,
        fuser: CVMetadataFuser,
    ) -> None:
        """Test fuse_from_dict with empty class_counts."""
        caption = "Test caption"
        cv_metadata = {
            "class_counts": {},
            "total_frames": 10,
        }
        result = fuser.fuse_from_dict(caption, cv_metadata)

        # Should return original caption without modification
        assert result == caption

    def test_fuse_from_dict_zero_frames(
        self,
        fuser: CVMetadataFuser,
    ) -> None:
        """Test fuse_from_dict handles zero frames gracefully."""
        caption = "Test"
        cv_metadata = {
            "class_counts": {"person": 5},
            "total_frames": 0,
        }
        result = fuser.fuse_from_dict(caption, cv_metadata)

        # Should still include the class counts but no avg/frame
        assert "person: 5 instances" in result
        assert "/frame" not in result
