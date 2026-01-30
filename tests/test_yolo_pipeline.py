"""Unit tests for YOLO Pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.cv_pipeline.yolo_pipeline import (
    DetectionResult,
    FrameDetection,
    TrackedObject,
    TrackingResult,
    YOLOPipeline,
)


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_num_detections(self) -> None:
        """Test num_detections property."""
        result = DetectionResult(
            boxes=np.array([[0, 0, 100, 100], [50, 50, 150, 150]]),
            masks=None,
            class_ids=np.array([0, 1]),
            class_names=["person", "car"],
            confidences=np.array([0.9, 0.8]),
        )
        assert result.num_detections == 2

    def test_num_detections_empty(self) -> None:
        """Test num_detections with empty result."""
        result = DetectionResult(
            boxes=np.array([]).reshape(0, 4),
            masks=None,
            class_ids=np.array([], dtype=np.int64),
            class_names=[],
            confidences=np.array([]),
        )
        assert result.num_detections == 0

    def test_filter_by_confidence(self) -> None:
        """Test filtering by confidence threshold."""
        result = DetectionResult(
            boxes=np.array([[0, 0, 100, 100], [50, 50, 150, 150], [100, 100, 200, 200]]),
            masks=None,
            class_ids=np.array([0, 1, 2]),
            class_names=["person", "car", "truck"],
            confidences=np.array([0.9, 0.5, 0.3]),
        )

        filtered = result.filter_by_confidence(0.6)
        assert filtered.num_detections == 1
        assert filtered.class_names == ["person"]
        assert filtered.confidences[0] == 0.9

    def test_filter_by_confidence_with_masks(self) -> None:
        """Test filtering by confidence with masks."""
        masks = np.random.rand(3, 100, 100)
        result = DetectionResult(
            boxes=np.array([[0, 0, 100, 100], [50, 50, 150, 150], [100, 100, 200, 200]]),
            masks=masks,
            class_ids=np.array([0, 1, 2]),
            class_names=["person", "car", "truck"],
            confidences=np.array([0.9, 0.5, 0.3]),
        )

        filtered = result.filter_by_confidence(0.6)
        assert filtered.masks is not None
        assert filtered.masks.shape[0] == 1

    def test_filter_by_classes(self) -> None:
        """Test filtering by class names."""
        result = DetectionResult(
            boxes=np.array([[0, 0, 100, 100], [50, 50, 150, 150], [100, 100, 200, 200]]),
            masks=None,
            class_ids=np.array([0, 1, 2]),
            class_names=["person", "car", "truck"],
            confidences=np.array([0.9, 0.8, 0.7]),
        )

        filtered = result.filter_by_classes(["person", "truck"])
        assert filtered.num_detections == 2
        assert "person" in filtered.class_names
        assert "truck" in filtered.class_names
        assert "car" not in filtered.class_names

    def test_filter_by_classes_no_match(self) -> None:
        """Test filtering by classes with no matches."""
        result = DetectionResult(
            boxes=np.array([[0, 0, 100, 100]]),
            masks=None,
            class_ids=np.array([0]),
            class_names=["person"],
            confidences=np.array([0.9]),
        )

        filtered = result.filter_by_classes(["car", "truck"])
        assert filtered.num_detections == 0
        assert len(filtered.class_names) == 0


class TestFrameDetection:
    """Tests for FrameDetection dataclass."""

    def test_frame_detection_creation(self) -> None:
        """Test creating a FrameDetection."""
        detection = DetectionResult(
            boxes=np.array([[0, 0, 100, 100]]),
            masks=None,
            class_ids=np.array([0]),
            class_names=["person"],
            confidences=np.array([0.9]),
        )

        frame_det = FrameDetection(
            frame_idx=10,
            timestamp=0.5,
            detections=detection,
        )

        assert frame_det.frame_idx == 10
        assert frame_det.timestamp == 0.5
        assert frame_det.detections.num_detections == 1


class TestTrackedObject:
    """Tests for TrackedObject dataclass."""

    def test_tracked_object_creation(self) -> None:
        """Test creating a TrackedObject."""
        obj = TrackedObject(
            track_id=1,
            box=np.array([0, 0, 100, 100]),
            mask=None,
            class_id=0,
            class_name="person",
            confidence=0.9,
            age=5,
            hits=10,
        )

        assert obj.track_id == 1
        assert obj.class_name == "person"
        assert obj.age == 5
        assert obj.hits == 10


class TestTrackingResult:
    """Tests for TrackingResult dataclass."""

    def test_tracking_result_creation(self) -> None:
        """Test creating a TrackingResult."""
        objects = [
            TrackedObject(
                track_id=1,
                box=np.array([0, 0, 100, 100]),
                mask=None,
                class_id=0,
                class_name="person",
                confidence=0.9,
                age=5,
                hits=10,
            )
        ]

        result = TrackingResult(
            frame_idx=10,
            timestamp=0.5,
            tracked_objects=objects,
        )

        assert result.frame_idx == 10
        assert len(result.tracked_objects) == 1


class TestYOLOPipeline:
    """Tests for YOLOPipeline class."""

    @pytest.fixture
    def mock_yolo_model(self) -> MagicMock:
        """Create a mock YOLO model."""
        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "car", 2: "truck"}
        mock_model.to = MagicMock(return_value=mock_model)
        return mock_model

    @pytest.fixture
    def mock_yolo_result(self) -> MagicMock:
        """Create a mock YOLO result."""
        mock_boxes = MagicMock()
        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[0, 0, 100, 100]])
        mock_boxes.conf = MagicMock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
        mock_boxes.cls = MagicMock()
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
        mock_boxes.__len__ = MagicMock(return_value=1)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.masks = None

        return mock_result

    @patch("ultralytics.YOLO")
    def test_init(self, mock_yolo_class: MagicMock, mock_yolo_model: MagicMock) -> None:
        """Test YOLOPipeline initialization."""
        mock_yolo_class.return_value = mock_yolo_model

        pipeline = YOLOPipeline(
            model="yolov8n-seg",
            confidence=0.5,
            device="cpu",
        )

        assert pipeline._confidence == 0.5
        assert pipeline._device == "cpu"
        mock_yolo_class.assert_called_once()

    @patch("ultralytics.YOLO")
    def test_detect(
        self,
        mock_yolo_class: MagicMock,
        mock_yolo_model: MagicMock,
        mock_yolo_result: MagicMock,
    ) -> None:
        """Test single frame detection."""
        mock_yolo_class.return_value = mock_yolo_model
        mock_yolo_model.return_value = [mock_yolo_result]

        pipeline = YOLOPipeline(model="yolov8n-seg", device="cpu")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.detect(frame)

        assert isinstance(result, DetectionResult)
        assert result.num_detections == 1
        assert result.class_names[0] == "person"

    @patch("ultralytics.YOLO")
    def test_detect_empty(
        self,
        mock_yolo_class: MagicMock,
        mock_yolo_model: MagicMock,
    ) -> None:
        """Test detection with no results."""
        mock_yolo_class.return_value = mock_yolo_model

        # Create empty result
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_result.masks = None
        mock_yolo_model.return_value = [mock_result]

        pipeline = YOLOPipeline(model="yolov8n-seg", device="cpu")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.detect(frame)

        assert isinstance(result, DetectionResult)
        assert result.num_detections == 0

    @patch("ultralytics.YOLO")
    def test_detect_with_target_classes(
        self,
        mock_yolo_class: MagicMock,
        mock_yolo_model: MagicMock,
    ) -> None:
        """Test detection with target class filtering."""
        mock_yolo_class.return_value = mock_yolo_model

        # Create result with multiple classes
        mock_boxes = MagicMock()
        mock_boxes.xyxy = MagicMock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [0, 0, 100, 100],
            [50, 50, 150, 150],
        ])
        mock_boxes.conf = MagicMock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])
        mock_boxes.cls = MagicMock()
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 1])
        mock_boxes.__len__ = MagicMock(return_value=2)

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.masks = None
        mock_yolo_model.return_value = [mock_result]

        pipeline = YOLOPipeline(
            model="yolov8n-seg",
            device="cpu",
            target_classes=["person"],
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.detect(frame)

        assert result.num_detections == 1
        assert result.class_names[0] == "person"

    @patch("ultralytics.YOLO")
    def test_detect_batch(
        self,
        mock_yolo_class: MagicMock,
        mock_yolo_model: MagicMock,
        mock_yolo_result: MagicMock,
    ) -> None:
        """Test batch detection."""
        mock_yolo_class.return_value = mock_yolo_model
        mock_yolo_model.return_value = [mock_yolo_result, mock_yolo_result]

        pipeline = YOLOPipeline(model="yolov8n-seg", device="cpu")

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        results = pipeline.detect_batch(frames)

        assert len(results) == 2
        assert all(isinstance(r, DetectionResult) for r in results)

    @patch("ultralytics.YOLO")
    def test_class_names_property(
        self,
        mock_yolo_class: MagicMock,
        mock_yolo_model: MagicMock,
    ) -> None:
        """Test class_names property."""
        mock_yolo_class.return_value = mock_yolo_model

        pipeline = YOLOPipeline(model="yolov8n-seg", device="cpu")
        class_names = pipeline.class_names

        assert isinstance(class_names, list)
        assert "person" in class_names
        assert "car" in class_names

    @patch("ultralytics.YOLO")
    def test_draw_annotations(
        self,
        mock_yolo_class: MagicMock,
        mock_yolo_model: MagicMock,
    ) -> None:
        """Test annotation drawing."""
        mock_yolo_class.return_value = mock_yolo_model

        pipeline = YOLOPipeline(model="yolov8n-seg", device="cpu")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = DetectionResult(
            boxes=np.array([[50, 50, 150, 150]]),
            masks=None,
            class_ids=np.array([0]),
            class_names=["person"],
            confidences=np.array([0.9]),
        )

        annotated = pipeline._draw_annotations(frame, detection)

        assert annotated.shape == frame.shape
        # Check that some pixels were modified (annotations drawn)
        assert not np.array_equal(annotated, frame)

    def test_model_variants(self) -> None:
        """Test that model variants are defined."""
        assert "yolov8n-seg" in YOLOPipeline.MODEL_VARIANTS
        assert "yolov8s-seg" in YOLOPipeline.MODEL_VARIANTS
        assert "yolov26n-seg" in YOLOPipeline.MODEL_VARIANTS

    def test_coco_classes(self) -> None:
        """Test that COCO classes are defined."""
        assert len(YOLOPipeline.COCO_CLASSES) == 80
        assert "person" in YOLOPipeline.COCO_CLASSES
        assert "car" in YOLOPipeline.COCO_CLASSES
        assert "truck" in YOLOPipeline.COCO_CLASSES
