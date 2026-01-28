# YOLOv26 Pipeline Specification

## Overview

The `YOLOPipeline` class provides object detection and segmentation using YOLOv26-seg models. This replaces the Grounding DINO + SAM (TensorRT/DeepStream) pipeline used in the original VSS engine.

## Gap Analysis

### Original Implementation
- `src/vss-engine/src/cv_pipeline/cv_pipeline.py` - `CVPipeline` orchestrates detection
- `src/vss-engine/src/cv_pipeline/gsam_pipeline_trt_ds.py` - Grounding DINO + SAM with TensorRT
- `src/vss-engine/src/cv_pipeline/gsam_model_trt.py` - TensorRT model wrapper
- Uses DeepStream for video processing
- Downloads models from NGC
- Complex TensorRT engine building

### PoC Requirement
- Use YOLOv26-seg from Ultralytics
- Simpler setup (pip install, no TensorRT build)
- Support detection, segmentation, and tracking
- Filter by target classes

## Component Location

```
./src/cv_pipeline/yolo_pipeline.py
```

## Dependencies

```python
# From requirements.txt
ultralytics>=8.3.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
supervision>=0.20.0  # For ByteTrack integration
```

## Data Classes

```python
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DetectionResult:
    """Result of object detection on a single frame."""
    boxes: np.ndarray              # (N, 4) xyxy format
    masks: Optional[np.ndarray]    # (N, H, W) binary masks or None
    class_ids: np.ndarray          # (N,) class indices
    class_names: list[str]         # (N,) class names
    confidences: np.ndarray        # (N,) confidence scores

    @property
    def num_detections(self) -> int:
        return len(self.boxes)

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """Filter detections by confidence threshold."""
        mask = self.confidences >= threshold
        return DetectionResult(
            boxes=self.boxes[mask],
            masks=self.masks[mask] if self.masks is not None else None,
            class_ids=self.class_ids[mask],
            class_names=[n for n, m in zip(self.class_names, mask) if m],
            confidences=self.confidences[mask],
        )

    def filter_by_classes(self, class_names: list[str]) -> "DetectionResult":
        """Filter detections by class names."""
        mask = np.array([n in class_names for n in self.class_names])
        return DetectionResult(
            boxes=self.boxes[mask],
            masks=self.masks[mask] if self.masks is not None else None,
            class_ids=self.class_ids[mask],
            class_names=[n for n, m in zip(self.class_names, mask) if m],
            confidences=self.confidences[mask],
        )


@dataclass
class FrameDetection:
    """Detection result for a specific frame."""
    frame_idx: int
    timestamp: float              # seconds
    detections: DetectionResult


@dataclass
class TrackedObject:
    """A tracked object across frames."""
    track_id: int                 # Persistent ID across frames
    box: np.ndarray               # (4,) xyxy
    mask: Optional[np.ndarray]    # (H, W) binary mask
    class_id: int
    class_name: str
    confidence: float
    age: int                      # Frames since first detection
    hits: int                     # Total detection count


@dataclass
class TrackingResult:
    """Result of tracking on a single frame."""
    frame_idx: int
    timestamp: float
    tracked_objects: list[TrackedObject]
```

## Class Interface

```python
from typing import Optional, Iterator
import numpy as np
from ultralytics import YOLO


class YOLOPipeline:
    """YOLOv26 detection and segmentation pipeline."""

    # Available model variants
    MODEL_VARIANTS = {
        "yolov26n-seg": "yolov8n-seg.pt",  # Placeholder - use YOLOv8 until v26 available
        "yolov26s-seg": "yolov8s-seg.pt",
        "yolov26m-seg": "yolov8m-seg.pt",
        "yolov26l-seg": "yolov8l-seg.pt",
        "yolov26x-seg": "yolov8x-seg.pt",
    }

    # COCO class names
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    def __init__(
        self,
        model: str = "yolov8n-seg",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda:0",
        half: bool = True,
        target_classes: Optional[list[str]] = None,
    ):
        """
        Initialize YOLO pipeline.

        Args:
            model: Model variant name or path to weights
            confidence: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            device: Inference device ("cuda:0", "cpu", etc.)
            half: Use FP16 inference (GPU only)
            target_classes: Filter to specific classes (None = all 80 COCO classes)
        """
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._device = device
        self._half = half and "cuda" in device
        self._target_classes = target_classes

        # Load model
        model_path = self.MODEL_VARIANTS.get(model, model)
        self._model = YOLO(model_path)

        # Move to device
        self._model.to(device)

        # Get class names from model
        self._class_names = self._model.names

    def detect(
        self,
        frame: np.ndarray,
        return_masks: bool = True,
    ) -> DetectionResult:
        """
        Run detection on single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)
            return_masks: Include segmentation masks

        Returns:
            DetectionResult with boxes, masks, classes, scores
        """
        pass

    def detect_batch(
        self,
        frames: list[np.ndarray],
        return_masks: bool = True,
    ) -> list[DetectionResult]:
        """
        Run detection on batch of frames.

        Args:
            frames: List of BGR images
            return_masks: Include segmentation masks

        Returns:
            List of DetectionResult for each frame
        """
        pass

    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        return_masks: bool = True,
        frame_interval: int = 1,
    ) -> list[FrameDetection]:
        """
        Run detection on video file.

        Args:
            video_path: Input video path
            output_path: Optional output video with annotations
            return_masks: Include segmentation masks
            frame_interval: Process every Nth frame

        Returns:
            List of detections per frame
        """
        pass

    def detect_video_stream(
        self,
        video_path: str,
        return_masks: bool = True,
        frame_interval: int = 1,
    ) -> Iterator[FrameDetection]:
        """
        Stream detection results from video.

        Args:
            video_path: Input video path
            return_masks: Include segmentation masks
            frame_interval: Process every Nth frame

        Yields:
            FrameDetection for each processed frame
        """
        pass

    @property
    def class_names(self) -> list[str]:
        """Get list of detectable class names."""
        return list(self._class_names.values())

    def _filter_results(self, result) -> DetectionResult:
        """Filter and convert YOLO results to DetectionResult."""
        pass

    def _draw_annotations(
        self,
        frame: np.ndarray,
        detections: DetectionResult,
    ) -> np.ndarray:
        """Draw detection annotations on frame."""
        pass
```

## Object Tracker

```python
# ./src/cv_pipeline/tracker.py

from typing import Optional
import numpy as np
import supervision as sv


class ObjectTracker:
    """Multi-object tracker using ByteTrack."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        """
        Initialize tracker.

        Args:
            track_thresh: Detection threshold for tracking
            track_buffer: Frames to keep lost tracks
            match_thresh: IoU threshold for matching
        """
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
        )
        self._frame_idx = 0

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
        tracked_objects = []
        for i in range(len(tracked)):
            tracked_objects.append(TrackedObject(
                track_id=int(tracked.tracker_id[i]),
                box=tracked.xyxy[i],
                mask=tracked.mask[i] if tracked.mask is not None else None,
                class_id=int(tracked.class_id[i]),
                class_name=detections.class_names[i] if i < len(detections.class_names) else "",
                confidence=float(tracked.confidence[i]),
                age=self._frame_idx,  # Simplified - would need proper tracking
                hits=1,  # Simplified
            ))

        return tracked_objects

    def reset(self):
        """Reset tracker state."""
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self._tracker.track_activation_threshold,
            lost_track_buffer=self._tracker.lost_track_buffer,
            minimum_matching_threshold=self._tracker.minimum_matching_threshold,
        )
        self._frame_idx = 0
```

## Implementation Notes

### Single Frame Detection

```python
def detect(
    self,
    frame: np.ndarray,
    return_masks: bool = True,
) -> DetectionResult:
    # Run inference
    results = self._model(
        frame,
        conf=self._confidence,
        iou=self._iou_threshold,
        half=self._half,
        verbose=False,
    )

    return self._filter_results(results[0], return_masks)

def _filter_results(self, result, return_masks: bool = True) -> DetectionResult:
    boxes = result.boxes

    # Get basic detection info
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    class_names = [self._class_names[int(c)] for c in class_ids]

    # Get masks if available and requested
    masks = None
    if return_masks and result.masks is not None:
        masks = result.masks.data.cpu().numpy()

    detection = DetectionResult(
        boxes=xyxy,
        masks=masks,
        class_ids=class_ids,
        class_names=class_names,
        confidences=confidences,
    )

    # Filter by target classes if specified
    if self._target_classes:
        detection = detection.filter_by_classes(self._target_classes)

    return detection
```

### Video Processing

```python
def detect_video(
    self,
    video_path: str,
    output_path: Optional[str] = None,
    return_masks: bool = True,
    frame_interval: int = 1,
) -> list[FrameDetection]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer if output requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            detections = self.detect(frame, return_masks)
            timestamp = frame_idx / fps

            frame_detections.append(FrameDetection(
                frame_idx=frame_idx,
                timestamp=timestamp,
                detections=detections,
            ))

            if writer:
                annotated = self._draw_annotations(frame, detections)
                writer.write(annotated)
        elif writer:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    return frame_detections
```

### Annotation Drawing

```python
def _draw_annotations(
    self,
    frame: np.ndarray,
    detections: DetectionResult,
) -> np.ndarray:
    import cv2

    annotated = frame.copy()

    for i in range(detections.num_detections):
        x1, y1, x2, y2 = detections.boxes[i].astype(int)
        class_name = detections.class_names[i]
        confidence = detections.confidences[i]

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(
            annotated, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        # Draw mask if available
        if detections.masks is not None:
            mask = detections.masks[i]
            colored_mask = np.zeros_like(annotated)
            colored_mask[mask > 0.5] = [0, 255, 0]
            annotated = cv2.addWeighted(annotated, 1, colored_mask, 0.3, 0)

    return annotated
```

## Configuration

```yaml
# config/config.yaml
cv_pipeline:
  enabled: true

  yolo:
    model: "yolov8n-seg"
    confidence: 0.5
    iou_threshold: 0.45
    device: "cuda:0"
    half_precision: true
    target_classes:
      - person
      - car
      - truck
      - bicycle
      - motorcycle
      - forklift  # Custom class if using fine-tuned model

  tracker:
    algorithm: "bytetrack"
    track_thresh: 0.5
    track_buffer: 30
    match_thresh: 0.8
```

## Environment Variables

```bash
YOLO_MODEL=yolov8n-seg
YOLO_CONFIDENCE=0.5
YOLO_IOU_THRESHOLD=0.45
YOLO_DEVICE=cuda:0
ENABLE_CV_PIPELINE=true
ENABLE_TRACKING=true
```

## Testing

```python
# tests/test_yolo_pipeline.py

import pytest
import numpy as np
from poc.src.cv_pipeline.yolo_pipeline import YOLOPipeline, DetectionResult


class TestYOLOPipeline:
    @pytest.fixture
    def pipeline(self):
        return YOLOPipeline(model="yolov8n-seg", device="cpu")

    def test_detect_single_frame(self, pipeline):
        """Test detection on a single frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.detect(frame)
        assert isinstance(result, DetectionResult)

    def test_filter_by_classes(self, pipeline):
        """Test class filtering."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline._target_classes = ["person"]
        result = pipeline.detect(frame)
        for name in result.class_names:
            assert name == "person"

    def test_batch_detection(self, pipeline):
        """Test batch detection."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
        results = pipeline.detect_batch(frames)
        assert len(results) == 4

    def test_video_detection(self, pipeline, tmp_path):
        """Test video detection."""
        # Would need a test video file
        pass
```

## Comparison with Original

| Feature | Original (GSAM) | PoC (YOLOv26) |
|---------|-----------------|---------------|
| Detection | Grounding DINO | YOLOv26 |
| Segmentation | SAM | YOLOv26-seg |
| Framework | TensorRT + DeepStream | Ultralytics |
| Setup | Complex (NGC, TRT build) | Simple (pip install) |
| Classes | Open vocabulary | 80 COCO + custom |
| Speed | ~10-20 FPS | ~30-100+ FPS |
| GPU Memory | High | Low-Medium |

## CV Metadata Fuser Integration

```python
# ./src/cv_pipeline/cv_metadata_fuser.py

class CVMetadataFuser:
    """Fuse CV detection metadata with VLM captions."""

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
        # Count objects per class across all frames
        class_counts = {}
        for fd in detections:
            for name in fd.detections.class_names:
                class_counts[name] = class_counts.get(name, 0) + 1

        # Create metadata summary
        metadata = "\n\nDetected objects:\n"
        for class_name, count in sorted(class_counts.items()):
            metadata += f"- {class_name}: {count} instances\n"

        return captions + metadata
```
