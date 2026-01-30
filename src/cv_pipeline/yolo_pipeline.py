"""YOLOv26 Pipeline for object detection and segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class DetectionResult:
    """Result of object detection on a single frame."""

    boxes: NDArray[np.float64]  # (N, 4) xyxy format
    masks: Optional[NDArray[np.float64]]  # (N, H, W) binary masks or None
    class_ids: NDArray[np.int64]  # (N,) class indices
    class_names: list[str]  # (N,) class names
    confidences: NDArray[np.float64]  # (N,) confidence scores

    @property
    def num_detections(self) -> int:
        """Return number of detections."""
        return len(self.boxes)

    def filter_by_confidence(self, threshold: float) -> DetectionResult:
        """Filter detections by confidence threshold."""
        mask = self.confidences >= threshold
        return DetectionResult(
            boxes=self.boxes[mask],
            masks=self.masks[mask] if self.masks is not None else None,
            class_ids=self.class_ids[mask],
            class_names=[n for n, m in zip(self.class_names, mask) if m],
            confidences=self.confidences[mask],
        )

    def filter_by_classes(self, target_classes: list[str]) -> DetectionResult:
        """Filter detections by class names."""
        mask = np.array([n in target_classes for n in self.class_names])
        return DetectionResult(
            boxes=self.boxes[mask] if mask.any() else np.array([]).reshape(0, 4),
            masks=self.masks[mask] if self.masks is not None and mask.any() else None,
            class_ids=self.class_ids[mask] if mask.any() else np.array([], dtype=np.int64),
            class_names=[n for n, m in zip(self.class_names, mask) if m],
            confidences=self.confidences[mask] if mask.any() else np.array([]),
        )


@dataclass
class FrameDetection:
    """Detection result for a specific frame."""

    frame_idx: int
    timestamp: float  # seconds
    detections: DetectionResult


@dataclass
class TrackedObject:
    """A tracked object across frames."""

    track_id: int  # Persistent ID across frames
    box: NDArray[np.float64]  # (4,) xyxy
    mask: Optional[NDArray[np.float64]]  # (H, W) binary mask
    class_id: int
    class_name: str
    confidence: float
    age: int  # Frames since first detection
    hits: int  # Total detection count


@dataclass
class TrackingResult:
    """Result of tracking on a single frame."""

    frame_idx: int
    timestamp: float
    tracked_objects: list[TrackedObject]


class YOLOPipeline:
    """YOLOv26 detection and segmentation pipeline."""

    # Available model variants (using YOLOv8 until v26 is available)
    MODEL_VARIANTS: dict[str, str] = {
        "yolov26n-seg": "yolov8n-seg.pt",
        "yolov26s-seg": "yolov8s-seg.pt",
        "yolov26m-seg": "yolov8m-seg.pt",
        "yolov26l-seg": "yolov8l-seg.pt",
        "yolov26x-seg": "yolov8x-seg.pt",
        "yolov8n-seg": "yolov8n-seg.pt",
        "yolov8s-seg": "yolov8s-seg.pt",
        "yolov8m-seg": "yolov8m-seg.pt",
        "yolov8l-seg": "yolov8l-seg.pt",
        "yolov8x-seg": "yolov8x-seg.pt",
    }

    # COCO class names
    COCO_CLASSES: list[str] = [
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
    ) -> None:
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
        # Import here to avoid import errors if ultralytics not installed
        from ultralytics import YOLO

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
        self._class_names: dict[int, str] = self._model.names

    def detect(
        self,
        frame: NDArray[np.uint8],
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
        # Run inference
        results = self._model(
            frame,
            conf=self._confidence,
            iou=self._iou_threshold,
            half=self._half,
            verbose=False,
        )

        return self._filter_results(results[0], return_masks)

    def detect_batch(
        self,
        frames: list[NDArray[np.uint8]],
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
        # Run batch inference
        results = self._model(
            frames,
            conf=self._confidence,
            iou=self._iou_threshold,
            half=self._half,
            verbose=False,
        )

        return [self._filter_results(r, return_masks) for r in results]

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
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Setup video writer if output requested
        writer: Optional[cv2.VideoWriter] = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_detections: list[FrameDetection] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                detections = self.detect(frame, return_masks)
                timestamp = frame_idx / fps if fps > 0 else 0.0

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
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    detections = self.detect(frame, return_masks)
                    timestamp = frame_idx / fps if fps > 0 else 0.0

                    yield FrameDetection(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        detections=detections,
                    )

                frame_idx += 1
        finally:
            cap.release()

    @property
    def class_names(self) -> list[str]:
        """Get list of detectable class names."""
        return list(self._class_names.values())

    def _filter_results(
        self,
        result: "ultralytics.engine.results.Results",  # type: ignore[name-defined]
        return_masks: bool = True,
    ) -> DetectionResult:
        """Filter and convert YOLO results to DetectionResult."""
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return DetectionResult(
                boxes=np.array([]).reshape(0, 4),
                masks=None,
                class_ids=np.array([], dtype=np.int64),
                class_names=[],
                confidences=np.array([]),
            )

        # Get basic detection info
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(np.int64)
        class_names_list = [self._class_names[int(c)] for c in class_ids]

        # Get masks if available and requested
        masks: Optional[NDArray[np.float64]] = None
        if return_masks and result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        detection = DetectionResult(
            boxes=xyxy,
            masks=masks,
            class_ids=class_ids,
            class_names=class_names_list,
            confidences=confidences,
        )

        # Filter by target classes if specified
        if self._target_classes:
            detection = detection.filter_by_classes(self._target_classes)

        return detection

    def _draw_annotations(
        self,
        frame: NDArray[np.uint8],
        detections: DetectionResult,
    ) -> NDArray[np.uint8]:
        """Draw detection annotations on frame."""
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
            if detections.masks is not None and i < len(detections.masks):
                mask = detections.masks[i]
                # Resize mask to frame size if needed
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (frame.shape[1], frame.shape[0])
                    )
                colored_mask = np.zeros_like(annotated)
                colored_mask[mask > 0.5] = [0, 255, 0]
                annotated = cv2.addWeighted(annotated, 1, colored_mask, 0.3, 0)

        return annotated
