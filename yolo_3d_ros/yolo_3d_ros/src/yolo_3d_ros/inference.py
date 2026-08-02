"""Thin, lazy-loaded wrapper around Ultralytics YOLO segmentation inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class PredictionBatch:
    masks: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray
    names: Any
    annotated_bgr: np.ndarray | None


class YOLOSegmenter:
    """Run an Ultralytics segmentation model on RGB images."""

    def __init__(
        self,
        model: str,
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
        classes: list[int] | None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed in the yolo_3d_ros Python environment. "
                "Run `colcon build --packages-up-to yolo_3d_ros --symlink-install` "
                "so CMake can prepare the package .venv with uv."
            ) from exc

        self._model = YOLO(model)
        self._confidence = float(confidence)
        self._iou = float(iou)
        self._image_size = int(image_size)
        self._device = str(device).strip()
        self._classes = classes

    def predict(self, rgb: np.ndarray, annotate: bool = False) -> PredictionBatch:
        """Run segmentation on an HWC uint8 RGB array.

        Ultralytics accepts NumPy/OpenCV images in BGR order, so the camera RGB array
        is explicitly converted before inference.
        """

        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(
                "Expected HWC uint8 RGB input, "
                f"got shape={rgb.shape}, dtype={rgb.dtype}"
            )
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        kwargs: dict[str, Any] = {
            "conf": self._confidence,
            "iou": self._iou,
            "imgsz": self._image_size,
            "verbose": False,
            "retina_masks": True,
        }
        if self._device:
            kwargs["device"] = self._device
        if self._classes:
            kwargs["classes"] = self._classes

        results = self._model.predict(source=bgr, **kwargs)
        if not results:
            return PredictionBatch(
                masks=np.empty((0, rgb.shape[0], rgb.shape[1]), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int64),
                scores=np.empty((0,), dtype=np.float32),
                names=getattr(self._model, "names", {}),
                annotated_bgr=None,
            )

        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks_obj = getattr(result, "masks", None)
        if boxes is None or masks_obj is None or len(boxes) == 0:
            return PredictionBatch(
                masks=np.empty((0, rgb.shape[0], rgb.shape[1]), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int64),
                scores=np.empty((0,), dtype=np.float32),
                names=getattr(result, "names", getattr(self._model, "names", {})),
                annotated_bgr=result.plot() if annotate else None,
            )

        mask_data = masks_obj.data.detach().cpu().numpy().astype(np.float32, copy=False)
        class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
        return PredictionBatch(
            masks=mask_data,
            class_ids=class_ids,
            scores=scores,
            names=getattr(result, "names", getattr(self._model, "names", {})),
            annotated_bgr=result.plot() if annotate else None,
        )
