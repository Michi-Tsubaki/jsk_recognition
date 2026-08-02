"""Unit tests for the Ultralytics adapter without downloading model weights."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from yolo_3d_ros.inference import YOLOSegmenter


class _TensorLike:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def detach(self) -> _TensorLike:
        return self

    def cpu(self) -> _TensorLike:
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _Boxes:
    def __init__(self, class_ids: np.ndarray, scores: np.ndarray) -> None:
        self.cls = _TensorLike(class_ids)
        self.conf = _TensorLike(scores)

    def __len__(self) -> int:
        return int(self.cls.numpy().size)


class _FakeModel:
    def __init__(self, results: list[object]) -> None:
        self.names = {0: "person"}
        self.results = results
        self.last_source: np.ndarray | None = None
        self.last_kwargs: dict[str, object] | None = None

    def predict(self, source: np.ndarray, **kwargs):
        self.last_source = source
        self.last_kwargs = kwargs
        return self.results


def _install_fake_ultralytics(monkeypatch, model: _FakeModel) -> None:
    fake_module = SimpleNamespace(YOLO=lambda _model_path: model)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)


def test_predict_converts_rgb_to_bgr_and_extracts_instance_arrays(monkeypatch) -> None:
    mask = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    annotated = np.full((2, 2, 3), 7, dtype=np.uint8)
    result = SimpleNamespace(
        boxes=_Boxes(np.array([0.0]), np.array([0.875], dtype=np.float32)),
        masks=SimpleNamespace(data=_TensorLike(mask)),
        names={0: "person"},
        plot=lambda: annotated,
    )
    model = _FakeModel([result])
    _install_fake_ultralytics(monkeypatch, model)

    segmenter = YOLOSegmenter("fixture.pt", 0.25, 0.7, 640, "cpu", [0])
    rgb = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.uint8,
    )
    prediction = segmenter.predict(rgb, annotate=True)

    np.testing.assert_array_equal(model.last_source, rgb[..., ::-1])
    assert model.last_kwargs == {
        "conf": 0.25,
        "iou": 0.7,
        "imgsz": 640,
        "verbose": False,
        "retina_masks": True,
        "device": "cpu",
        "classes": [0],
    }
    np.testing.assert_array_equal(prediction.masks, mask)
    np.testing.assert_array_equal(prediction.class_ids, [0])
    np.testing.assert_allclose(prediction.scores, [0.875])
    np.testing.assert_array_equal(prediction.annotated_bgr, annotated)
    assert prediction.names == {0: "person"}


def test_predict_returns_empty_batch_when_model_returns_no_results(monkeypatch) -> None:
    model = _FakeModel([])
    _install_fake_ultralytics(monkeypatch, model)

    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    prediction = YOLOSegmenter("fixture.pt", 0.2, 0.5, 320, "", None).predict(rgb)

    assert prediction.masks.shape == (0, 3, 4)
    assert prediction.class_ids.shape == (0,)
    assert prediction.scores.shape == (0,)
    assert prediction.annotated_bgr is None


def test_predict_rejects_non_uint8_or_non_hwc_input(monkeypatch) -> None:
    model = _FakeModel([])
    _install_fake_ultralytics(monkeypatch, model)
    segmenter = YOLOSegmenter("fixture.pt", 0.2, 0.5, 320, "", None)

    try:
        segmenter.predict(np.zeros((2, 2, 3), dtype=np.float32))
    except ValueError as exc:
        assert "HWC uint8 RGB" in str(exc)
    else:
        raise AssertionError("Expected ValueError for float input")
