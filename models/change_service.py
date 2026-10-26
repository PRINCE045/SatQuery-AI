from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio

from services.change_result import ChangeResultBuilder


class DCFFChangeService:
    """
    SatQuery AI adapter for the DCFF-Net change-detection specialist.

    If a real DCFF backend is configured, this service is ready to call it.
    When it is not configured, the local path uses a deterministic raster
    difference baseline so the application can still demonstrate a genuine
    pixel-level change workflow without falsely claiming DCFF inference.
    """

    MODEL_NAME = "DCFF-Net"
    BASELINE_MODEL_NAME = "Raster Difference Baseline"
    CHECKPOINT_NAME = "dcff_final_best.pt"

    BENCHMARK = {
        "dataset": "LEVIR-CD",
        "test_samples": 2048,
        "plain_iou": 0.8185,
        "plain_f1": 0.9002,
        "best_iou": 0.8296,
        "best_f1": 0.9069,
        "best_configuration": "MS-TTA 1/1.25/1.5",
    }

    def __init__(self) -> None:
        self.backend = os.getenv(
            "SATQUERY_CHANGE_BACKEND",
            "local",
        )
        self.result_builder = ChangeResultBuilder()

    def build_prediction_result(
        self,
        probability_map: np.ndarray,
        image_t1: str,
        image_t2: str,
        started: float | None = None,
        *,
        model_name: str | None = None,
        backend_name: str | None = None,
        message: str | None = None,
    ) -> Dict[str, Any]:
        if started is None:
            started = time.perf_counter()

        try:
            prediction = self.result_builder.build(
                probability_map
            )
        except Exception as exc:
            return {
                "task": "bi_temporal_change_analysis",
                "status": "error",
                "model": model_name or self.MODEL_NAME,
                "message": f"Failed to process change prediction: {exc}",
                "change_detected": None,
                "changed_area_percent": None,
                "change_regions": [],
                "confidence": 0.0,
                "evidence": [],
                "metrics": {},
                "execution": {
                    "backend": backend_name or self.backend,
                    "runtime_seconds": round(
                        time.perf_counter() - started,
                        4,
                    ),
                },
            }

        runtime = round(
            time.perf_counter() - started,
            4,
        )

        return {
            "task": "bi_temporal_change_analysis",
            "status": "success",
            "model": model_name or self.MODEL_NAME,
            "message": (
                message
                or "Change prediction processed successfully."
            ),
            "change_detected": prediction["change_detected"],
            "changed_area_percent": prediction["changed_area_percent"],
            "change_regions": prediction["change_regions"],
            "confidence": None,
            "evidence": [
                {
                    "type": "input_image",
                    "role": "time_1",
                    "path": str(Path(image_t1).resolve()),
                },
                {
                    "type": "input_image",
                    "role": "time_2",
                    "path": str(Path(image_t2).resolve()),
                },
                {
                    "type": "change_prediction",
                    "description": (
                        "Binary change mask derived from the "
                        "change probability map."
                    ),
                },
            ],
            "metrics": {
                "changed_pixels": prediction["changed_pixels"],
                "total_pixels": prediction["total_pixels"],
                "threshold": prediction["threshold"],
            },
            "execution": {
                "backend": backend_name or self.backend,
                "runtime_seconds": runtime,
            },
        }

    def analyze(
        self,
        image_t1: str,
        image_t2: str,
    ) -> Dict[str, Any]:
        started = time.perf_counter()

        validation_error = self._validate_paths(
            image_t1,
            image_t2,
        )

        if validation_error:
            return self._result(
                status="input_error",
                message=validation_error,
                image_t1=image_t1,
                image_t2=image_t2,
                started=started,
            )

        if self.backend == "local":
            return self._run_local(
                image_t1,
                image_t2,
                started,
            )

        if self.backend == "remote":
            return self._run_remote(
                image_t1,
                image_t2,
                started,
            )

        if self.backend == "unavailable":
            return self._result(
                status="unavailable",
                message=(
                    "DCFF-Net is registered, but no DCFF inference "
                    "backend is configured. Use SATQUERY_CHANGE_BACKEND=local "
                    "for the deterministic raster-difference baseline."
                ),
                image_t1=image_t1,
                image_t2=image_t2,
                started=started,
            )

        return self._result(
            status="error",
            message=f"Unsupported change backend: {self.backend}",
            image_t1=image_t1,
            image_t2=image_t2,
            started=started,
        )

    def _result(
        self,
        *,
        status: str,
        message: str,
        image_t1: str,
        image_t2: str,
        started: float,
    ) -> Dict[str, Any]:
        runtime = round(
            time.perf_counter() - started,
            4,
        )

        return {
            "task": "bi_temporal_change_analysis",
            "status": status,
            "model": self.MODEL_NAME,
            "message": message,
            "change_detected": None,
            "changed_area_percent": None,
            "change_regions": [],
            "confidence": 0.0,
            "evidence": [
                {
                    "type": "input_image",
                    "role": "time_1",
                    "path": str(Path(image_t1).resolve()),
                },
                {
                    "type": "input_image",
                    "role": "time_2",
                    "path": str(Path(image_t2).resolve()),
                },
            ],
            "metrics": {},
            "execution": {
                "backend": self.backend,
                "runtime_seconds": runtime,
            },
        }

    def _validate_paths(
        self,
        image_t1: str,
        image_t2: str,
    ) -> str | None:
        if not image_t1:
            return "Time-1 image path is missing."

        if not image_t2:
            return "Time-2 image path is missing."

        if not Path(image_t1).is_file():
            return f"Time-1 image does not exist: {image_t1}"

        if not Path(image_t2).is_file():
            return f"Time-2 image does not exist: {image_t2}"

        return None

    @staticmethod
    def _read_normalized_gray(
        path: str,
        out_height: int,
        out_width: int,
    ) -> np.ndarray:
        with rasterio.open(path) as dataset:
            data = dataset.read(
                1,
                out_shape=(out_height, out_width),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)

        finite = np.isfinite(data)

        if not finite.any():
            return np.zeros_like(data, dtype=np.float32)

        values = data[finite]
        low, high = np.percentile(values, [2, 98])

        if high <= low:
            low = float(values.min())
            high = float(values.max())

        if high <= low:
            return np.zeros_like(data, dtype=np.float32)

        return np.clip(
            (data - low) / (high - low),
            0.0,
            1.0,
        )

    def _build_raster_difference(
        self,
        image_t1: str,
        image_t2: str,
    ) -> np.ndarray:
        with rasterio.open(image_t1) as t1:
            with rasterio.open(image_t2) as t2:
                height = min(t1.height, t2.height, 1400)
                width = min(t1.width, t2.width, 1400)

        first = self._read_normalized_gray(
            image_t1,
            height,
            width,
        )
        second = self._read_normalized_gray(
            image_t2,
            height,
            width,
        )

        probability = np.abs(second - first)

        max_value = float(probability.max())
        if max_value > 0:
            probability /= max_value

        return probability.astype(np.float32)

    def _run_local(
        self,
        image_t1: str,
        image_t2: str,
        started: float,
    ) -> Dict[str, Any]:
        try:
            probability_map = self._build_raster_difference(
                image_t1,
                image_t2,
            )

            return self.build_prediction_result(
                probability_map=probability_map,
                image_t1=image_t1,
                image_t2=image_t2,
                started=started,
                model_name=self.BASELINE_MODEL_NAME,
                backend_name="local_baseline",
                message=(
                    "Deterministic raster-difference baseline completed. "
                    "This result is not DCFF-Net inference."
                ),
            )

        except Exception as exc:
            return self._result(
                status="error",
                message=f"Raster-difference baseline failed: {exc}",
                image_t1=image_t1,
                image_t2=image_t2,
                started=started,
            )

    def _run_remote(
        self,
        image_t1: str,
        image_t2: str,
        started: float,
    ) -> Dict[str, Any]:
        return self._result(
            status="unavailable",
            message=(
                "Remote DCFF backend is selected, but no inference "
                "endpoint has been configured."
            ),
            image_t1=image_t1,
            image_t2=image_t2,
            started=started,
        )