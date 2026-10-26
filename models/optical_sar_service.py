from pathlib import Path
from typing import Any, Dict
import time


class OpticalSARService:
    """
    SatQuery adapter for the MM-OVSeg optical-SAR specialist.

    The heavy MM-OVSeg model is not executed on the local
    development machine. This service provides the stable
    interface used by the SatQuery agent.

    A cloud inference backend can be connected without
    changing the agent/orchestrator layer.
    """

    MODEL_NAME = "MM-OVSeg B/16"

    def __init__(self) -> None:
        import os

        self.backend = os.getenv(
            "SATQUERY_OPTICAL_SAR_BACKEND",
            "unavailable",
        )

    def analyze(
        self,
        optical_image: str,
        sar_image: str,
    ) -> Dict[str, Any]:

        started = time.perf_counter()

        optical = Path(optical_image)
        sar = Path(sar_image)

        if not optical.is_file():
            return self._error(
                "Optical image does not exist.",
                optical_image,
                sar_image,
                started,
            )

        if not sar.is_file():
            return self._error(
                "SAR image does not exist.",
                optical_image,
                sar_image,
                started,
            )

        if self.backend == "unavailable":
            return {
                "task": "optical_sar_analysis",
                "status": "unavailable",
                "model": self.MODEL_NAME,
                "message": (
                    "MM-OVSeg B/16 is registered, but its "
                    "GPU inference backend is not configured."
                ),
                "confidence": 0.0,
                "evidence": [
                    {
                        "type": "input_image",
                        "role": "optical",
                        "path": str(optical.resolve()),
                    },
                    {
                        "type": "input_image",
                        "role": "sar",
                        "path": str(sar.resolve()),
                    },
                ],
                "execution": {
                    "backend": self.backend,
                    "runtime_seconds": round(
                        time.perf_counter() - started,
                        4,
                    ),
                },
            }

        return {
            "task": "optical_sar_analysis",
            "status": "error",
            "model": self.MODEL_NAME,
            "message": (
                f"Unsupported MM-OVSeg backend: {self.backend}"
            ),
            "confidence": 0.0,
            "evidence": [],
            "execution": {
                "backend": self.backend,
                "runtime_seconds": round(
                    time.perf_counter() - started,
                    4,
                ),
            },
        }

    @staticmethod
    def _error(
        message: str,
        optical_image: str,
        sar_image: str,
        started: float,
    ) -> Dict[str, Any]:

        return {
            "task": "optical_sar_analysis",
            "status": "input_error",
            "model": OpticalSARService.MODEL_NAME,
            "message": message,
            "confidence": 0.0,
            "evidence": [
                {
                    "type": "input_image",
                    "role": "optical",
                    "path": str(
                        Path(optical_image).resolve()
                    ),
                },
                {
                    "type": "input_image",
                    "role": "sar",
                    "path": str(
                        Path(sar_image).resolve()
                    ),
                },
            ],
            "execution": {
                "backend": "validation",
                "runtime_seconds": round(
                    time.perf_counter() - started,
                    4,
                ),
            },
        }