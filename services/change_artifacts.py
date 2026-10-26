from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


class ChangeArtifactWriter:
    """
    Writes visual evidence artifacts produced by change detection.

    The primary artifact is a binary PNG change mask.

    The mask contains:
        0   = no detected change
        255 = detected change
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write_mask(
        self,
        mask: np.ndarray,
        filename: str = "change_mask.png",
    ) -> Dict[str, Any]:

        mask = self._validate_mask(mask)

        output_path = (
            self.output_dir / filename
        )

        image = Image.fromarray(
            mask.astype(np.uint8),
            mode="L",
        )

        image.save(output_path)

        return {
            "type": "change_mask",
            "path": str(output_path.resolve()),
            "format": "PNG",
            "width": int(mask.shape[1]),
            "height": int(mask.shape[0]),
            "changed_pixels": int(
                np.count_nonzero(mask)
            ),
        }

    def write_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filename: str = "change_overlay.png",
    ) -> Dict[str, Any]:

        image = self._validate_image(image)
        mask = self._validate_mask(mask)

        if image.shape[:2] != mask.shape:
            raise ValueError(
                "Image and change mask dimensions "
                "must match."
            )

        base = image.astype(np.float32)

        changed = mask > 0

        # Highlight changed pixels without modifying
        # unchanged pixels.
        overlay = base.copy()

        overlay[changed] = np.array(
            [255, 0, 0],
            dtype=np.float32,
        )

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)

        output_path = (
            self.output_dir / filename
        )

        Image.fromarray(
            overlay,
            mode="RGB",
        ).save(output_path)

        return {
            "type": "change_overlay",
            "path": str(output_path.resolve()),
            "format": "PNG",
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }

    @staticmethod
    def _validate_image(
        image: np.ndarray,
    ) -> np.ndarray:

        array = np.asarray(image)

        if array.ndim != 3:
            raise ValueError(
                "Image must be an HxWx3 RGB array."
            )

        if array.shape[2] != 3:
            raise ValueError(
                "Image must have exactly 3 channels."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "Image contains invalid values."
            )

        if array.min() < 0 or array.max() > 255:
            raise ValueError(
                "Image values must be in [0, 255]."
            )

        return array.astype(np.uint8)

    @staticmethod
    def _validate_mask(
        mask: np.ndarray,
    ) -> np.ndarray:

        array = np.asarray(mask)

        if array.ndim != 2:
            raise ValueError(
                "Change mask must be a 2-D array."
            )

        if array.size == 0:
            raise ValueError(
                "Change mask cannot be empty."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "Change mask contains invalid values."
            )

        unique_values = np.unique(array)

        if not np.all(
            np.isin(unique_values, [0, 1, 255])
        ):
            raise ValueError(
                "Change mask must contain only "
                "0, 1, or 255 values."
            )

        if np.any(array == 1):
            array = (
                array.astype(np.uint8) * 255
            )
        else:
            array = array.astype(np.uint8)

        return array