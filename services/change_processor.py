from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class ChangeRegion:
    """
    Spatial description of one connected change region.

    Coordinates are normalized to [0, 1].
    """

    x: float
    y: float
    width: float
    height: float
    area_pixels: int
    area_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "width": round(self.width, 6),
            "height": round(self.height, 6),
            "area_pixels": self.area_pixels,
            "area_percent": round(self.area_percent, 6),
        }


class ChangeProcessor:
    """
    Converts a model probability map into interpretable
    change evidence.

    This class contains no model-specific code.

    Input:
        2-D probability map in [0, 1]

    Output:
        binary mask
        changed area
        connected regions
        region bounding boxes
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_region_pixels: int = 16,
    ) -> None:

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        if min_region_pixels < 1:
            raise ValueError(
                "min_region_pixels must be >= 1."
            )

        self.threshold = threshold
        self.min_region_pixels = min_region_pixels

    def process(
        self,
        probability_map: np.ndarray,
    ) -> Dict[str, Any]:

        probability_map = self._validate_probability_map(
            probability_map
        )

        binary_mask = (
            probability_map >= self.threshold
        )

        total_pixels = binary_mask.size
        changed_pixels = int(binary_mask.sum())

        changed_area_percent = (
            changed_pixels / total_pixels * 100.0
            if total_pixels
            else 0.0
        )

        regions = self._find_regions(
            binary_mask
        )

        return {
            "change_detected": changed_pixels > 0,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "changed_area_percent": round(
                changed_area_percent,
                6,
            ),
            "threshold": self.threshold,
            "change_regions": [
                region.to_dict()
                for region in regions
            ],
            "mask": binary_mask,
        }

    @staticmethod
    def _validate_probability_map(
        probability_map: np.ndarray,
    ) -> np.ndarray:

        array = np.asarray(
            probability_map,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError(
                "probability_map must be a 2-D array."
            )

        if array.size == 0:
            raise ValueError(
                "probability_map cannot be empty."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "probability_map contains NaN or infinite values."
            )

        if array.min() < 0.0 or array.max() > 1.0:
            raise ValueError(
                "probability_map values must be in [0, 1]."
            )

        return array

    def _find_regions(
        self,
        binary_mask: np.ndarray,
    ) -> List[ChangeRegion]:

        height, width = binary_mask.shape

        visited = np.zeros(
            binary_mask.shape,
            dtype=bool,
        )

        regions: List[ChangeRegion] = []

        for y in range(height):
            for x in range(width):

                if not binary_mask[y, x]:
                    continue

                if visited[y, x]:
                    continue

                pixels = self._flood_fill(
                    binary_mask,
                    visited,
                    x,
                    y,
                )

                area = len(pixels)

                if area < self.min_region_pixels:
                    continue

                xs = [point[0] for point in pixels]
                ys = [point[1] for point in pixels]

                min_x = min(xs)
                max_x = max(xs)

                min_y = min(ys)
                max_y = max(ys)

                region_width = (
                    max_x - min_x + 1
                )

                region_height = (
                    max_y - min_y + 1
                )

                regions.append(
                    ChangeRegion(
                        x=min_x / width,
                        y=min_y / height,
                        width=region_width / width,
                        height=region_height / height,
                        area_pixels=area,
                        area_percent=(
                            area
                            / binary_mask.size
                            * 100.0
                        ),
                    )
                )

        regions.sort(
            key=lambda region: region.area_pixels,
            reverse=True,
        )

        return regions

    @staticmethod
    def _flood_fill(
        mask: np.ndarray,
        visited: np.ndarray,
        start_x: int,
        start_y: int,
    ) -> List[Tuple[int, int]]:

        height, width = mask.shape

        stack = [(start_x, start_y)]

        visited[start_y, start_x] = True

        pixels: List[Tuple[int, int]] = []

        while stack:

            x, y = stack.pop()

            pixels.append((x, y))

            neighbors = (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            )

            for nx, ny in neighbors:

                if nx < 0 or nx >= width:
                    continue

                if ny < 0 or ny >= height:
                    continue

                if visited[ny, nx]:
                    continue

                if not mask[ny, nx]:
                    continue

                visited[ny, nx] = True

                stack.append((nx, ny))

        return pixels