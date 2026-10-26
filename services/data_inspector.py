from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio


def inspect_raster(file_path: str) -> Dict[str, Any]:
    """
    Inspect a remote-sensing raster without loading the full dataset
    into memory.

    This is intentionally lightweight so it can run on the
    SIH development laptop.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {file_path}")

    with rasterio.open(path) as src:
        info = {
            "filename": path.name,
            "format": src.driver,
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs) if src.crs else None,
            "transform": str(src.transform),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "resolution": {
                "x": src.res[0],
                "y": src.res[1],
            },
        }

        return info


def read_preview(file_path: str, max_size: int = 1024) -> np.ndarray:
    """
    Read a small preview of a raster.

    We intentionally avoid loading huge imagery at full resolution.
    """

    with rasterio.open(file_path) as src:
        scale = min(
            1.0,
            max_size / max(src.width, src.height),
        )

        width = max(1, int(src.width * scale))
        height = max(1, int(src.height * scale))

        preview = src.read(
            out_shape=(src.count, height, width)
        )

        return preview