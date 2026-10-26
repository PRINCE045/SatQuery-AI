from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


output_path = Path("data/sample/sample.tif")
output_path.parent.mkdir(parents=True, exist_ok=True)

width = 256
height = 256
bands = 3

data = np.random.randint(
    0,
    255,
    size=(bands, height, width),
    dtype=np.uint8,
)

transform = from_origin(
    77.0,
    29.0,
    0.0001,
    0.0001,
)

with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=bands,
    dtype=data.dtype,
    crs="EPSG:4326",
    transform=transform,
) as dst:
    dst.write(data)

print(f"Created: {output_path}")