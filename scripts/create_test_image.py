"""
scripts/create_test_image.py
============================
Creates a synthetic 3-band (RGB) GeoTIFF for pipeline testing.
Run this if you don't yet have a real Sentinel-2 image.

Usage:
    python scripts/create_test_image.py
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS


def create_synthetic_sentinel2(
    output_path: str = "data/input/sentinel2_rgb.tif",
    width: int = 1024,
    height: int = 1024,
):
    """
    Creates a plausible-looking 3-band synthetic GeoTIFF (uint16, Sentinel-2-like).
    Adds spatial patterns mimicking forest/non-forest regions.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Background: mixed reflectance
    red   = rng.integers(300,  900, (height, width), dtype=np.uint16)
    green = rng.integers(500, 1400, (height, width), dtype=np.uint16)
    blue  = rng.integers(200,  800, (height, width), dtype=np.uint16)

    # Add forest patches (dark, green-dominated blobs)
    from scipy.ndimage import gaussian_filter
    forest_mask = rng.random((height, width)) > 0.6
    forest_mask = gaussian_filter(forest_mask.astype(float), sigma=30) > 0.45

    red[forest_mask]   = rng.integers(200, 500, forest_mask.sum(), dtype=np.uint16)
    green[forest_mask] = rng.integers(700, 1800, forest_mask.sum(), dtype=np.uint16)
    blue[forest_mask]  = rng.integers(150, 400, forest_mask.sum(), dtype=np.uint16)

    # Bounding box: somewhere over the Amazon (for demo)
    lon_min, lat_min = -60.5, -3.5
    lon_max, lat_max = -60.0, -3.0
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    crs = CRS.from_epsg(4326)

    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": width,
        "height": height,
        "count": 3,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(red,   1)
        dst.write(green, 2)
        dst.write(blue,  3)
        dst.update_tags(
            description="Synthetic Sentinel-2 RGB — test image",
            bands="R=1, G=2, B=3",
        )

    print(f"[create_test_image] Saved → {output_path}  ({width}×{height} px, uint16)")


if __name__ == "__main__":
    create_synthetic_sentinel2()
