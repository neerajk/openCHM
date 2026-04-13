"""
scripts/fetch_test_image.py
============================
Fetches a real 3-band (RGB) Sentinel-2 GeoTIFF from Microsoft Planetary Computer 
using odc-stac, replacing the synthetic data generator.

Usage:
    python scripts/fetch_test_image.py
"""

from pathlib import Path
import planetary_computer as pc
from pystac_client import Client
import odc.stac
import rioxarray  # Required for exporting xarray to GeoTIFF

def fetch_sentinel2_rgb(
    output_path: str = "data/input/sentinel2_rgb.tif",
    bbox: list = [77.95, 30.25, 78.15, 30.50], # Example: Seattle/Cascades area
    date_range: str = "2025-10-01/2025-11-30", # Summer months for visible canopy
    max_cloud_cover: int = 5
):
    """
    Connects to Planetary Computer STAC, finds the least cloudy Sentinel-2 Level-2A 
    image for the given bbox, downloads the RGB bands, and saves it as a GeoTIFF.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("🌍 Connecting to Microsoft Planetary Computer STAC API...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    print(f"🔍 Searching for scenes in bbox {bbox} between {date_range}...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )
    
    # Get all items and sort them by cloud cover so we get the clearest image
    items = list(search.items())
    if not items:
        print("❌ No images found. Try expanding your date range or cloud cover tolerance.")
        return
        
    items.sort(key=lambda x: x.properties["eo:cloud_cover"])
    best_item = items[0]
    
    print(f"📥 Found {len(items)} scenes. Fetching the clearest one: {best_item.id}")
    print(f"☁️  Cloud Cover: {best_item.properties['eo:cloud_cover']}%")

    # Load data using odc.stac
    # We only request RGB bands (B04=Red, B03=Green, B02=Blue)
    # Resolution is natively 10 meters for these bands.
    print("⏳ Downloading and assembling chunks into memory...")
    data = odc.stac.load(
        [best_item],
        bands=["B04", "B03", "B02"],
        bbox=bbox,
        resolution=10,
        chunks={"x": 2048, "y": 2048} # Dask chunking protects your Mac's RAM
    )
    
    # odc.stac returns data with a 'time' dimension (since we passed a list of items).
    # Since we only passed one item, we squeeze the time dimension out.
    data = data.squeeze("time")
    
    # Convert the Xarray Dataset (which has variables 'b04', 'b03', 'b02') 
    # into a single 3D DataArray with a 'band' dimension.
    print("🔄 Formatting data array for GeoTIFF export...")
    img_array = data.to_array(dim="band")
    
    # Write directly to GeoTIFF using rioxarray
    print(f"💾 Saving to {output_path}...")
    
    # rioxarray reads the Spatial metadata (CRS, transform) embedded by odc.stac
    # and writes it perfectly to the .tif file.
    img_array.rio.to_raster(
        output_path,
        compress="lzw",
        driver="GTiff"
    )
    
    print(f"✅ Success! Saved real Sentinel-2 uint16 data to {output_path}")

if __name__ == "__main__":
    fetch_sentinel2_rgb()