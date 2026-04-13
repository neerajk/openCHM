# CHMv2 Canopy Height Inference Pipeline

**DINOv3 + DPT depth estimation head — per-pixel canopy height from RGB satellite imagery**

Based on:
- Paper: [CHMv2: Improvements in Global Canopy Height Mapping using DINOv3](https://arxiv.org/abs/2603.06382)
- Backbone repo: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- Model weights: [facebook/dinov3-vitl16-chmv2-dpt-head on HuggingFace](https://huggingface.co/facebook/dinov3-vitl16-chmv2-dpt-head)

---

## What does this pipeline do?

| Step | Description |
|------|-------------|
| Load image | Reads a 3-band RGB GeoTIFF (Sentinel-2 or any optical imagery) |
| Tile | Splits the scene into overlapping patches (default 512×512 px) |
| Infer | Runs CHMv2 (DINOv3 ViT-L/16 + DPT head) on every patch |
| Mosaic | Blends patches back into a full-scene canopy height map |
| Visualise | Saves per-patch RGB/heatmap panels + DINOv3 embedding PCA maps + full mosaic |
| Export | Saves georeferenced GeoTIFF of canopy heights in **metres** |

### Model inputs & outputs

| | Detail |
|--|--------|
| **Input** | 3-band RGB image (uint8 or uint16 GeoTIFF). The processor auto-normalises. |
| **Output (primary)** | Float32 raster: per-pixel **canopy height in metres** (range 0–96 m) |
| **Output (auxiliary)** | DINOv3 backbone feature embeddings → visualised as PCA heatmap |

> **Note on Sentinel-2:** The model was trained on high-resolution sub-metre imagery (Maxar/Planet). With Sentinel-2 (10 m GSD) the predictions are still meaningful but spatially coarser. For best results use 0.5–2 m GSD imagery.

---

## Getting model weights

### Option A — HuggingFace Hub (recommended, automatic)
The weights download automatically on first run from HuggingFace:
```
facebook/dinov3-vitl16-chmv2-dpt-head
```
No account or request form needed. ~1.2 GB download on first run.

### Option B — Meta AI direct download
1. Request access at: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
2. You will receive an email with a download URL.
3. Use `wget` (not a browser) to download.
4. Point to local weights via `torch.hub.load(..., weights="/path/to/weights.pth")`

---

## Installation (Mac M1, conda)

```bash
# 1. Clone the dinov3 repo (optional, for reference)
git clone https://github.com/facebookresearch/dinov3.git

# 2. Create and activate conda environment
conda env create -f environment.yml
conda activate chmv2

# 3. Verify PyTorch sees MPS (Apple Metal GPU)
python -c "import torch; print(torch.backends.mps.is_available())"
```

> **Tip for M1:** Set `device: "mps"` in `config.yaml` to use the Apple GPU for faster inference. If you hit MPS errors, fall back to `"cpu"` — 16 GB RAM handles 512×512 patches easily on CPU.

---

## Project structure

```
chmv2_pipeline/
├── run_inference.py          ← Entry point
├── config.yaml               ← All parameters (edit this)
├── environment.yml           ← Conda environment
├── pipeline/
│   ├── __init__.py
│   ├── runner.py             ← Orchestrates all steps
│   ├── model.py              ← Load CHMv2 from HuggingFace
│   ├── tiling.py             ← Patch extraction & mosaicking
│   ├── inference.py          ← Run model per patch
│   └── visualise.py          ← All heatmap & mosaic visuals
├── scripts/
│   └── create_test_image.py  ← Generate synthetic Sentinel-2 for testing
└── data/
    ├── input/
    │   └── sentinel2_rgb.tif  ← Put your image here
    └── output/               ← All outputs written here
        ├── canopy_height_mosaic.tif
        ├── mosaic_visualisation.png
        ├── mosaic_canopy_height.png
        ├── mosaic_embeddings_pca.png
        └── patches/
            ├── patch_0000.png
            ├── patch_0001.png
            └── …
```

---

## Quick start commands

```bash
# Activate environment
conda activate chmv2

# (Optional) Create a synthetic test image if you don't have real data yet
python scripts/create_test_image.py

# Edit config.yaml to point to your image, then run:
python run_inference.py --config config.yaml
```

### Using your own Sentinel-2 image

1. Download a Sentinel-2 L2A scene from [Copernicus Data Space](https://dataspace.copernicus.eu/) or [Earth Engine](https://earthengine.google.com/)
2. Export bands B4 (Red), B3 (Green), B2 (Blue) as a 3-band GeoTIFF
3. Update `config.yaml`:
   ```yaml
   input:
     image_path: "data/input/your_sentinel2_scene.tif"
     band_order: [1, 2, 3]   # Already in R,G,B order
   ```
4. Run: `python run_inference.py --config config.yaml`

---

## config.yaml parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input.image_path` | `data/input/sentinel2_rgb.tif` | Path to your RGB GeoTIFF |
| `input.band_order` | `[1, 2, 3]` | Rasterio band indices (1-based) for R, G, B |
| `model.hf_model_id` | `facebook/dinov3-vitl16-chmv2-dpt-head` | HuggingFace model ID |
| `model.device` | `cpu` | `cpu`, `mps` (Apple GPU), or `cuda` |
| `tiling.patch_size` | `512` | Patch size in pixels |
| `tiling.overlap` | `64` | Overlap between patches (reduces edge artefacts) |
| `tiling.blend_mode` | `linear` | `linear` (feathered) or `hard` |
| `output.colormap` | `viridis` | Matplotlib colormap for height heatmap |
| `output.embedding_colormap` | `turbo` | Colormap for embedding PCA maps |

---

## Expected outputs

| File | Description |
|------|-------------|
| `canopy_height_mosaic.tif` | Georeferenced float32 GeoTIFF, canopy height in metres |
| `mosaic_visualisation.png` | 3-panel: RGB · height heatmap · DINOv3 embeddings PCA |
| `mosaic_canopy_height.png` | Height heatmap only (with colourbar) |
| `mosaic_embeddings_pca.png` | Full-scene DINOv3 embedding PCA (paper-style figure) |
| `patches/patch_NNNN.png` | Per-patch 3-panel: RGB · height · embedding |

---

## Useful links

| Resource | URL |
|----------|-----|
| CHMv2 paper | https://arxiv.org/abs/2603.06382 |
| DINOv3 paper | https://arxiv.org/abs/2508.10104 |
| DINOv3 GitHub | https://github.com/facebookresearch/dinov3 |
| HuggingFace weights | https://huggingface.co/facebook/dinov3-vitl16-chmv2-dpt-head |
| HuggingFace model docs | https://huggingface.co/docs/transformers/main/en/model_doc/chmv2 |
| Meta DINOv3 weight request | https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ |
| Copernicus Sentinel-2 data | https://dataspace.copernicus.eu/ |
| CHMv2 inference notebook | https://github.com/facebookresearch/dinov3/tree/main/notebooks |

---

## Troubleshooting

**`ImportError: cannot import name 'CHMv2ForDepthEstimation'`**
→ Upgrade transformers: `pip install --upgrade transformers`  (needs ≥ 4.56.0)

**Out of memory on M1**
→ Reduce `tiling.patch_size` in config.yaml to `256`

**MPS errors**
→ Set `model.device: "cpu"` in config.yaml

**Blank/zero canopy height output**
→ Your image might have no vegetation or very low GSD. Try a forest-dense area.
