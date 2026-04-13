"""
pipeline/runner.py
==================
Orchestrates the full CHMv2 inference pipeline:

  1. Load model
  2. Load & tile input image
  3. Run inference per patch
  4. Mosaic predictions
  5. Save GeoTIFFs + visualisations
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np

from .model import load_model_and_processor
from .tiling import load_rgb_image, extract_patches, mosaic_patches
from .inference import run_patch_inference
from .visualise import (
    per_patch_visual,
    mosaic_visual,
    save_geotiff,
)


class InferencePipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self):
        cfg = self.cfg

        # ── 0. Prepare output directories ──────────────────────────────────
        out_dir = Path(cfg["output"]["output_dir"])
        patches_dir = out_dir / "patches"
        out_dir.mkdir(parents=True, exist_ok=True)
        patches_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Load model ───────────────────────────────────────────────────
        model, processor, device = load_model_and_processor(cfg)

        # ── 2. Load image ───────────────────────────────────────────────────
        rgb, geo_profile = load_rgb_image(cfg)
        original_shape = rgb.shape[:2]          # (H, W)

        # ── 3. Tile image ───────────────────────────────────────────────────
        patch_size = cfg["tiling"]["patch_size"]
        overlap    = cfg["tiling"]["overlap"]
        blend_mode = cfg["tiling"].get("blend_mode", "linear")

        patches, rgb_padded = extract_patches(rgb, patch_size, overlap)
        padded_shape = rgb_padded.shape[:2]

        # ── 4. Run inference ────────────────────────────────────────────────
        predictions, embeddings = run_patch_inference(
            patches, model, processor, device, cfg
        )

        # ── 5. Per-patch visualisations ─────────────────────────────────────
        if cfg["output"].get("save_patch_visuals", True):
            print("[runner] Saving per-patch visualisations…")
            for patch, pred, emb in zip(patches, predictions, embeddings):
                per_patch_visual(patch, pred, emb, patches_dir, cfg)
            print(f"[runner] Patch visuals → {patches_dir}/")

        # ── 6. Mosaic ───────────────────────────────────────────────────────
        print("[runner] Mosaicking patches…")
        mosaic = mosaic_patches(
            patches,
            predictions,
            padded_shape,
            original_shape,
            overlap,
            blend_mode,
        )
        print(
            f"[runner] Mosaic shape: {mosaic.shape}  "
            f"min={mosaic.min():.2f}m  max={mosaic.max():.2f}m"
        )

        # ── 7. Save mosaic GeoTIFF ──────────────────────────────────────────
        if cfg["output"].get("save_mosaic_tif", True):
            mosaic_tif = out_dir / "canopy_height_mosaic.tif"
            save_geotiff(mosaic, geo_profile, mosaic_tif, band_name="canopy_height_m")

        # ── 8. Full-scene visualisation ─────────────────────────────────────
        if cfg["output"].get("save_mosaic_visual", True):
            print("[runner] Generating full-scene visualisation…")
            mosaic_visual(
                rgb_full=rgb,
                mosaic=mosaic,
                patches=patches,
                predictions=predictions,
                embeddings=embeddings,
                out_dir=out_dir,
                cfg=cfg,
            )

        # ── 9. Summary ──────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  Pipeline complete! Outputs:")
        print(f"  GeoTIFF  : {out_dir / 'canopy_height_mosaic.tif'}")
        print(f"  Mosaic   : {out_dir / 'mosaic_visualisation.png'}")
        print(f"  Patches  : {patches_dir}/")
        print(f"{'='*60}\n")
