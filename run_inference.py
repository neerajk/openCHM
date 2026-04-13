"""
CHMv2 Canopy Height Inference Pipeline
======================================
DINOv3 + DPT head for per-pixel canopy height estimation (metres).

Usage:
    python run_inference.py --config config.yaml
"""

import argparse
import yaml
from pathlib import Path
from pipeline.runner import InferencePipeline


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="CHMv2 Canopy Height Inference — DINOv3 backbone"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"\n{'='*60}")
    print("  CHMv2 Canopy Height Pipeline  (DINOv3 + DPT Head)")
    print(f"{'='*60}")
    print(f"  Input  : {cfg['input']['image_path']}")
    print(f"  Output : {cfg['output']['output_dir']}")
    print(f"  Device : {cfg['model']['device']}")
    print(f"{'='*60}\n")

    pipeline = InferencePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
