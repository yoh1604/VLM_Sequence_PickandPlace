#!/usr/bin/env bash
set -e

PROJECT_DIR="$HOME/pick_place_occlusion_noetic"
cd "$PROJECT_DIR"

echo "Running D455 VLM + YOLO + FastSAM + Depth pipeline..."

python3 run_d455_pipeline.py