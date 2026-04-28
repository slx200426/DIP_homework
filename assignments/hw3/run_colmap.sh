#!/bin/bash
set -e

DATASET_PATH="data"
IMAGE_PATH="$DATASET_PATH/images"
COLMAP_PATH="$DATASET_PATH/colmap"

mkdir -p "$COLMAP_PATH/sparse"

echo "=== Step 1: Feature Extraction ==="
colmap feature_extractor \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

echo "=== Step 2: Feature Matching ==="
colmap exhaustive_matcher \
    --database_path "$COLMAP_PATH/database.db"

echo "=== Step 3: Sparse Reconstruction ==="
colmap mapper \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --output_path "$COLMAP_PATH/sparse"

echo "=== Done! ==="
echo "Sparse reconstruction saved to: $COLMAP_PATH/sparse/0/"
echo "You can visualize it with: colmap gui"