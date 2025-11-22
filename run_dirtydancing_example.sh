#!/bin/bash
# Example script to run pipeline on dirtydancing.mp4

# Input video
INPUT_VIDEO="sber-swap/examples/videos/dirtydancing.mp4"

# Source face image
SOURCE_FACE="sber-swap/examples/images/mark.jpg"

# Sticker (using bts.png as example - you can use any PNG with transparency)
STICKER="sber-swap/examples/images/bts.png"

# Run pipeline with results directory organization
python pipeline.py \
  -i "$INPUT_VIDEO" \
  -o dirtydancing_result.mp4 \
  --source_paths "$SOURCE_FACE" \
  --stickers "$STICKER:forehead:1.5" \
  --use_results_dir \
  --keep_intermediate

echo ""
echo "Pipeline completed!"
echo "Check results in:"
echo "  - Intermediate: results/intermediate/"
echo "  - Final: results/final/"

