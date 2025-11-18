# Testing Sticker Overlay on Images

## Quick Start

The test script works with **system Python 3.10** (which has MediaPipe 0.10.21). The conda environment's MediaPipe 0.10.11 has a known bug.

### Basic Usage

```bash
cd mediapipe
python3 test_sticker_image.py \
    -i ../ghost/examples/images/beckham.jpg \
    -o output.jpg \
    -s path/to/sticker.png \
    -p forehead \
    --scale 1.0
```

### Command Line Options

- `-i, --input`: Input image path (required)
- `-o, --output`: Output image path (required)
- `-s, --sticker`: Path to sticker image (required)
- `-p, --position`: Where to place sticker
  - Options: `forehead`, `nose`, `left_cheek`, `right_cheek`, `left_eye`, `right_eye`, `chin`, `mouth`
  - Default: `forehead`
- `--scale`: Scale factor for sticker size (default: 1.0)
  - `0.5` = smaller, `2.0` = larger
- `--min-detection-confidence`: Face detection confidence (0.0-1.0, default: 0.5)
- `--min-tracking-confidence`: Face tracking confidence (0.0-1.0, default: 0.5)

### Examples

**Test with forehead placement:**
```bash
python3 test_sticker_image.py \
    -i ../ghost/examples/images/beckham.jpg \
    -o test_forehead.jpg \
    -s ../ghost/examples/images/example1.png \
    -p forehead \
    --scale 1.0
```

**Test with nose placement:**
```bash
python3 test_sticker_image.py \
    -i ../ghost/examples/images/beckham.jpg \
    -o test_nose.jpg \
    -s ../ghost/examples/images/example1.png \
    -p nose \
    --scale 0.8
```

**Test with different scale:**
```bash
python3 test_sticker_image.py \
    -i ../ghost/examples/images/beckham.jpg \
    -o test_large.jpg \
    -s ../ghost/examples/images/example1.png \
    -p forehead \
    --scale 1.5
```

## Using System Python vs Conda

**System Python 3.10** (recommended):
```bash
python3 test_sticker_image.py -i input.jpg -o output.jpg -s sticker.png
```

**Conda environment** (currently broken):
```bash
conda activate ghost_env
python test_sticker_image.py -i input.jpg -o output.jpg -s sticker.png
# This will fail due to MediaPipe 0.10.11 bug
```

## Tips

1. **Sticker format**: Use PNG images with transparency (alpha channel) for best results
2. **Sticker size**: Original sticker size doesn't matter; it will be automatically resized
3. **Multiple faces**: The script supports multiple faces in an image
4. **Position**: Try different positions (`forehead`, `nose`, etc.) to see what works best

## Troubleshooting

- **No faces detected**: Lower `--min-detection-confidence` (e.g., `0.3`)
- **Sticker too large/small**: Adjust `--scale` parameter
- **MediaPipe errors**: Use system Python 3.10 instead of conda environment

