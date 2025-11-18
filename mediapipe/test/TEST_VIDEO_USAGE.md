# Testing Sticker Overlay on Videos

## Quick Start

The test script works with **system Python 3.10** (which has MediaPipe 0.10.21). The conda environment's MediaPipe 0.10.11 has a known bug.

### Basic Usage

```bash
cd mediapipe
python3 test/test_sticker_video.py \
    -i ../ghost/examples/videos/dance.mp4 \
    -o test_output.mp4 \
    -s path/to/sticker.png \
    -p forehead
```

### Command Line Options

#### Required Arguments

- `-i, --input`: Input video path
- `-o, --output`: Output video path
- `-s, --stickers`: Sticker configuration (can specify multiple times)
  - Format: `path[:position[:scale[:rotation]]]`
  - Examples:
    - `sticker.png` (uses default position and scale)
    - `sticker.png:forehead` (specify position)
    - `sticker.png:forehead:1.5` (specify position and scale)
    - `sticker.png:forehead:1.5:45` (specify position, scale, and rotation)

#### Optional Arguments

- `-p, --position`: Default position if not specified in sticker config
  - Options: `forehead`, `nose`, `left_cheek`, `right_cheek`, `left_eye`, `right_eye`, `chin`, `mouth`
  - Default: `forehead`
- `--scale`: Default scale factor if not specified in sticker config (default: 1.0)
  - `0.5` = smaller, `2.0` = larger
- `--max-frames`: Maximum number of frames to process (for quick testing)
  - Example: `--max-frames 100` (process only first 100 frames)
- `--min-detection-confidence`: Face detection confidence (0.0-1.0, default: 0.5)
- `--min-tracking-confidence`: Face tracking confidence (0.0-1.0, default: 0.5)
- `--codec`: Video codec (default: `mp4v`)
  - Options: `mp4v`, `XVID`, `H264` (for better compatibility)

### Examples

**Basic single sticker:**
```bash
python3 test/test_sticker_video.py \
    -i ../ghost/examples/videos/dance.mp4 \
    -o test_output.mp4 \
    -s ../ghost/examples/images/example1.png \
    -p forehead
```

**Multiple stickers:**
```bash
python3 test/test_sticker_video.py \
    -i ../ghost/examples/videos/dance.mp4 \
    -o test_output.mp4 \
    -s sticker1.png:forehead:1.5 \
    -s sticker2.png:nose:1.0 \
    -s sticker3.png:left_cheek:0.8
```

**Test with limited frames (quick test):**
```bash
python3 test/test_sticker_video.py \
    -i ../ghost/examples/videos/dance.mp4 \
    -o test_output.mp4 \
    -s sticker.png:forehead \
    --max-frames 100
```

**Different positions:**
```bash
# Forehead
python3 test/test_sticker_video.py \
    -i input.mp4 \
    -o output_forehead.mp4 \
    -s sticker.png:forehead

# Nose
python3 test/test_sticker_video.py \
    -i input.mp4 \
    -o output_nose.mp4 \
    -s sticker.png:nose:0.8

# Cheeks
python3 test/test_sticker_video.py \
    -i input.mp4 \
    -o output_cheeks.mp4 \
    -s sticker1.png:left_cheek:1.0 \
    -s sticker2.png:right_cheek:1.0
```

**With rotation:**
```bash
python3 test/test_sticker_video.py \
    -i input.mp4 \
    -o output_rotated.mp4 \
    -s sticker.png:forehead:1.0:45
```

## Using System Python vs Conda

**System Python 3.10** (recommended):
```bash
python3 test/test_sticker_video.py -i input.mp4 -o output.mp4 -s sticker.png
```

**Conda environment** (currently broken):
```bash
conda activate ghost_env
python test/test_sticker_video.py -i input.mp4 -o output.mp4 -s sticker.png
# This will fail due to MediaPipe 0.10.11 bug
```

## Tips

1. **Sticker format**: Use PNG images with transparency (alpha channel) for best results
2. **Sticker size**: Original sticker size doesn't matter; it will be automatically resized based on face size
3. **Multiple faces**: The script supports multiple faces in a video
4. **Performance**: For long videos, use `--max-frames` to test on a subset first
5. **Video codec**: If the output video doesn't play, try `--codec XVID` or `--codec H264`

## Troubleshooting

- **No faces detected**: Lower `--min-detection-confidence` (e.g., `0.3`)
- **Sticker too large/small**: Adjust scale in sticker config (e.g., `:0.5` for smaller, `:2.0` for larger)
- **MediaPipe errors**: Use system Python 3.10 instead of conda environment
- **Video won't play**: Try different codec with `--codec XVID` or `--codec H264`
- **Processing too slow**: Use `--max-frames` to test on a subset first, or resize the video before processing

## Performance Notes

- Video processing can be slow depending on video length and resolution
- MediaPipe uses CPU, so ensure sufficient system resources
- For testing, use `--max-frames 100` to process only the first 100 frames
- Consider resizing videos before processing if they're very high resolution

