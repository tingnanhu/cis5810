# Face Swap + Sticker Overlay Pipeline

This pipeline chains two processing steps:
1. **Face Swap** using sber-swap
2. **Sticker Overlay** using mediapipe

## Quick Start

```bash
python pipeline.py \
  -i input_video.mp4 \
  -o output_video.mp4 \
  --source_paths source_face.jpg \
  --stickers "sticker.png:forehead:1.5"
```

## Usage

### Basic Example

```bash
python pipeline.py \
  -i examples/videos/input.mp4 \
  -o final_output.mp4 \
  --source_paths examples/images/mark.jpg \
  --stickers "path/to/sticker.png:forehead:1.5"
```

### Multiple Faces and Stickers

```bash
python pipeline.py \
  -i input_video.mp4 \
  -o output_video.mp4 \
  --source_paths face1.jpg face2.jpg \
  --target_faces_paths target1.jpg target2.jpg \
  --stickers "sticker1.png:forehead:1.5" \
  --stickers "sticker2.png:nose:1.0" \
  --stickers "sticker3.png:left_cheek:0.8"
```

### Keep Intermediate Video

To keep the face-swapped video (before stickers are added):

```bash
python pipeline.py \
  -i input_video.mp4 \
  -o output_video.mp4 \
  --source_paths source_face.jpg \
  --stickers "sticker.png:forehead:1.5" \
  --keep_intermediate \
  --intermediate_output face_swapped_only.mp4
```

### Use Results Directory Structure

To organize outputs in `test/results/` directory (intermediate and final videos):

```bash
python pipeline.py \
  -i input_video.mp4 \
  -o output_video.mp4 \
  --source_paths source_face.jpg \
  --stickers "sticker.png:forehead:1.5" \
  --use_results_dir
```

This will:
- Save intermediate (face-swapped) videos to `test/results/intermediate/`
- Save final videos to `test/results/final/`
- Automatically add timestamps to filenames

## Arguments

### Required Arguments

- `-i, --input`: Input video path
- `-o, --output`: Final output video path (with face swap and stickers)
- `--source_paths`: Path(s) to source face image(s) for face swap (one or more)
- `-s, --stickers`: Sticker configuration(s) in format: `path:position[:scale[:rotation]]`
  - Can specify multiple times with `-s` flag
  - Example: `"sticker.png:forehead:1.5"` or `"sticker.png:nose:1.0:45"`

### Optional Arguments

#### Face Swap (sber-swap) Options

- `--target_faces_paths`: Path(s) to target face image(s) in video (optional)
- `--G_path`: Path to sber-swap generator weights (default: `weights/G_unet_2blocks.pth`)
- `--batch_size`: Batch size for inference (default: 40)
- `--use_sr`: Enable super resolution (requires GPU)
- `--similarity_th`: Similarity threshold for face matching (default: 0.15)

#### Sticker Overlay (mediapipe) Options

- `--min-detection-confidence`: Minimum confidence for face detection (0.0-1.0, default: 0.5)
- `--min-tracking-confidence`: Minimum confidence for face tracking (0.0-1.0, default: 0.5)
- `--codec`: Video codec (default: `mp4v`, use `XVID` or `H264` for better compatibility)

#### Pipeline Options

- `--keep_intermediate`: Keep the intermediate face-swapped video
- `--intermediate_output`: Path for intermediate video (default: auto-generated)
- `--use_results_dir`: Organize outputs in `results/` directory structure
- `--sber_swap_dir`: Path to sber-swap directory (default: `./sber-swap`)
- `--mediapipe_dir`: Path to mediapipe directory (default: `./mediapipe`)

## Sticker Positions

Valid sticker positions:
- `forehead`
- `nose`
- `left_cheek`
- `right_cheek`
- `left_eye`
- `right_eye`
- `chin`
- `mouth`

## Sticker Configuration Format

Format: `path:position[:scale[:rotation]]`

- `path`: Path to sticker image (PNG with transparency recommended)
- `position`: One of the valid positions listed above
- `scale`: Scale factor (optional, default: 1.0)
- `rotation`: Rotation angle in degrees (optional, default: auto-calculated)

Examples:
- `"sticker.png:forehead"` - Basic sticker on forehead
- `"sticker.png:forehead:1.5"` - Sticker on forehead, 1.5x scale
- `"sticker.png:nose:1.0:45"` - Sticker on nose, normal scale, 45° rotation

## How It Works

1. **Face Swap Stage**: 
   - Takes input video and source face image(s)
   - Runs sber-swap inference to create face-swapped video
   - Saves intermediate video (unless `--keep_intermediate` is used)

2. **Sticker Overlay Stage**:
   - Takes the face-swapped video
   - Detects faces using MediaPipe
   - Overlays stickers at specified positions
   - Saves final output video

## Requirements

- Both `sber-swap` and `mediapipe` directories must be present
- All dependencies for both projects must be installed
- See `sber-swap/requirements.txt` and `mediapipe/requirements.txt`

## Results Directory Structure

When using `--use_results_dir`, outputs are organized as:

```
results/
├── intermediate/    # Face-swapped videos (before stickers)
│   └── output_intermediate_20250101_120000.mp4
└── final/          # Final videos (with face swap and stickers)
    └── output_final_20250101_120000.mp4
```

Files are automatically named with timestamps to avoid overwrites.

## Notes

- The intermediate face-swapped video is automatically cleaned up unless `--keep_intermediate` is specified
- Make sure sticker images have transparency (PNG format recommended)
- For best results, use high-quality source face images with clear frontal views
- Processing time depends on video length and resolution
- See `test/README.md` for information about running tests

