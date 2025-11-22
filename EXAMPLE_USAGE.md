# Example: Running Pipeline on dirtydancing.mp4

This document shows how to run the pipeline on the `dirtydancing.mp4` video.

## Prerequisites

Make sure you have all dependencies installed:

```bash
# Install sber-swap dependencies
cd sber-swap
pip install -r requirements.txt
cd ..

# Install mediapipe dependencies
cd mediapipe
pip install -r requirements.txt
cd ..
```

## Quick Run

Use the provided script:

```bash
./run_dirtydancing_example.sh
```

## Manual Run

Or run the pipeline manually:

```bash
python pipeline.py \
  -i sber-swap/examples/videos/dirtydancing.mp4 \
  -o dirtydancing_result.mp4 \
  --source_paths sber-swap/examples/images/mark.jpg \
  --stickers "sber-swap/examples/images/bts.png:forehead:1.5" \
  --use_results_dir \
  --keep_intermediate
```

## Available Source Faces

You can use any of these source face images:
- `sber-swap/examples/images/mark.jpg`
- `sber-swap/examples/images/elon_musk.jpg`
- `sber-swap/examples/images/beckham.jpg`
- `sber-swap/examples/images/murakami.jpg`

## Available Stickers

You can use any PNG image as a sticker. Examples:
- `sber-swap/examples/images/bts.png`
- `sber-swap/examples/images/example1.png`
- `sber-swap/examples/images/example2.png`

## Output Locations

With `--use_results_dir`:
- **Intermediate video** (face-swapped only): `results/intermediate/dirtydancing_result_intermediate_TIMESTAMP.mp4`
- **Final video** (with stickers): `results/final/dirtydancing_result_final_TIMESTAMP.mp4`

## Multiple Stickers Example

To add multiple stickers:

```bash
python pipeline.py \
  -i sber-swap/examples/videos/dirtydancing.mp4 \
  -o dirtydancing_result.mp4 \
  --source_paths sber-swap/examples/images/mark.jpg \
  --stickers "sber-swap/examples/images/bts.png:forehead:1.5" \
  --stickers "sber-swap/examples/images/example1.png:nose:1.0" \
  --use_results_dir
```

## Sticker Positions

Valid positions:
- `forehead`
- `nose`
- `left_cheek`
- `right_cheek`
- `left_eye`
- `right_eye`
- `chin`
- `mouth`

## Notes

- Processing time depends on video length and resolution
- The intermediate face-swapped video is kept when using `--keep_intermediate`
- Make sure sticker images have transparency (PNG format recommended)

