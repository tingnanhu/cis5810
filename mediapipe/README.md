# MediaPipe Sticker Overlay for Videos

This module provides functionality to add stickers to faces in videos using MediaPipe Face Landmarker/Face Mesh.

## Features

- **Face Detection**: Uses MediaPipe Face Mesh for accurate face landmark detection
- **Multiple Sticker Positions**: Support for forehead, nose, cheeks, eyes, chin, and mouth
- **Customizable**: Adjustable scale, rotation, and position
- **Alpha Channel Support**: Handles transparent PNG stickers
- **Real-time Processing**: Efficient video frame processing

## Installation

Make sure you have the required dependencies:

```bash
pip install mediapipe opencv-python numpy tqdm
```

## Usage

### Command Line Interface

Basic usage:

```bash
python add_stickers_to_video.py -i input_video.mp4 -o output_video.mp4 -s "sticker.png:forehead:1.5"
```

Add multiple stickers:

```bash
python add_stickers_to_video.py -i input_video.mp4 -o output_video.mp4 \
  -s "sticker1.png:forehead:1.5" \
  -s "sticker2.png:nose:1.0" \
  -s "sticker3.png:left_cheek:1.2"
```

Add rotated sticker:

```bash
python add_stickers_to_video.py -i input_video.mp4 -o output_video.mp4 \
  -s "sticker.png:left_cheek:1.2:45"
```

### Sticker Configuration Format

The sticker configuration follows this format:
```
path:position[:scale[:rotation]]
```

- **path**: Path to sticker image (PNG with transparency recommended)
- **position**: One of: `forehead`, `nose`, `left_cheek`, `right_cheek`, `left_eye`, `right_eye`, `chin`, `mouth`
- **scale**: (optional) Scale factor for sticker size (default: 1.0)
- **rotation**: (optional) Rotation angle in degrees (default: auto-calculated)

### Python API

You can also use the module programmatically:

```python
from mediapipe.sticker_overlay import MediaPipeStickerOverlay, StickerPosition
import cv2

# Initialize overlay
overlay = MediaPipeStickerOverlay(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load sticker
sticker = cv2.imread('sticker.png', cv2.IMREAD_UNCHANGED)

# Process a frame
frame = cv2.imread('frame.jpg')
stickers = [{
    'image': sticker,
    'position': StickerPosition.FOREHEAD,
    'scale': 1.5,
    'rotation': None  # Auto-calculate
}]

result = overlay.process_video_frame(frame, stickers)
cv2.imwrite('output.jpg', result)
```

## Available Sticker Positions

- `forehead`: Center of forehead
- `nose`: Tip of the nose
- `left_cheek`: Left cheek area
- `right_cheek`: Right cheek area
- `left_eye`: Left eye region
- `right_eye`: Right eye region
- `chin`: Chin area
- `mouth`: Mouth region

## Examples

Process a video from the examples folder:

```bash
cd ../ghost
python ../mediapipe/add_stickers_to_video.py \
  -i examples/videos/dance.mp4 \
  -o examples/results/dance_with_stickers.mp4 \
  -s "path/to/sticker.png:forehead:1.5"
```

## Requirements

- Python 3.7+
- mediapipe
- opencv-python
- numpy
- tqdm

## Notes

- Stickers work best with PNG images that have transparency (alpha channel)
- The module automatically handles face tracking across frames
- For best results, use stickers that are roughly square-shaped
- Processing time depends on video resolution and length

