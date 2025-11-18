"""
Test script for adding stickers to images using MediaPipe
"""
import cv2
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import sticker_overlay
sys.path.insert(0, str(Path(__file__).parent.parent))
from sticker_overlay import MediaPipeStickerOverlay, StickerPosition


def load_sticker(sticker_path: str):
    """Load sticker image with alpha channel support"""
    if not os.path.exists(sticker_path):
        raise FileNotFoundError(f"Sticker file not found: {sticker_path}")
    
    sticker = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
    if sticker is None:
        raise ValueError(f"Could not load sticker from: {sticker_path}")
    
    if sticker.shape[2] == 3:
        alpha = np.ones((sticker.shape[0], sticker.shape[1], 1), 
                       dtype=sticker.dtype) * 255
        sticker = np.concatenate([sticker, alpha], axis=2)
    
    return sticker


def main():
    parser = argparse.ArgumentParser(
        description='Add stickers to images using MediaPipe'
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input image path')
    parser.add_argument('-o', '--output', required=True,
                       help='Output image path')
    parser.add_argument('-s', '--sticker', required=True,
                       help='Path to sticker image (PNG with transparency recommended)')
    parser.add_argument('-p', '--position', 
                       choices=['forehead', 'nose', 'left_cheek', 'right_cheek', 
                               'left_eye', 'right_eye', 'chin', 'mouth'],
                       default='forehead',
                       help='Where to place sticker (default: forehead)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor for sticker size (default: 1.0)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5,
                       help='Minimum confidence for face detection (0.0-1.0)')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5,
                       help='Minimum confidence for face tracking (0.0-1.0)')
    parser.add_argument('--iteration2', action='store_true',
                       help='Enable Iteration 2 improvements (temporal smoothing, head pose, fallback)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        return
    
    print(f"Loading image: {args.input}")
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load image: {args.input}")
        return
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    print(f"Loading sticker: {args.sticker}")
    try:
        sticker = load_sticker(args.sticker)
        print(f"Sticker size: {sticker.shape[1]}x{sticker.shape[0]}")
        if sticker.shape[2] == 4:
            print("Sticker has alpha channel (transparency)")
    except Exception as e:
        print(f"Error loading sticker: {e}")
        return
    
    position_map = {
        'forehead': StickerPosition.FOREHEAD,
        'nose': StickerPosition.NOSE,
        'left_cheek': StickerPosition.LEFT_CHEEK,
        'right_cheek': StickerPosition.RIGHT_CHEEK,
        'left_eye': StickerPosition.LEFT_EYE,
        'right_eye': StickerPosition.RIGHT_EYE,
        'chin': StickerPosition.CHIN,
        'mouth': StickerPosition.MOUTH,
    }
    position = position_map[args.position]
    
    print("\nInitializing MediaPipe...")
    if args.iteration2:
        print("  Using Iteration 2 improvements (head pose-aware rotation)")
        overlay = MediaPipeStickerOverlay(
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            enable_temporal_smoothing=False,
            enable_head_pose=True,
            enable_confidence_fallback=False
        )
    else:
        overlay = MediaPipeStickerOverlay(
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence
        )
    
    print(f"Processing image with sticker on {args.position}...")
    try:
        result = overlay.overlay_sticker(
            image,
            sticker,
            position,
            scale=args.scale
        )
        
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(args.output, result)
        print(f"\n✓ Success! Output saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import numpy as np
    main()

