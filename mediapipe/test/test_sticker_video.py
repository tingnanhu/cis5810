"""
Test script for adding stickers to videos using MediaPipe
"""
import cv2
import argparse
import os
import sys
import numpy as np
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
    
    # If no alpha channel, add one
    if sticker.shape[2] == 3:
        alpha = np.ones((sticker.shape[0], sticker.shape[1], 1), 
                       dtype=sticker.dtype) * 255
        sticker = np.concatenate([sticker, alpha], axis=2)
    
    return sticker


def main():
    parser = argparse.ArgumentParser(
        description='Add stickers to videos using MediaPipe',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single sticker
  python test_sticker_video.py \\
    -i ../ghost/examples/videos/dance.mp4 \\
    -o output.mp4 \\
    -s path/to/sticker.png \\
    -p forehead

  # Multiple stickers
  python test_sticker_video.py \\
    -i input.mp4 \\
    -o output.mp4 \\
    -s sticker1.png:forehead:1.5 \\
    -s sticker2.png:nose:1.0

  # Test with limited frames
  python test_sticker_video.py \\
    -i input.mp4 \\
    -o output.mp4 \\
    -s sticker.png:forehead \\
    --max-frames 100
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input video path')
    parser.add_argument('-o', '--output', required=True,
                       help='Output video path')
    parser.add_argument('-s', '--stickers', action='append', required=True,
                       help='Sticker configuration: path[:position[:scale[:rotation]]] (can specify multiple times)')
    parser.add_argument('-p', '--position', 
                       choices=['forehead', 'nose', 'left_cheek', 'right_cheek', 
                               'left_eye', 'right_eye', 'chin', 'mouth'],
                       help='Default position if not specified in sticker config (default: forehead)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Default scale factor if not specified in sticker config (default: 1.0)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (for testing, default: all frames)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5,
                       help='Minimum confidence for face detection (0.0-1.0)')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5,
                       help='Minimum confidence for face tracking (0.0-1.0)')
    parser.add_argument('--codec', type=str, default='mp4v',
                       help='Video codec (default: mp4v, use "XVID" or "H264" for better compatibility)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return
    
    # Parse sticker configurations
    stickers = []
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
    
    default_position = position_map.get(args.position or 'forehead', StickerPosition.FOREHEAD)
    
    print("Loading stickers...")
    for sticker_config in args.stickers:
        parts = sticker_config.split(':')
        sticker_path = parts[0]
        
        # Parse position, scale, rotation from config string
        position_str = parts[1].lower() if len(parts) > 1 else None
        scale = float(parts[2]) if len(parts) > 2 else args.scale
        rotation = float(parts[3]) if len(parts) > 3 else None
        
        position = position_map.get(position_str, default_position) if position_str else default_position
        
        try:
            sticker_img = load_sticker(sticker_path)
            print(f"  Loaded: {sticker_path} ({sticker_img.shape[1]}x{sticker_img.shape[0]})")
            stickers.append({
                'image': sticker_img,
                'position': position,
                'scale': scale,
                'rotation': rotation
            })
        except Exception as e:
            print(f"Error loading sticker {sticker_path}: {e}")
            return
    
    print(f"Loaded {len(stickers)} sticker(s)")
    
    # Open video
    print(f"\nOpening video: {args.input}")
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.input}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    if args.max_frames:
        print(f"  Processing first {args.max_frames} frames only")
    
    # Initialize sticker overlay
    print("\nInitializing MediaPipe...")
    overlay = MediaPipeStickerOverlay(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    
    # Prepare output video writer
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {args.output}")
        cap.release()
        return
    
    # Process frames
    print("\nProcessing frames...")
    frame_count = 0
    processed_count = 0
    
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=min(args.max_frames or total_frames, total_frames), 
                           desc="Processing frames")
    except ImportError:
        progress_bar = None
        print("(Install tqdm for progress bar: pip install tqdm)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Check if we've reached max frames
        if args.max_frames and frame_count > args.max_frames:
            break
        
        # Process frame with stickers
        try:
            processed_frame = overlay.process_video_frame(frame, stickers)
            out.write(processed_frame)
            processed_count += 1
        except Exception as e:
            print(f"\nError processing frame {frame_count}: {e}")
            # Write original frame if processing fails
            out.write(frame)
        
        if progress_bar:
            progress_bar.update(1)
        elif frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")
    
    if progress_bar:
        progress_bar.close()
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n✓ Success! Processed {processed_count} frames")
    print(f"  Output saved to: {args.output}")
    print(f"  Video properties: {width}x{height} @ {fps:.2f} FPS")


if __name__ == "__main__":
    main()

