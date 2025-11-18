"""
Main script to add stickers to videos using MediaPipe Face Landmarker
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

from sticker_overlay import MediaPipeStickerOverlay, StickerPosition


def load_sticker(sticker_path: str) -> np.ndarray:
    """
    Load sticker image with alpha channel support
    
    Args:
        sticker_path: Path to sticker image (PNG with transparency recommended)
    
    Returns:
        Sticker image as numpy array (BGR or BGRA)
    """
    if not os.path.exists(sticker_path):
        raise FileNotFoundError(f"Sticker file not found: {sticker_path}")
    
    sticker = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
    
    if sticker is None:
        raise ValueError(f"Could not load sticker from: {sticker_path}")
    
    if sticker.shape[2] == 3:
        alpha = np.ones((sticker.shape[0], sticker.shape[1], 1), dtype=sticker.dtype) * 255
        sticker = np.concatenate([sticker, alpha], axis=2)
    
    return sticker


def read_video(video_path: str) -> tuple:
    """
    Read video and return frames and metadata
    
    Returns:
        frames: List of frames
        fps: Frames per second
        width: Video width
        height: Video height
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    print(f"Reading video: {total_frames} frames at {fps} FPS")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    return frames, fps, width, height


def write_video(frames: List[np.ndarray], 
                output_path: str, 
                fps: float, 
                width: int, 
                height: int,
                codec: str = 'mp4v') -> None:
    """
    Write frames to video file
    
    Args:
        frames: List of video frames
        output_path: Output video path
        fps: Frames per second
        width: Video width
        height: Video height
        codec: Video codec (default: 'mp4v')
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Writing video to: {output_path}")
    for frame in tqdm(frames, desc="Writing frames"):
        out.write(frame)
    
    out.release()
    print(f"Video saved successfully!")


def parse_sticker_config(sticker_configs: List[str]) -> List[Dict]:
    """
    Parse sticker configuration from command line arguments
    
    Format: "path:position:scale:rotation" or "path:position:scale" or "path:position"
    Example: "sticker.png:forehead:1.5" or "sticker.png:nose:1.0:45"
    
    Args:
        sticker_configs: List of sticker configuration strings
    
    Returns:
        List of sticker configuration dictionaries
    """
    stickers = []
    
    for config_str in sticker_configs:
        parts = config_str.split(':')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid sticker config format: {config_str}. Expected: path:position[:scale[:rotation]]")
        
        sticker_path = parts[0]
        position_str = parts[1].lower()
        scale = float(parts[2]) if len(parts) > 2 else 1.0
        rotation = float(parts[3]) if len(parts) > 3 else None
        
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
        
        if position_str not in position_map:
            raise ValueError(f"Invalid position: {position_str}. Valid positions: {list(position_map.keys())}")
        
        sticker_img = load_sticker(sticker_path)
        
        stickers.append({
            'image': sticker_img,
            'position': position_map[position_str],
            'scale': scale,
            'rotation': rotation
        })
    
    return stickers


def main():
    parser = argparse.ArgumentParser(
        description='Add stickers to videos using MediaPipe Face Landmarker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a single sticker on forehead
  python add_stickers_to_video.py -i input.mp4 -o output.mp4 -s "sticker.png:forehead:1.5"
  
  # Add multiple stickers
  python add_stickers_to_video.py -i input.mp4 -o output.mp4 \\
    -s "sticker1.png:forehead:1.5" -s "sticker2.png:nose:1.0"
  
  # Add rotated sticker
  python add_stickers_to_video.py -i input.mp4 -o output.mp4 \\
    -s "sticker.png:left_cheek:1.2:45"
  
Valid positions: forehead, nose, left_cheek, right_cheek, left_eye, right_eye, chin, mouth
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input video path')
    parser.add_argument('-o', '--output', required=True,
                       help='Output video path')
    parser.add_argument('-s', '--stickers', action='append', required=True,
                       help='Sticker configuration: path:position[:scale[:rotation]] (can specify multiple times)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5,
                       help='Minimum confidence for face detection (0.0-1.0)')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5,
                       help='Minimum confidence for face tracking (0.0-1.0)')
    parser.add_argument('--codec', type=str, default='mp4v',
                       help='Video codec (default: mp4v, use "XVID" or "H264" for better compatibility)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return
    
    try:
        stickers = parse_sticker_config(args.stickers)
        print(f"Loaded {len(stickers)} sticker(s)")
    except Exception as e:
        print(f"Error parsing sticker configurations: {e}")
        return
    
    try:
        frames, fps, width, height = read_video(args.input)
        print(f"Loaded {len(frames)} frames")
    except Exception as e:
        print(f"Error reading video: {e}")
        return
    
    overlay = MediaPipeStickerOverlay(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        enable_temporal_smoothing=True,
        enable_head_pose=True,
        enable_confidence_fallback=True
    )
    
    overlay.reset_temporal_state()
    
    print("Processing frames with stickers...")
    processed_frames = []
    
    for frame in tqdm(frames, desc="Processing frames"):
        processed_frame = overlay.process_video_frame(frame, stickers)
        processed_frames.append(processed_frame)
    
    try:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        write_video(processed_frames, args.output, fps, width, height, args.codec)
    except Exception as e:
        print(f"Error writing video: {e}")
        return
    
    print("Done!")


if __name__ == "__main__":
    main()

