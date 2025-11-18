"""
Example script demonstrating how to use the MediaPipe sticker overlay
"""

import cv2
import numpy as np
from sticker_overlay import MediaPipeStickerOverlay, StickerPosition


def create_sample_sticker(size=200, color=(0, 255, 0), shape='circle'):
    """
    Create a sample sticker for testing
    
    Args:
        size: Size of the sticker
        color: BGR color tuple
        shape: 'circle' or 'square'
    
    Returns:
        Sticker image with alpha channel
    """
    sticker = np.zeros((size, size, 4), dtype=np.uint8)
    
    if shape == 'circle':
        center = (size // 2, size // 2)
        radius = size // 2 - 10
        cv2.circle(sticker, center, radius, (*color, 255), -1)
    else:
        margin = 10
        cv2.rectangle(sticker, (margin, margin), (size - margin, size - margin), 
                     (*color, 255), -1)
    
    return sticker


def example_single_frame():
    """Example: Add sticker to a single image"""
    print("Example: Processing a single frame")
    
    # Initialize overlay
    overlay = MediaPipeStickerOverlay()
    
    # Load or create a test image
    # For this example, we'll create a simple test image
    # In practice, you would load: image = cv2.imread('path/to/image.jpg')
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (200, 200, 200)  # Gray background
    
    # Create a sample sticker
    sticker = create_sample_sticker(size=150, color=(0, 0, 255), shape='circle')
    
    # Configure sticker
    stickers = [{
        'image': sticker,
        'position': StickerPosition.FOREHEAD,
        'scale': 1.0,
        'rotation': None
    }]
    
    # Process frame
    result = overlay.process_video_frame(image, stickers)
    
    # Save result
    cv2.imwrite('example_output.jpg', result)
    print("Saved example_output.jpg")


def example_multiple_stickers():
    """Example: Add multiple stickers to a frame"""
    print("Example: Processing with multiple stickers")
    
    overlay = MediaPipeStickerOverlay()
    
    # Load image (replace with actual image path)
    # image = cv2.imread('path/to/image.jpg')
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (200, 200, 200)
    
    # Create multiple stickers
    stickers = [
        {
            'image': create_sample_sticker(size=100, color=(255, 0, 0), shape='circle'),
            'position': StickerPosition.LEFT_CHEEK,
            'scale': 1.0
        },
        {
            'image': create_sample_sticker(size=100, color=(0, 255, 0), shape='circle'),
            'position': StickerPosition.RIGHT_CHEEK,
            'scale': 1.0
        },
        {
            'image': create_sample_sticker(size=80, color=(0, 0, 255), shape='circle'),
            'position': StickerPosition.NOSE,
            'scale': 0.8
        }
    ]
    
    result = overlay.process_video_frame(image, stickers)
    cv2.imwrite('example_multiple_output.jpg', result)
    print("Saved example_multiple_output.jpg")


if __name__ == "__main__":
    print("MediaPipe Sticker Overlay Examples")
    print("=" * 50)
    
    # Note: These examples create dummy images
    # For real usage, load actual images with faces
    print("\nTo use with real images:")
    print("1. Load an image: image = cv2.imread('path/to/image.jpg')")
    print("2. Load a sticker: sticker = cv2.imread('sticker.png', cv2.IMREAD_UNCHANGED)")
    print("3. Use the overlay.process_video_frame() method")
    print("\nSee add_stickers_to_video.py for video processing examples")

