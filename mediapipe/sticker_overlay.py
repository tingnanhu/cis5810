"""
MediaPipe-based sticker overlay for videos
Uses MediaPipe Face Landmarker to detect facial landmarks and overlay stickers
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
from enum import Enum


class StickerPosition(Enum):
    """Predefined sticker positions on face"""
    FOREHEAD = "forehead"
    NOSE = "nose"
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    CHIN = "chin"
    MOUTH = "mouth"
    CUSTOM = "custom"  # Use custom landmark indices


class MediaPipeStickerOverlay:
    """Class for overlaying stickers on faces in videos using MediaPipe"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Face Mesh
        
        Args:
            model_path: Path to custom MediaPipe model (None for default)
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Face Mesh
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,  # Use True for better compatibility
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        except (TypeError, RuntimeError):
            # Fallback for older MediaPipe versions
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
        # Face landmark indices for different regions
        # MediaPipe Face Mesh has 468 landmarks
        self.landmark_indices = {
            StickerPosition.FOREHEAD: [10, 151, 9, 10, 337, 299],  # Forehead region
            StickerPosition.NOSE: [4, 5, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 363, 360],  # Nose region
            StickerPosition.LEFT_CHEEK: [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147],  # Left cheek
            StickerPosition.RIGHT_CHEEK: [346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 416, 376, 433, 410, 454],  # Right cheek
            StickerPosition.LEFT_EYE: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],  # Left eye
            StickerPosition.RIGHT_EYE: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],  # Right eye
            StickerPosition.CHIN: [18, 200, 199, 175, 169, 170, 140, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],  # Chin
            StickerPosition.MOUTH: [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],  # Mouth region
        }
    
    def get_landmark_points(self, landmarks, indices: List[int], image_width: int, image_height: int) -> np.ndarray:
        """Extract specific landmark points"""
        points = []
        
        for idx in indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                # Convert normalized coordinates (0-1) to pixel coordinates
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                points.append([x, y])
        
        return np.array(points)
    
    def get_position_from_landmarks(self, 
                                   landmarks, 
                                   position: StickerPosition,
                                   image_width: int,
                                   image_height: int,
                                   custom_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, float, float]:
        """
        Get sticker position, size, and angle from landmarks
        
        Returns:
            center: (x, y) center point
            size: sticker size (width/height)
            angle: rotation angle in degrees
        """
        if position == StickerPosition.CUSTOM and custom_indices:
            indices = custom_indices
        else:
            indices = self.landmark_indices.get(position, self.landmark_indices[StickerPosition.FOREHEAD])
        
        points = self.get_landmark_points(landmarks, indices, image_width, image_height)
        
        if len(points) == 0:
            return None, 0, 0
        
        # Calculate center
        center = np.mean(points, axis=0).astype(int)
        
        # Calculate size based on bounding box
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        size = max(width, height) * 1.5  # Add some padding
        
        # Calculate angle (optional, for rotation)
        angle = 0
        if len(points) >= 2:
            # Use first and last point to estimate angle
            dx = points[-1][0] - points[0][0]
            dy = points[-1][1] - points[0][1]
            angle = np.degrees(np.arctan2(dy, dx))
        
        return center, size, angle
    
    def overlay_sticker(self,
                       image: np.ndarray,
                       sticker: np.ndarray,
                       position: StickerPosition,
                       custom_indices: Optional[List[int]] = None,
                       scale: float = 1.0,
                       rotation: Optional[float] = None) -> np.ndarray:
        """
        Overlay a sticker on the face in the image
        
        Args:
            image: Input image (BGR format)
            sticker: Sticker image (BGR format with alpha channel if available)
            position: Where to place the sticker
            custom_indices: Custom landmark indices (if position is CUSTOM)
            scale: Scale factor for sticker size
            rotation: Override rotation angle (None to use calculated angle)
        
        Returns:
            Image with sticker overlaid
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return image
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Get position and size
        center, size, angle = self.get_position_from_landmarks(
            face_landmarks, position, w, h, custom_indices
        )
        
        if center is None:
            return image
        
        # Apply scale
        size = int(size * scale)
        
        # Use provided rotation or calculated angle
        if rotation is not None:
            angle = rotation
        
        # Resize sticker
        sticker_resized = cv2.resize(sticker, (size, size))
        
        # Rotate sticker if needed
        if abs(angle) > 1:
            M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
            sticker_resized = cv2.warpAffine(sticker_resized, M, (size, size))
        
        # Calculate position (center the sticker)
        x1 = max(0, center[0] - size // 2)
        y1 = max(0, center[1] - size // 2)
        x2 = min(w, center[0] + size // 2)
        y2 = min(h, center[1] + size // 2)
        
        # Adjust sticker if it goes out of bounds
        sticker_x1 = max(0, size // 2 - center[0])
        sticker_y1 = max(0, size // 2 - center[1])
        sticker_x2 = sticker_x1 + (x2 - x1)
        sticker_y2 = sticker_y1 + (y2 - y1)
        
        # Extract sticker region
        sticker_region = sticker_resized[sticker_y1:sticker_y2, sticker_x1:sticker_x2]
        
        # Handle alpha channel if present
        if sticker_region.shape[2] == 4:
            alpha = sticker_region[:, :, 3:4] / 255.0
            bgr = sticker_region[:, :, :3]  # Already in BGR format from OpenCV
            
            # Blend with background
            roi = image[y1:y2, x1:x2]
            if roi.shape[:2] != bgr.shape[:2]:
                # Resize if dimensions don't match
                bgr = cv2.resize(bgr, (roi.shape[1], roi.shape[0]))
                alpha = cv2.resize(alpha, (roi.shape[1], roi.shape[0]))
                alpha = alpha[:, :, np.newaxis]
            blended = (alpha * bgr + (1 - alpha) * roi).astype(np.uint8)
            image[y1:y2, x1:x2] = blended
        else:
            # No alpha channel, just paste (already in BGR format)
            if sticker_region.shape[:2] != image[y1:y2, x1:x2].shape[:2]:
                sticker_region = cv2.resize(sticker_region, (x2 - x1, y2 - y1))
            image[y1:y2, x1:x2] = sticker_region
        
        return image
    
    def process_video_frame(self,
                           frame: np.ndarray,
                           stickers: List[Dict]) -> np.ndarray:
        """
        Process a single video frame with multiple stickers
        
        Args:
            frame: Video frame (BGR format)
            stickers: List of sticker dictionaries with keys:
                - 'image': sticker image (numpy array)
                - 'position': StickerPosition enum
                - 'scale': float (optional, default 1.0)
                - 'rotation': float (optional, default None)
                - 'custom_indices': List[int] (optional, for CUSTOM position)
        
        Returns:
            Frame with stickers overlaid
        """
        result_frame = frame.copy()
        
        for sticker_config in stickers:
            sticker_img = sticker_config['image']
            position = sticker_config['position']
            scale = sticker_config.get('scale', 1.0)
            rotation = sticker_config.get('rotation', None)
            custom_indices = sticker_config.get('custom_indices', None)
            
            result_frame = self.overlay_sticker(
                result_frame,
                sticker_img,
                position,
                custom_indices,
                scale,
                rotation
            )
        
        return result_frame
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

