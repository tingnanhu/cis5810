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
    CUSTOM = "custom"


class MediaPipeStickerOverlay:
    """Class for overlaying stickers on faces in videos using MediaPipe"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 enable_temporal_smoothing: bool = True,
                 smoothing_alpha: float = 0.7,
                 enable_head_pose: bool = True,
                 enable_confidence_fallback: bool = True,
                 min_confidence_threshold: float = 0.3):
        """
        Initialize MediaPipe Face Mesh
        
        Args:
            model_path: Path to custom MediaPipe model (None for default)
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            enable_temporal_smoothing: Enable temporal smoothing for stable positioning
            smoothing_alpha: EMA smoothing factor (0.0-1.0, higher = more smoothing)
            enable_head_pose: Enable head pose-aware rotation adjustments
            enable_confidence_fallback: Enable fallback to last known position on low confidence
            min_confidence_threshold: Minimum confidence to use current detection (otherwise use fallback)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        except (TypeError, RuntimeError):
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
        self.landmark_indices = {
            StickerPosition.FOREHEAD: [10, 151, 9, 10, 337, 299],
            StickerPosition.NOSE: [4, 5, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 363, 360],
            StickerPosition.LEFT_CHEEK: [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147],
            StickerPosition.RIGHT_CHEEK: [346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 416, 376, 433, 410, 454],
            StickerPosition.LEFT_EYE: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            StickerPosition.RIGHT_EYE: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            StickerPosition.CHIN: [18, 200, 199, 175, 169, 170, 140, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
            StickerPosition.MOUTH: [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
        }
        
        self.pose_landmarks = {
            'nose_tip': 4,
            'chin': 18,
            'left_eye': 33,
            'right_eye': 263,
            'left_mouth': 61,
            'right_mouth': 291,
        }
        
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.smoothing_alpha = smoothing_alpha
        self.enable_head_pose = enable_head_pose
        self.enable_confidence_fallback = enable_confidence_fallback
        self.min_confidence_threshold = min_confidence_threshold
        
        self.previous_frame_data = {}
    
    def get_landmark_points(self, landmarks, indices: List[int], image_width: int, image_height: int) -> np.ndarray:
        """Extract specific landmark points"""
        points = []
        
        for idx in indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                points.append([x, y])
        
        return np.array(points)
    
    def estimate_head_pose(self, landmarks, image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Estimate head pose (yaw and pitch) from facial landmarks
        
        Returns:
            yaw: Rotation around vertical axis (degrees, negative = left, positive = right)
            pitch: Rotation around horizontal axis (degrees, negative = up, positive = down)
        """
        try:
            nose_tip = landmarks.landmark[self.pose_landmarks['nose_tip']]
            chin = landmarks.landmark[self.pose_landmarks['chin']]
            left_eye = landmarks.landmark[self.pose_landmarks['left_eye']]
            right_eye = landmarks.landmark[self.pose_landmarks['right_eye']]
            left_mouth = landmarks.landmark[self.pose_landmarks['left_mouth']]
            right_mouth = landmarks.landmark[self.pose_landmarks['right_mouth']]
            
            nose = np.array([nose_tip.x * image_width,
                            nose_tip.y * image_height])
            chin_pt = np.array([chin.x * image_width,
                               chin.y * image_height])
            left_eye_pt = np.array([left_eye.x * image_width,
                                   left_eye.y * image_height])
            right_eye_pt = np.array([right_eye.x * image_width,
                                    right_eye.y * image_height])
            
            eye_vector = right_eye_pt - left_eye_pt
            eye_distance = np.linalg.norm(eye_vector)
            
            if eye_distance > 0:
                eye_vector_norm = eye_vector / eye_distance
                yaw = np.degrees(np.arcsin(
                    np.clip(eye_vector_norm[1], -1, 1)))
            else:
                yaw = 0
            
            nose_to_chin = chin_pt - nose
            
            if np.linalg.norm(nose_to_chin) > 0:
                pitch = np.degrees(np.arcsin(np.clip(nose_to_chin[1] / np.linalg.norm(nose_to_chin), -1, 1)))
            else:
                pitch = 0
            
            return yaw, pitch
            
        except (KeyError, IndexError, AttributeError):
            return 0.0, 0.0
    
    def apply_temporal_smoothing(self, 
                                 current_center: np.ndarray,
                                 current_size: float,
                                 current_angle: float,
                                 face_index: int,
                                 position: StickerPosition) -> Tuple[np.ndarray, float, float]:
        """
        Apply exponential moving average smoothing to reduce jitter
        
        Args:
            current_center: Current center position
            current_size: Current size
            current_angle: Current rotation angle
            face_index: Index of face (for multi-face scenarios)
            position: Sticker position
        
        Returns:
            Smoothed (center, size, angle)
        """
        if not self.enable_temporal_smoothing:
            return current_center, current_size, current_angle
        
        key = (face_index, position)
        
        if key in self.previous_frame_data:
            prev_center, prev_size, prev_angle, _ = self.previous_frame_data[key]
            
            alpha = self.smoothing_alpha
            smoothed_center = alpha * current_center + (1 - alpha) * prev_center
            smoothed_size = alpha * current_size + (1 - alpha) * prev_size
            
            angle_diff = current_angle - prev_angle
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
            smoothed_angle = prev_angle + alpha * angle_diff
            
            return smoothed_center.astype(int), smoothed_size, smoothed_angle
        else:
            return current_center, current_size, current_angle
    
    def get_position_from_landmarks(self, 
                                   landmarks, 
                                   position: StickerPosition,
                                   image_width: int,
                                   image_height: int,
                                   custom_indices: Optional[List[int]] = None,
                                   use_head_pose: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Get sticker position, size, and angle from landmarks
        
        Args:
            landmarks: MediaPipe face landmarks
            position: Sticker position
            image_width: Image width
            image_height: Image height
            custom_indices: Custom landmark indices (if position is CUSTOM)
            use_head_pose: Whether to adjust angle based on head pose
        
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
        
        center = np.mean(points, axis=0).astype(int)
        
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        size = max(width, height) * 1.5
        
        angle = 0
        if len(points) >= 2:
            dx = points[-1][0] - points[0][0]
            dy = points[-1][1] - points[0][1]
            angle = np.degrees(np.arctan2(dy, dx))
        
        if use_head_pose and self.enable_head_pose:
            yaw, pitch = self.estimate_head_pose(landmarks, image_width, image_height)
            angle += yaw * 0.5
        
        return center, size, angle
    
    def overlay_sticker(self,
                       image: np.ndarray,
                       sticker: np.ndarray,
                       position: StickerPosition,
                       custom_indices: Optional[List[int]] = None,
                       scale: float = 1.0,
                       rotation: Optional[float] = None,
                       face_index: int = 0,
                       is_video_frame: bool = False) -> np.ndarray:
        """
        Overlay a sticker on the face in the image
        
        Args:
            image: Input image (BGR format)
            sticker: Sticker image (BGR format with alpha channel if available)
            position: Where to place the sticker
            custom_indices: Custom landmark indices (if position is CUSTOM)
            scale: Scale factor for sticker size
            rotation: Override rotation angle (None to use calculated angle)
            face_index: Index of face (for multi-face scenarios and tracking)
            is_video_frame: Whether this is a video frame (enables smoothing/fallback)
        
        Returns:
            Image with sticker overlaid
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        h, w = image.shape[:2]
        key = (face_index, position)
        
        if not results.multi_face_landmarks:
            if is_video_frame and self.enable_confidence_fallback and key in self.previous_frame_data:
                prev_center, prev_size, prev_angle, _ = self.previous_frame_data[key]
                center, size, angle = prev_center, prev_size, prev_angle
            else:
                return image
        else:
            face_idx = min(face_index, len(results.multi_face_landmarks) - 1)
            face_landmarks = results.multi_face_landmarks[face_idx]
            
            center, size, angle = self.get_position_from_landmarks(
                face_landmarks, position, w, h, custom_indices, use_head_pose=True
            )
            
            if center is None:
                if is_video_frame and self.enable_confidence_fallback and key in self.previous_frame_data:
                    prev_center, prev_size, prev_angle, _ = self.previous_frame_data[key]
                    center, size, angle = prev_center, prev_size, prev_angle
                else:
                    return image
            
            confidence = 1.0
            
            if is_video_frame and self.enable_confidence_fallback:
                if confidence < self.min_confidence_threshold and key in self.previous_frame_data:
                    prev_center, prev_size, prev_angle, _ = self.previous_frame_data[key]
                    center, size, angle = prev_center, prev_size, prev_angle
                else:
                    center, size, angle = self.apply_temporal_smoothing(
                        center, size, angle, face_index, position
                    )
            
            if is_video_frame:
                self.previous_frame_data[key] = (center.copy(), size, angle, confidence)
        
        size = int(size * scale)
        
        if rotation is not None:
            angle = rotation
        
        sticker_resized = cv2.resize(sticker, (size, size))
        
        if abs(angle) > 1:
            M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
            sticker_resized = cv2.warpAffine(sticker_resized, M, (size, size))
        
        x1 = max(0, center[0] - size // 2)
        y1 = max(0, center[1] - size // 2)
        x2 = min(w, center[0] + size // 2)
        y2 = min(h, center[1] + size // 2)
        
        sticker_x1 = max(0, size // 2 - center[0])
        sticker_y1 = max(0, size // 2 - center[1])
        sticker_x2 = sticker_x1 + (x2 - x1)
        sticker_y2 = sticker_y1 + (y2 - y1)
        
        sticker_region = sticker_resized[sticker_y1:sticker_y2, sticker_x1:sticker_x2]
        
        if sticker_region.shape[2] == 4:
            alpha = sticker_region[:, :, 3:4] / 255.0
            bgr = sticker_region[:, :, :3]
            
            roi = image[y1:y2, x1:x2]
            if roi.shape[:2] != bgr.shape[:2]:
                bgr = cv2.resize(bgr, (roi.shape[1], roi.shape[0]))
                alpha = cv2.resize(alpha, (roi.shape[1], roi.shape[0]))
                alpha = alpha[:, :, np.newaxis]
            blended = (alpha * bgr + (1 - alpha) * roi).astype(np.uint8)
            image[y1:y2, x1:x2] = blended
        else:
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
                - 'face_index': int (optional, default 0, for multi-face scenarios)
        
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
            face_index = sticker_config.get('face_index', 0)
            
            result_frame = self.overlay_sticker(
                result_frame,
                sticker_img,
                position,
                custom_indices,
                scale,
                rotation,
                face_index=face_index,
                is_video_frame=True
            )
        
        return result_frame
    
    def reset_temporal_state(self):
        """Reset temporal smoothing state (useful when starting a new video)"""
        self.previous_frame_data = {}
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

