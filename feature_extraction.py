from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from mediapipe import Image, ImageFormat, tasks
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 13, 311, 291, 178, 14, 402]
POSE_POINTS = [1, 33, 263, 61, 291, 199]

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

FACE_MODEL_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (-30.0, 35.0, -30.0),
        (30.0, 35.0, -30.0),
        (-25.0, -30.0, -30.0),
        (25.0, -30.0, -30.0),
        (0.0, -63.0, -12.0),
    ],
    dtype=np.float64,
)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points: Sequence[np.ndarray]) -> float:
    vertical_1 = _distance(points[1], points[5])
    vertical_2 = _distance(points[2], points[4])
    horizontal = _distance(points[0], points[3])
    return (vertical_1 + vertical_2) / max(2.0 * horizontal, 1e-6)


def mouth_aspect_ratio(points: Sequence[np.ndarray]) -> float:
    vertical_1 = _distance(points[1], points[7])
    vertical_2 = _distance(points[2], points[6])
    vertical_3 = _distance(points[3], points[5])
    horizontal = _distance(points[0], points[4])
    return (vertical_1 + vertical_2 + vertical_3) / max(3.0 * horizontal, 1e-6)


def ensure_face_landmarker_model(model_path: str | Path) -> Path:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, path)
    return path


def _landmark_to_pixel(landmark, width: int, height: int) -> np.ndarray:
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float64)


def _rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0.0
    return tuple(math.degrees(v) for v in (x, y, z))


@dataclass
class FeatureExtractor:
    model_path: str = "models/face_landmarker.task"
    num_faces: int = 1
    eye_close_threshold: float = 0.21
    yawn_threshold: float = 0.65
    blink_count: int = 0
    yawn_count: int = 0
    eye_closed_frames: int = 0
    yawn_active: bool = False
    was_eye_closed: bool = False
    frame_index: int = 0
    landmarker: FaceLandmarker = field(init=False)

    def __post_init__(self) -> None:
        asset_path = ensure_face_landmarker_model(self.model_path)
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(asset_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=self.num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def extract(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float] | None]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms = self.frame_index * 33
        self.frame_index += 1
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not results.face_landmarks:
            return frame, None

        face_landmarks = results.face_landmarks[0]
        annotated = frame.copy()
        self._draw_landmarks(annotated, face_landmarks)

        height, width = frame.shape[:2]
        left_eye_pts = [_landmark_to_pixel(face_landmarks[idx], width, height) for idx in LEFT_EYE]
        right_eye_pts = [_landmark_to_pixel(face_landmarks[idx], width, height) for idx in RIGHT_EYE]
        mouth_pts = [_landmark_to_pixel(face_landmarks[idx], width, height) for idx in MOUTH]

        ear_left = eye_aspect_ratio(left_eye_pts)
        ear_right = eye_aspect_ratio(right_eye_pts)
        ear = (ear_left + ear_right) / 2.0
        mar = mouth_aspect_ratio(mouth_pts)
        pitch, yaw, roll = self._head_pose(face_landmarks, width, height)

        if ear < self.eye_close_threshold:
            self.eye_closed_frames += 1
            self.was_eye_closed = True
        else:
            if self.was_eye_closed and self.eye_closed_frames > 1:
                self.blink_count += 1
            self.eye_closed_frames = 0
            self.was_eye_closed = False

        if mar > self.yawn_threshold and not self.yawn_active:
            self.yawn_count += 1
            self.yawn_active = True
        elif mar <= self.yawn_threshold:
            self.yawn_active = False

        features = {
            "ear": float(ear),
            "mar": float(mar),
            "pitch": float(pitch),
            "roll": float(roll),
            "yaw": float(yaw),
            "blink_count": float(self.blink_count),
            "yawn_count": float(self.yawn_count),
            "eye_closed_frames": float(self.eye_closed_frames),
        }
        return annotated, features

    def _draw_landmarks(self, frame: np.ndarray, landmarks: List[object]) -> None:
        height, width = frame.shape[:2]
        for idx in LEFT_EYE + RIGHT_EYE + MOUTH + POSE_POINTS:
            point = _landmark_to_pixel(landmarks[idx], width, height).astype(int)
            cv2.circle(frame, tuple(point), 2, (0, 255, 255), -1, cv2.LINE_AA)

    def _head_pose(self, landmarks: List[object], width: int, height: int) -> Tuple[float, float, float]:
        image_points = np.array(
            [_landmark_to_pixel(landmarks[idx], width, height) for idx in POSE_POINTS],
            dtype=np.float64,
        )
        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, _ = cv2.solvePnP(
            FACE_MODEL_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = _rotation_matrix_to_euler(rotation_matrix)
        return pitch, yaw, roll
