from __future__ import annotations

import cv2
import numpy as np


def apply_gamma(frame: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
        "uint8"
    )
    return cv2.LUT(frame, table)


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def histogram_equalization(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    equalized = cv2.equalizeHist(y_channel)
    merged = cv2.merge((equalized, cr_channel, cb_channel))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    enhanced = apply_gamma(frame)
    enhanced = apply_clahe(enhanced)
    enhanced = histogram_equalization(enhanced)
    return enhanced
