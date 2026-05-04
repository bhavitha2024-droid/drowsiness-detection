from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


SEVERITY_COLORS = {
    "alert": (80, 200, 120),
    "low": (0, 215, 255),
    "medium": (0, 140, 255),
    "high": (0, 0, 255),
}


def draw_dashboard(frame: np.ndarray, features: Dict[str, float], fps: float) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (390, 245), (20, 20, 20), -1)
    output = cv2.addWeighted(overlay, 0.45, output, 0.55, 0)

    severity = str(features.get("severity", "alert"))
    color = SEVERITY_COLORS.get(severity, (255, 255, 255))
    lines = [
        f"Severity: {severity.upper()} ({features.get('severity_score', 0.0):.2f})",
        f"Label: {str(features.get('active_label', severity)).upper()}",
        f"EAR: {features.get('ear', 0.0):.3f}",
        f"MAR: {features.get('mar', 0.0):.3f}",
        f"Pitch: {features.get('pitch', 0.0):.1f}",
        f"Roll: {features.get('roll', 0.0):.1f}",
        f"Yaw: {features.get('yaw', 0.0):.1f}",
        f"Blinks: {int(features.get('blink_count', 0))}",
        f"Yawns: {int(features.get('yawn_count', 0))}",
        f"Closed Frames: {int(features.get('eye_closed_frames', 0))}",
        f"Baseline: {features.get('baseline_progress', 0.0) * 100:.0f}%",
        f"FPS: {fps:.1f}",
    ]

    for idx, line in enumerate(lines):
        line_color = color if idx == 0 else (240, 240, 240)
        cv2.putText(
            output,
            line,
            (24, 40 + idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            line_color,
            1,
            cv2.LINE_AA,
        )

    if severity != "alert":
        cv2.putText(
            output,
            f"{severity.upper()} DROWSINESS",
            (24, 270),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
    return output
