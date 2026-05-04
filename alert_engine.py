from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class AlertEngine:
    cooldown_seconds: float = 1.5
    last_alert_at: float = 0.0

    def severity_from_rules(self, features: Dict[str, float]) -> Tuple[str, float]:
        norm_ear = float(features.get("norm_ear", features.get("ear", 1.0)))
        norm_mar = float(features.get("norm_mar", features.get("mar", 0.0)))
        abs_pitch = abs(float(features.get("norm_pitch", features.get("pitch", 0.0))))
        abs_roll = abs(float(features.get("norm_roll", features.get("roll", 0.0))))
        abs_yaw = abs(float(features.get("norm_yaw", features.get("yaw", 0.0))))
        closure = float(features.get("eye_closed_frames", 0.0))

        if norm_ear < 0.65 or closure > 24:
            severity = "high"
            score = 0.95
        elif norm_ear < 0.78 or norm_mar > 1.35:
            severity = "medium"
            score = 0.75
        elif norm_ear < 0.90 or norm_mar > 1.15 or max(abs_pitch, abs_roll, abs_yaw) > 1.25:
            severity = "low"
            score = 0.55
        else:
            severity = "alert"
            score = 0.10
        return severity, score

    def trigger(self, severity: str) -> None:
        now = time.time()
        if severity == "alert":
            return
        if now - self.last_alert_at < self.cooldown_seconds and severity != "high":
            return
        self.last_alert_at = now

        try:
            import winsound

            if severity == "low":
                winsound.Beep(1000, 150)
            elif severity == "medium":
                winsound.Beep(1600, 300)
            else:
                for _ in range(3):
                    winsound.Beep(2200, 250)
        except Exception:
            # Alert fallback stays silent if the host does not support sound.
            return
