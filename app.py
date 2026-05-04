from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2

from alert_engine import AlertEngine
from dashboard import draw_dashboard
from feature_extraction import FeatureExtractor
from low_light import enhance_low_light
from personalize import PersonalizationEngine
from temporal_model import RealtimeTemporalPredictor


KEY_TO_LABEL = {
    ord("0"): "alert",
    ord("1"): "low",
    ord("2"): "medium",
    ord("3"): "high",
    ord("a"): "alert",
    ord("l"): "low",
    ord("m"): "medium",
    ord("h"): "high",
}

LOG_COLUMNS = [
    "timestamp",
    "ear",
    "mar",
    "pitch",
    "roll",
    "yaw",
    "blink_count",
    "yawn_count",
    "eye_closed_frames",
    "norm_ear",
    "norm_mar",
    "norm_pitch",
    "norm_roll",
    "norm_yaw",
    "baseline_progress",
    "severity",
    "severity_score",
    "label",
]


def ensure_layout() -> None:
    for directory in ("dataset", "models", "logs", "alerts"):
        Path(directory).mkdir(parents=True, exist_ok=True)


def open_logger(path: Path) -> tuple[csv.DictWriter, object]:
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
    if path.stat().st_size == 0:
        writer.writeheader()
    return writer, handle


def maybe_load_temporal_model(model_path: Path, sequence_length: int = 30):
    if not model_path.exists():
        return None
    try:
        return RealtimeTemporalPredictor(model_path=model_path, sequence_length=sequence_length)
    except Exception as exc:
        print(f"Temporal model disabled: {exc}")
        return None


def main() -> None:
    ensure_layout()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    extractor = FeatureExtractor()
    personalization = PersonalizationEngine(warmup_seconds=60.0, fps_hint=20.0)
    alerts = AlertEngine()
    temporal_predictor = maybe_load_temporal_model(Path("models/drowsiness_lstm.pt"))

    writer, handle = open_logger(Path("logs/session_features.csv"))
    previous_time = time.time()
    active_label = "alert"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Unable to read from webcam.")
                break

            enhanced = enhance_low_light(frame)
            annotated, features = extractor.extract(enhanced)
            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            if features is not None:
                personalization.update(features)
                features = personalization.normalize(features)

                predicted_label = None
                predicted_score = 0.0
                if temporal_predictor and personalization.is_ready():
                    predicted_label, predicted_score = temporal_predictor.push(features)

                if predicted_label is None:
                    predicted_label, predicted_score = alerts.severity_from_rules(features)

                features["severity"] = predicted_label
                features["severity_score"] = predicted_score
                alerts.trigger(predicted_label)

                row = {
                    "timestamp": current_time,
                    "ear": features.get("ear", 0.0),
                    "mar": features.get("mar", 0.0),
                    "pitch": features.get("pitch", 0.0),
                    "roll": features.get("roll", 0.0),
                    "yaw": features.get("yaw", 0.0),
                    "blink_count": features.get("blink_count", 0.0),
                    "yawn_count": features.get("yawn_count", 0.0),
                    "eye_closed_frames": features.get("eye_closed_frames", 0.0),
                    "norm_ear": features.get("norm_ear", 0.0),
                    "norm_mar": features.get("norm_mar", 0.0),
                    "norm_pitch": features.get("norm_pitch", 0.0),
                    "norm_roll": features.get("norm_roll", 0.0),
                    "norm_yaw": features.get("norm_yaw", 0.0),
                    "baseline_progress": features.get("baseline_progress", 0.0),
                    "severity": predicted_label,
                    "severity_score": predicted_score,
                    "label": active_label,
                }
                writer.writerow(row)
                features["active_label"] = active_label

                annotated = draw_dashboard(annotated, features, fps)
            else:
                cv2.putText(
                    annotated,
                    "No face detected",
                    (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Driver Drowsiness Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in KEY_TO_LABEL:
                active_label = KEY_TO_LABEL[key]
            if key == ord("q"):
                break
    finally:
        handle.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
