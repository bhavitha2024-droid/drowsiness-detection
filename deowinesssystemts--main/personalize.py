from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


BASELINE_KEYS = ("ear", "mar", "pitch", "roll", "yaw")


@dataclass
class PersonalizationEngine:
    warmup_seconds: float = 60.0
    fps_hint: float = 20.0
    samples: Dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in BASELINE_KEYS}
    )

    def update(self, features: Dict[str, float]) -> None:
        for key in BASELINE_KEYS:
            value = float(features.get(key, 0.0))
            self.samples[key].append(value)

    @property
    def target_samples(self) -> int:
        return int(self.warmup_seconds * self.fps_hint)

    def is_ready(self) -> bool:
        return len(self.samples["ear"]) >= self.target_samples

    def progress(self) -> float:
        if self.target_samples == 0:
            return 1.0
        return min(1.0, len(self.samples["ear"]) / self.target_samples)

    def baseline(self) -> Dict[str, float]:
        baselines: Dict[str, float] = {}
        for key, values in self.samples.items():
            if values:
                baselines[key] = float(np.median(values))
            else:
                baselines[key] = 1.0
        return baselines

    def normalize(self, features: Dict[str, float]) -> Dict[str, float]:
        baseline = self.baseline()
        normalized = dict(features)
        normalized["baseline_ready"] = self.is_ready()
        normalized["baseline_progress"] = self.progress()
        for key in BASELINE_KEYS:
            base = baseline.get(key, 1.0)
            normalized[f"norm_{key}"] = float(features.get(key, 0.0)) / max(abs(base), 1e-6)
        return normalized
