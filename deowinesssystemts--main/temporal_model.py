from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


FEATURE_COLUMNS = [
    "ear",
    "mar",
    "pitch",
    "roll",
    "yaw",
    "blink_count",
    "yawn_count",
    "norm_ear",
    "norm_mar",
    "norm_pitch",
    "norm_roll",
    "norm_yaw",
    "eye_closed_frames",
]
LABELS = ["alert", "low", "medium", "high"]


class LSTMSeverityModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, layers: int = 2, classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


class RealtimeTemporalPredictor:
    def __init__(self, model_path: str | Path, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.buffer: Deque[List[float]] = deque(maxlen=sequence_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = LSTMSeverityModel(
            input_size=len(checkpoint["feature_columns"]),
            hidden_size=checkpoint["hidden_size"],
            layers=checkpoint["layers"],
            classes=len(checkpoint["labels"]),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.feature_columns = checkpoint["feature_columns"]
        self.labels = checkpoint["labels"]

    def push(self, features: Dict[str, float]) -> Tuple[str | None, float]:
        row = [float(features.get(column, 0.0)) for column in self.feature_columns]
        self.buffer.append(row)
        if len(self.buffer) < self.sequence_length:
            return None, 0.0
        tensor = torch.tensor([list(self.buffer)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        index = int(np.argmax(probs))
        return self.labels[index], float(probs[index])
