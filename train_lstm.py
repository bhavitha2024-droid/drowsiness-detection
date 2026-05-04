from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from temporal_model import FEATURE_COLUMNS, LABELS, LSTMSeverityModel


LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


def build_sequences(df: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    required_columns = FEATURE_COLUMNS + ["label"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    feature_matrix = df[FEATURE_COLUMNS].fillna(0.0).astype(np.float32).values
    labels = df["label"].astype(str).str.lower().map(LABEL_TO_INDEX)
    if labels.isnull().any():
        unknown = sorted(df.loc[labels.isnull(), "label"].astype(str).unique().tolist())
        raise ValueError(f"Unknown labels found: {unknown}")

    sequence_list: List[np.ndarray] = []
    label_list: List[int] = []
    for index in range(sequence_length - 1, len(df)):
        sequence_list.append(feature_matrix[index - sequence_length + 1 : index + 1])
        label_list.append(int(labels.iloc[index]))

    if not sequence_list:
        raise ValueError("Not enough rows to build sequences. Collect more data first.")

    return np.stack(sequence_list), np.array(label_list, dtype=np.int64)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    predictions: List[int] = []
    targets: List[int] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            predictions.extend(preds)
            targets.extend(labels.numpy().tolist())

    return {
        "accuracy": accuracy_score(targets, predictions),
        "f1_weighted": f1_score(targets, predictions, average="weighted"),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
        "classification_report": classification_report(
            targets,
            predictions,
            target_names=LABELS,
            zero_division=0,
        ),
    }


def train(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    model_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    sequences, labels = build_sequences(df, args.sequence_length)
    x_train, x_val, y_train, y_val = train_test_split(
        sequences,
        labels,
        test_size=args.validation_split,
        random_state=42,
        stratify=labels,
    )

    train_dataset = SequenceDataset(x_train, y_train)
    val_dataset = SequenceDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSeverityModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=args.hidden_size,
        layers=args.layers,
        classes=len(LABELS),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for features, batch_labels in train_loader:
            features = features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"epoch={epoch} loss={avg_loss:.4f}")

    metrics = evaluate(model, val_loader, device)
    print(f"validation_accuracy={metrics['accuracy']:.4f}")
    print(f"validation_f1_weighted={metrics['f1_weighted']:.4f}")
    print("confusion_matrix=")
    print(np.array(metrics["confusion_matrix"]))
    print(metrics["classification_report"])

    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "labels": LABELS,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "sequence_length": args.sequence_length,
    }
    output_path = model_dir / "drowsiness_lstm.pt"
    torch.save(checkpoint, output_path)
    print(f"saved_model={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM for temporal drowsiness severity.")
    parser.add_argument("--csv", required=True, help="CSV file generated from the real-time logger.")
    parser.add_argument("--output-dir", default="models", help="Directory to save the trained model.")
    parser.add_argument("--sequence-length", type=int, default=30, help="Frames per training sequence.")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--layers", type=int, default=2, help="Number of stacked LSTM layers.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction reserved for validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
