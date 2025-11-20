from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_logger(log_level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_level,
    )
    return logging.getLogger("imdb-bert")


def train_val_split(
    dataframe: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = int(len(shuffled) * val_ratio)
    val_df = shuffled.iloc[:val_size].reset_index(drop=True)
    train_df = shuffled.iloc[val_size:].reset_index(drop=True)
    return train_df, val_df


def compute_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    preds = predictions.detach().cpu().numpy()
    labs = labels.detach().cpu().numpy()
    pred_ids = preds.argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labs, pred_ids)),
        "f1": float(f1_score(labs, pred_ids)),
    }

