from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from dataset import IMDBDataset, collate_samples, load_imdb_dataframe
from model import build_model, save_model
from utils import (
    compute_classification_metrics,
    create_logger,
    get_device,
    set_seed,
    train_val_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMDB sentiment classifier")
    parser.add_argument("--data-path", type=str, default="../data/IMDB_Dataset.csv")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = create_logger()
    device = get_device()

    logger.info("載入資料：%s", args.data_path)
    dataframe = load_imdb_dataframe(args.data_path)
    train_df, val_df = train_val_split(dataframe, val_ratio=args.val_ratio, seed=args.seed)

    # 加入這幾行，顯示切分結果
    logger.info("資料切分完成：訓練集 %d 筆，驗證集 %d 筆", len(train_df), len(val_df))

    train_dataset = IMDBDataset(
        train_df,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
    )
    val_dataset = IMDBDataset(
        val_df,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_samples,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
        num_workers=2,
    )

    model = build_model(
        model_name=args.model_name,
        num_labels=2,
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} / {args.num_epochs} - train"):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        logger.info("Epoch %d | train_loss %.4f", epoch, avg_train_loss)

        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d | val_accuracy %.4f | val_f1 %.4f",
            epoch,
            val_metrics["accuracy"],
            val_metrics["f1"],
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            logger.info("發現更佳模型，儲存至：%s", output_path)
            save_model(model, str(output_path), tokenizer_name=args.tokenizer_name)


@torch.no_grad()
def evaluate(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_labels = []
    for batch in tqdm(dataloader, desc="eval"):
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("labels")
        outputs = model(**batch)
        logits = outputs.logits
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    return compute_classification_metrics(logits_tensor, labels_tensor)


if __name__ == "__main__":
    train()

