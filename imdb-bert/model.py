from __future__ import annotations

from pathlib import Path

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from dataset import ID2LABEL, LABEL2ID


def build_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
    dropout: float = 0.1,
) -> AutoModelForSequenceClassification:
    """
    建立並回傳 Hugging Face Transformers 的分類模型。
    """
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: ID2LABEL.get(i, str(i)) for i in range(num_labels)},
        label2id={k: LABEL2ID.get(k, i) for i, k in enumerate(ID2LABEL.values())},
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )
    return model


def save_model(
    model: AutoModelForSequenceClassification,
    output_dir: str,
    tokenizer_name: str | None = None,
) -> None:
    """
    儲存模型與 config，方便之後推論使用。
    如果提供 tokenizer_name，也會保存 tokenizer。
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    
    # 如果提供了 tokenizer_name，也保存 tokenizer
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(path)


def load_model(output_dir: str) -> AutoModelForSequenceClassification:
    """
    從既有輸出資料夾載入模型，常用於推論或繼續訓練。
    """
    path = Path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"找不到模型資料夾：{path}")
    return AutoModelForSequenceClassification.from_pretrained(path)

