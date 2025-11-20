from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer

from model import load_model
from utils import get_device


def predict_sentiment(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 256,
) -> tuple[str, float]:
    """
    預測單一文字的情感。
    
    Returns:
        (label, confidence): 例如 ("positive", 0.95)
    """
    model.eval()
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = logits.argmax(dim=-1).item()
        confidence = probs[0][pred_id].item()
    
    label = model.config.id2label[pred_id]
    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="IMDB 情感分析推論")
    parser.add_argument("--model-dir", type=str, default="./checkpoints")
    parser.add_argument("--text", type=str, help="要預測的文字")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    
    device = get_device()
    
    # 載入模型
    print(f"載入模型：{args.model_dir}")
    model = load_model(args.model_dir)
    model.to(device)
    
    # 載入 tokenizer（優先從 checkpoints 載入，否則從模型名稱載入）
    tokenizer_path = os.path.join(args.model_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_path):
        # 如果 checkpoints 中有 tokenizer，直接載入
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        print("從 checkpoints 載入 tokenizer")
    else:
        # 如果沒有，從模型配置中取得原始模型名稱，或使用預設值
        tokenizer_name = (
            getattr(model.config, 'name_or_path', None) or 
            getattr(model.config, '_name_or_path', None) or 
            "bert-base-uncased"
        )
        print(f"從原始模型載入 tokenizer：{tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 如果有提供文字，直接預測
    if args.text:
        label, confidence = predict_sentiment(model, tokenizer, args.text, device, args.max_length)
        print(f"\n預測結果：{label}")
        print(f"信心度：{confidence:.4f}")
        print(f"\n輸入文字：{args.text}")
    else:
        # 互動模式
        print("\n進入互動模式（輸入 'quit' 結束）")
        while True:
            text = input("\n請輸入評論：")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if not text.strip():
                continue
            
            label, confidence = predict_sentiment(model, tokenizer, text, device, args.max_length)
            print(f"預測：{label} (信心度: {confidence:.4f})")


if __name__ == "__main__":
    main()
