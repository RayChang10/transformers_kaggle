# IMDB-BERT：使用 Transformers 與 PyTorch 的情感分析教學

本專案示範如何在 **Linux/WSL** 環境下，利用 Hugging Face Transformers 與 PyTorch 對 IMDB 電影評論做二元情感分類。資料集已預先放置於 `../data/IMDB Dataset.csv`。

## 專案結構

```
imdb-bert/
│── dataset.py        # 資料讀取與 Dataset 實作
│── model.py          # 模型建立與儲存
│── train.py          # 訓練腳本
│── predict.py        # 推論腳本（載入模型進行預測）
│── utils.py          # 共用工具：seed、split、metrics、logger
│── requirements.txt  # 依賴套件
└── README.md         # 本教學文件
```

## MVP 功能範圍

1. 讀取 IMDB CSV，完成文字清洗與標籤映射。
2. 透過 `IMDBDataset` 自動 tokenize，輸出供 BERT 使用的 tensors。
3. `train.py` 包含訓練、驗證 loop、指標計算與最佳模型儲存。
4. 模組化工具（seed、split、metrics、logger），方便後續擴充。

> 後續迭代可加入：測試集推論、早停、TensorBoard、超參數配置檔等。

## 安裝步驟（Linux/WSL Bash）

```bash
# 可視需要搭配 python -m venv 建立虛擬環境
cd /home/ray/transformers/imdb-bert
pip install -r requirements.txt
```

## 訓練指南

```bash
python train.py \
  --data-path ../data/IMDB\ Dataset.csv \
  --model-name bert-base-uncased \
  --tokenizer-name bert-base-uncased \
  --batch-size 8 \
  --num-epochs 2 \
  --output-dir ./checkpoints
```

- 若於 WSL 中使用 GPU，記得安裝相容的 CUDA/驅動並於 `pip install` 時拉取 `torch` 的 GPU 版本。
- `train.py` 會自動偵測 `cuda`，若無 GPU 則 fallback 至 `cpu`。

## 推論使用

訓練完成後，可以使用 `predict.py` 載入模型進行預測：

### 方法 1：直接指定文字
```bash
python predict.py \
  --model-dir ./checkpoints \
  --text "This movie is absolutely fantastic! I loved every minute of it."
```

### 方法 2：互動模式
```bash
python predict.py --model-dir ./checkpoints
```
進入互動模式後，輸入評論文字即可獲得預測結果。輸入 `quit` 或 `exit` 結束。

### 輸出範例
```
載入模型：./checkpoints

預測結果：positive
信心度：0.9856

輸入文字：This movie is absolutely fantastic! I loved every minute of it.
```

## 安全與最佳實踐

- 在 Dataset 端就限制最長序列長度與標籤檢查，避免不合法資料進入訓練流程。
- 設定 `max_grad_norm` 防止梯度爆炸，並保留早期返回/錯誤訊息以利除錯。
- 使用 `set_seed` 可重現結果；`requirements.txt` 鎖最小版本避免相依衝突。

## 下一步建議

- 增加測試集與推論腳本，支援 CLI 或 API 服務。
- 將訓練結果以 TensorBoard 或 Weights & Biases 紀錄。
- 撰寫單元測試（如 Dataset、metrics），確保重構時安全。 

