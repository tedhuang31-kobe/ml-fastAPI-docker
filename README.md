# Cloud-Based Regression API: XGBoost & Neural Network

這是一個完整的機器學習端到端專案。透過 `FastAPI` 封裝了兩套不同的回歸模型（XGBoost 與 PyTorch 神經網路），並實現了完整的一鍵式 `Docker` 容器化部署。

## 🌟 專案亮點
- **雙模型預測**：同時支援經典的 `XGBoost` 與深度的 `Neural Network` (PyTorch) 進行數值預測。
- **特徵持久化**：使用 `RobustScaler` 進行特徵標準化，並確保訓練與預測環境的特徵縮放一致。
- **生產級封裝**：使用 `FastAPI` 提供非同步 (Asynchronous) 的 RESTful API 服務。
- **雲端就緒**：包含 `Dockerfile`，可直接部署至 AWS App Runner、ECS 或 Kubernetes。

## 📂 專案結構
```text
.
├── app.py              # FastAPI 服務主程式
├── Train.py            # 模型訓練與特徵工程腳本
├── Dockerfile          # 容器鏡像構建設定
├── requirements.txt    # 專案依賴套件 (XGBoost, torch, fastapi...)
├── .gitignore          # 排除大檔案與暫存檔 (data/, __pycache__)
├── models/             # [持久化目錄] 存放 .pkl 與 .pth 模型檔
└── data/               # [資料目錄] 存放原始數據 (rawdataset.xlsx)