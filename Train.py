import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- 定義神經網路架構 (Regression 專用) ---
class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128), # 加速收斂
            nn.ReLU(),
            nn.Dropout(0.2),    # 防止過擬合
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)    # 回歸任務：最後只輸出一個數值
        )
    def forward(self, x):
        return self.network(x)

def run_pipeline():
    # 建立相對路徑系統
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "rawdataset.xlsx")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    # 確保資料夾存在
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. 讀取資料
    print(f"正在從 {DATA_PATH} 讀取資料...")
    try:
        df = pd.read_excel(DATA_PATH)
    except FileNotFoundError:
        print("錯誤：找不到資料檔案，請確認檔案已放入 data/ 資料夾中。")
        return

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1] # Target 為連續數值 (Regressor)

    # 2. 特徵工程：RobustScaler (Persistence Point 1)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 切分數據 (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 3. 訓練 XGBoost Regressor (Persistence Point 2)
    print("正在訓練 XGBoost Regressor...")
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=6, objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    
    # 4. 訓練 Neural Network (Persistence Point 3)
    print("正在訓練 PyTorch Neural Network...")
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train.values).reshape(-1, 1)
    
    nn_model = RegressionNN(X_train.shape[1])
    criterion = nn.MSELoss() # 回歸專用損失函數
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

    for epoch in range(150): # 訓練 150 輪
        optimizer.zero_grad()
        outputs = nn_model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    # 5. 評估與存檔
    xgb_r2 = r2_score(y_test, xgb_model.predict(X_test))
    print(f"訓練完成！XGBoost R2 Score: {xgb_r2:.4f}")

    joblib.dump(scaler, os.path.join(MODEL_DIR, 'robust_scaler.pkl'))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    torch.save(nn_model.state_dict(), os.path.join(MODEL_DIR, 'nn_model.pth'))
    print(f"所有模型檔案已存入: {MODEL_DIR}")

if __name__ == "__main__":
    run_pipeline()