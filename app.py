from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import torch
import os
import uvicorn

# 必須重新定義 NN 結構以供加載
class RegressionNN(torch.nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)

app = FastAPI(title="Cloud Regressor API")

# --- 模型持久化載入 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'robust_scaler.pkl'))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    
    # 載入 NN 模型
    input_dim = xgb_model.n_features_in_
    nn_model = RegressionNN(input_dim)
    nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'nn_model.pth')))
    nn_model.eval()
    print("成功載入所有回歸模型！")
except Exception as e:
    print(f"載入失敗，請先運行 train.py。錯誤資訊: {e}")

@app.get("/")
def root():
    return {"message": "Regressor API is Online"}

@app.post("/predict/xgb")
def predict_xgb(data: dict):
    try:
        df_input = pd.DataFrame([data])
        X_scaled = scaler.transform(df_input) # 使用持久化的 scaler
        val = xgb_model.predict(X_scaled)
        return {"model": "XGBoost", "predicted_value": float(val[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/nn")
def predict_nn(data: dict):
    try:
        df_input = pd.DataFrame([data])
        X_scaled = scaler.transform(df_input)
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            val = nn_model(X_tensor)
        return {"model": "Neural Network", "predicted_value": float(val.item())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)