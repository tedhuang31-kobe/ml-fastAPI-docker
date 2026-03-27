# 1. 使用官方 Python 輕量版作為底層環境
FROM python:3.9-slim

# 2. 設定容器內的工作目錄為 /app
WORKDIR /app

# 3. 先複製 requirements.txt 進去安裝套件
# 這樣做的好處是：如果程式碼改了但套件沒改，Docker 會跳過這步，速度更快
COPY requirements.txt .

# 4. 安裝執行環境所需的工具
RUN pip install --no-cache-dir -r requirements.txt

# 5. 把目前資料夾下所有的程式碼與訓練好的模型 (models/) 複製進去容器
COPY . .

# 6. 告訴 Docker，這台虛擬機器要開放 8000 端口
EXPOSE 8000

# 7. 啟動指令：執行 app.py 來啟動 FastAPI 服務
CMD ["python", "app.py"]