# 🚀 Quick Setup for Your Runpod Pod

## Your Pod Details
- **Pod Name**: greasy_lime_clownfish
- **Pod ID**: u2bn60prhjml75
- **IP**: 213.173.109.80
- **Available Ports**:
  - Port 8888: Jupyter Lab (HTTP) ✅
  - Port 19123: Web Terminal ✅

## 🎯 Fastest Setup Method (No SSH!)

### Method 1: Using Jupyter Lab (Easiest)

1. **Generate the bundle**:
   ```bash
   python upload_via_jupyter.py
   ```
   This creates `jupyter_setup_bundle.py`

2. **Open Jupyter Lab**:
   - Click the Jupyter Lab link in your Runpod dashboard (port 8888)
   - Or go to: `http://greasy-lime-clownfish.runpod.io:8888`

3. **In Jupyter Lab**:
   - Create a new Python notebook
   - Upload or paste the content of `jupyter_setup_bundle.py`
   - Run the cell - it will create all server files!

4. **Install dependencies** (in Jupyter):
   ```python
   !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   !pip install fastapi uvicorn python-multipart aiohttp pdf2image PyMuPDF Pillow pydantic pyyaml transformers accelerate huggingface-hub
   ```

5. **Download model** (in Jupyter):
   ```python
   !huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR
   ```

6. **Start server** (in Jupyter):
   ```python
   import os
   os.environ['MODEL_PATH'] = '/workspace/models/deepseek-ai/DeepSeek-OCR'
   os.environ['PORT'] = '8888'

   # Run in background
   !cd /workspace/deepseek-ocr-server && nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8888 > server.log 2>&1 &
   ```

### Method 2: Using Web Terminal

1. **Open Web Terminal**:
   - Click "Open Web Terminal" in Runpod dashboard
   - Or go to port 19123

2. **Run setup commands**:
   Copy and paste from `WEB_TERMINAL_COMMANDS.md`

3. **Upload files via Jupyter**:
   Since Web Terminal doesn't support file upload:
   - Open Jupyter Lab (port 8888)
   - Upload the `server/app/` folder
   - Move files to `/workspace/deepseek-ocr-server/app/`

## 📱 Using Your Go Client

1. **Build the client**:
   ```bash
   cd client
   go build -o deepseek-client.exe
   ```

2. **Use the Runpod config**:
   ```bash
   # Using the runpod config file
   ./deepseek-client -workers 4 -scan E:/PDFs -config config.runpod.yaml

   # Or specify the URL directly
   ./deepseek-client -workers 4 -scan E:/PDFs -api http://greasy-lime-clownfish.runpod.io:8888
   ```

## 🧪 Testing Your Setup

### Test from Web Terminal:
```bash
curl http://localhost:8888/health
```

### Test from your local machine:
```bash
curl http://greasy-lime-clownfish.runpod.io:8888/health
```

### Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_memory_free": "23.5GB"
}
```

## ⚡ Important Notes

1. **Port 8888**: We're using the Jupyter port since it's already exposed. The server will run on 8888 instead of 8000.

2. **Model Download**: The model is 8-10GB. Download will take 10-20 minutes depending on connection.

3. **GPU Memory**: With RTX 4090, you have 24GB VRAM. The model uses about 8-10GB, leaving plenty for processing.

4. **Background Server**: Use `nohup` or `tmux` to keep the server running when you close the terminal:
   ```bash
   # Using nohup
   nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8888 &

   # Using tmux
   tmux new -s deepseek
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8888
   # Press Ctrl+B, then D to detach
   ```

## 🎯 Complete Setup in 5 Minutes

### Jupyter Notebook - Copy & Run This:
```python
# Cell 1: Install packages
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install fastapi uvicorn python-multipart aiohttp pdf2image PyMuPDF Pillow pydantic pyyaml transformers accelerate huggingface-hub

# Cell 2: Run the jupyter_setup_bundle.py content
# [Paste the bundled script here]

# Cell 3: Download model
!huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR

# Cell 4: Start server
import os
os.chdir('/workspace/deepseek-ocr-server')
os.environ['MODEL_PATH'] = '/workspace/models/deepseek-ai/DeepSeek-OCR'
os.environ['PORT'] = '8888'

!nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8888 > server.log 2>&1 &

# Cell 5: Check status
!sleep 30  # Wait for server to start
!curl http://localhost:8888/health
```

## 📞 Your API Endpoint

Once running, your API is available at:
```
http://greasy-lime-clownfish.runpod.io:8888
```

Test it with:
```bash
# From your local machine
curl http://greasy-lime-clownfish.runpod.io:8888/health
```

## 🔧 Troubleshooting

### If server doesn't start:
1. Check GPU: Run `!nvidia-smi` in Jupyter
2. Check logs: `!cat /workspace/deepseek-ocr-server/server.log`
3. Verify model downloaded: `!ls -la /workspace/models/deepseek-ai/DeepSeek-OCR`

### If client can't connect:
1. Verify server is running: `!ps aux | grep uvicorn`
2. Check firewall: Port 8888 should be open (it's the Jupyter port)
3. Try the direct IP: `http://213.173.109.80:8888/health`

That's it! No SSH needed, everything through Jupyter Lab or Web Terminal. 🎉