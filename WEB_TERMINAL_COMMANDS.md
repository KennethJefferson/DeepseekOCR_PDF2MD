# Web Terminal Commands for Your Runpod Pod

## Quick Copy-Paste Commands for Web Terminal

Open your Web Terminal at port 19123 and run these commands in sequence:

### 1. Initial Setup (Run Once)
```bash
# Update system
apt-get update && apt-get install -y python3-pip python3-venv poppler-utils git wget

# Create virtual environment
cd /workspace
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install DeepSeek-OCR Dependencies
```bash
# Activate virtual environment
source /workspace/venv/bin/activate

# Install required packages
pip install fastapi uvicorn python-multipart aiohttp
pip install pdf2image PyMuPDF Pillow
pip install pydantic pyyaml python-dotenv
pip install transformers accelerate huggingface-hub

# Try to install vLLM (optional, may fail on some GPUs)
pip install vllm || echo "vLLM not available, using transformers"
```

### 3. Create Server Directory Structure
```bash
# Create directories
mkdir -p /workspace/deepseek-ocr-server/app
mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/temp

# Create __init__.py
cat > /workspace/deepseek-ocr-server/app/__init__.py << 'EOF'
"""DeepSeek-OCR Server"""
__version__ = "1.0.0"
EOF
```

### 4. Download Model (8-10GB, ~15 minutes)
```bash
# Activate environment
source /workspace/venv/bin/activate

# Download model
huggingface-cli download deepseek-ai/DeepSeek-OCR \
    --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR \
    --local-dir-use-symlinks False
```

### 5. Quick Test Script
```bash
# Create a test script to verify GPU
cat > /workspace/test_gpu.py << 'EOF'
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF

python /workspace/test_gpu.py
```

### 6. Upload Server Files

Since Web Terminal doesn't support direct file upload, use **Jupyter Lab** (port 8888):

1. Open Jupyter Lab from your Runpod dashboard
2. Upload the `server/app/` folder files:
   - `main.py`
   - `models.py`
   - `ocr_service.py`
   - `pdf_processor.py`
3. Move them to `/workspace/deepseek-ocr-server/app/`

Or use the manual creation commands below:

### 7. Manual File Creation (Alternative)

If you can't upload files, create them manually:

```bash
# Create main.py (copy from the server/app/main.py file)
cat > /workspace/deepseek-ocr-server/app/main.py << 'EOF'
# [Paste the content of server/app/main.py here]
EOF

# Create models.py
cat > /workspace/deepseek-ocr-server/app/models.py << 'EOF'
# [Paste the content of server/app/models.py here]
EOF

# Continue for other files...
```

### 8. Start Server on Port 8888
```bash
# Use port 8888 since it's already exposed
cd /workspace/deepseek-ocr-server

# Set environment variables
export MODEL_PATH=/workspace/models/deepseek-ai/DeepSeek-OCR
export GPU_MEMORY_UTILIZATION=0.85
export PORT=8888  # Using the Jupyter port

# Activate environment and start server
source /workspace/venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8888
```

### 9. Test the Server
```bash
# In a new terminal or after backgrounding the server (Ctrl+Z, then 'bg')
curl http://localhost:8888/health
```

## Your Server URL

Once running, your server will be accessible at:
```
http://greasy-lime-clownfish.runpod.io:8888
```

Update your Go client config:
```yaml
api:
  url: "http://greasy-lime-clownfish.runpod.io:8888"
```

## Troubleshooting

### If vLLM fails to install:
The server will automatically fall back to transformers backend. This is normal and will work fine.

### If CUDA is not detected:
```bash
nvidia-smi  # Check GPU status
```

### If model download fails:
```bash
# Use wget as alternative
cd /workspace/models
wget https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/[model-files]
```

### To run server in background:
```bash
# Using nohup
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8888 > /workspace/logs/server.log 2>&1 &

# Or using tmux
tmux new -s deepseek
python -m uvicorn app.main:app --host 0.0.0.0 --port 8888
# Press Ctrl+B then D to detach
# tmux attach -t deepseek  # to reattach
```

## Quick Health Check

After starting the server, verify it's working:
```bash
# Check locally
curl http://localhost:8888/health

# Check API info
curl http://localhost:8888/
```

You should see:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_memory_free": "XX.XGB"
}
```