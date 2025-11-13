# Manual Runpod Deployment Guide (No Docker)

This guide provides step-by-step instructions for manually deploying the DeepSeek-OCR server on Runpod without Docker.

**Time Required:** ~50-60 minutes
**Port:** 7777
**Backend:** vLLM (with transformers fallback)
**GPU:** RTX 4090 (24GB)

---

## Prerequisites

- Runpod pod with GPU (RTX 4090)
- Web Terminal or Jupyter Lab access
- 50GB+ free disk space
- Port 7777 exposed in Runpod settings

---

## Step 1: Connect to Your Pod (2 minutes)

### Option A: Web Terminal (Recommended)
1. In Runpod dashboard, click your pod
2. Click "Connect" → "Web Terminal"
3. Terminal opens in new tab

### Option B: Jupyter Lab
1. Access Jupyter at `http://[your-pod-name]-8888.proxy.runpod.net`
2. Open terminal: File → New → Terminal

---

## Step 2: Create Directory Structure (2 minutes)

Run these commands in the terminal:

```bash
# Navigate to workspace
cd /workspace

# Create all required directories
mkdir -p /workspace/deepseek-ocr-server/app
mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/temp

# Verify creation
ls -la /workspace/
```

**Expected output:** 4 directories created

---

## Step 3: Upload Server Files (10 minutes)

You need to upload 5 Python files to `/workspace/deepseek-ocr-server/app/`

### Option A: Via Jupyter Lab Upload

1. Open Jupyter Lab file browser
2. Navigate to `/workspace/deepseek-ocr-server/app/`
3. Click Upload button
4. Select these 5 files from your local `server/app/` directory:
   - `main.py` (contains ASCII banner)
   - `ocr_service.py` (vLLM backend)
   - `pdf_processor.py`
   - `models.py`
   - `__init__.py`
5. Click "Upload" for each file

### Option B: Copy-Paste via Terminal

Use the commands from `COMPLETE_TERMINAL_SETUP.md` to create files directly:

**File 1: main.py**
```bash
cat > /workspace/deepseek-ocr-server/app/main.py << 'EOF'
[Copy content from server/app/main.py]
EOF
```

**File 2: models.py**
```bash
cat > /workspace/deepseek-ocr-server/app/models.py << 'EOF'
[Copy content from server/app/models.py]
EOF
```

**File 3: ocr_service.py**
```bash
cat > /workspace/deepseek-ocr-server/app/ocr_service.py << 'EOF'
[Copy content from server/app/ocr_service.py]
EOF
```

**File 4: pdf_processor.py**
```bash
cat > /workspace/deepseek-ocr-server/app/pdf_processor.py << 'EOF'
[Copy content from server/app/pdf_processor.py]
EOF
```

**File 5: __init__.py**
```bash
cat > /workspace/deepseek-ocr-server/app/__init__.py << 'EOF'
"""DeepSeek-OCR Server"""
__version__ = "1.0.0"
EOF
```

### Verify Files Were Uploaded

```bash
ls -la /workspace/deepseek-ocr-server/app/
```

**Expected:** 5 Python files with total size ~50KB

---

## Step 4: Install Dependencies (15 minutes)

Run these commands **in order**:

### 4.1 System Dependencies

```bash
# Update package list
apt-get update

# Install PDF processing tools
apt-get install -y poppler-utils

# Install other utilities
apt-get install -y git curl wget
```

### 4.2 Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (for fast inference)
pip install vllm==0.8.5

# If vLLM fails, try alternative:
# pip install vllm --no-build-isolation

# Install FastAPI and web server
pip install fastapi uvicorn[standard] python-multipart aiohttp aiofiles

# Install PDF processing libraries
pip install pdf2image PyMuPDF Pillow

# Install ML libraries
pip install transformers accelerate huggingface-hub pydantic pydantic-settings pyyaml

# Install fast download support
pip install hf-transfer

# Install missing dependencies (if needed)
pip install addict matplotlib einops timm
```

### 4.3 Verify Installation

```bash
# Check CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check vLLM
python3 -c "import vllm; print('vLLM installed successfully')"

# Check FastAPI
python3 -c "import fastapi; print('FastAPI installed successfully')"
```

**Expected:** All should print successfully with `CUDA available: True`

---

## Step 5: Download DeepSeek-OCR Model (15-20 minutes)

This step downloads ~8-10GB of model files.

```bash
# Enable fast downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download the model
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR

# Verify download completed
du -sh /workspace/models/deepseek-ai/DeepSeek-OCR

# List model files
ls -la /workspace/models/deepseek-ai/DeepSeek-OCR/
```

**Expected:**
- Directory size: ~8-10GB
- Files: config.json, generation_config.json, model*.safetensors, tokenizer files

---

## Step 6: Create Helper Scripts (3 minutes)

### 6.1 Create Start Script

```bash
cat > /workspace/start_server.sh << 'EOF'
#!/bin/bash

# Navigate to server directory
cd /workspace/deepseek-ocr-server || {
    echo "Error: Server directory not found at /workspace/deepseek-ocr-server"
    exit 1
}

# Set environment variables
export MODEL_PATH="/workspace/models/deepseek-ai/DeepSeek-OCR"
export PORT=7777
export HOST="0.0.0.0"
export GPU_MEMORY_UTILIZATION=0.85
export CUDA_VISIBLE_DEVICES=0

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please run: huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir $MODEL_PATH"
    exit 1
fi

# Clear screen and show banner
clear
echo ""
echo "============================================================"
echo "    Starting DEEPSEEK-OCR SERVER"
echo "    Port: 7777 | Backend: vLLM | Mode: Foreground"
echo "============================================================"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
echo "Model Path: $MODEL_PATH"
echo ""
echo "Loading model... This may take 30-60 seconds"
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================================"
echo ""

# Start the server
python3 -m uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info \
    --access-log
EOF

chmod +x /workspace/start_server.sh
```

### 6.2 Create Check Script

```bash
cat > /workspace/check_server.sh << 'EOF'
#!/bin/bash

echo ""
echo "============================================================"
echo "    DEEPSEEK-OCR SERVER STATUS CHECK"
echo "============================================================"
echo ""

# Check process
echo "1. Process Status:"
if pgrep -f "uvicorn.*7777" > /dev/null; then
    echo "   ✓ Server is running (PID: $(pgrep -f 'uvicorn.*7777'))"
else
    echo "   ✗ Server not running"
fi
echo ""

# Check port
echo "2. Port Status:"
if netstat -tuln 2>/dev/null | grep -q ":7777 "; then
    echo "   ✓ Port 7777 is listening"
else
    echo "   ✗ Port 7777 not active"
fi
echo ""

# Check GPU
echo "3. GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo ""

# Check health endpoint
echo "4. Health Check:"
if curl -s -m 5 http://localhost:7777/health > /dev/null 2>&1; then
    echo "   ✓ Server is responding"
    echo ""
    echo "   Health Response:"
    curl -s http://localhost:7777/health | python3 -m json.tool | head -20
else
    echo "   ✗ Server not responding on http://localhost:7777"
    echo "   (Server may still be starting up - takes 30-60 seconds)"
fi
echo ""
echo "============================================================"
EOF

chmod +x /workspace/check_server.sh
```

### 6.3 Create Stop Script

```bash
cat > /workspace/stop_server.sh << 'EOF'
#!/bin/bash

echo ""
echo "Stopping DeepSeek-OCR server..."

# Kill uvicorn processes
pkill -f "uvicorn.*7777" && echo "✓ Server process stopped" || echo "No server process found"

# Free the port
fuser -k 7777/tcp 2>/dev/null && echo "✓ Port 7777 freed" || echo "Port 7777 already free"

echo "Server stopped"
echo ""
EOF

chmod +x /workspace/stop_server.sh
```

### Verify Scripts Created

```bash
ls -la /workspace/*.sh
```

**Expected:** 3 executable scripts (start_server.sh, check_server.sh, stop_server.sh)

---

## Step 7: Start the Server (2 minutes)

```bash
# Run the start script
bash /workspace/start_server.sh
```

**Expected Output:**

```
============================================================
    Starting DEEPSEEK-OCR SERVER
    Port: 7777 | Backend: vLLM | Mode: Foreground
============================================================

GPU Information:
NVIDIA GeForce RTX 4090, 24564 MiB, 24090 MiB

Model Path: /workspace/models/deepseek-ai/DeepSeek-OCR

Loading model... This may take 30-60 seconds
Press Ctrl+C to stop the server

============================================================

INFO:     Started server process [12345]
INFO:     Waiting for application startup.

============================================================
    DEEPSEEK-OCR SERVER
    PDF to Markdown Conversion
    Port: 7777 | Backend: vLLM
============================================================

INFO:     Loading model from: /workspace/models/deepseek-ai/DeepSeek-OCR
INFO:     Loading model with vLLM...
INFO:     Model loaded with vLLM backend
INFO:     Server startup complete!

✓ Server ready at http://0.0.0.0:7777

INFO:     Uvicorn running on http://0.0.0.0:7777 (Press CTRL+C to quit)
```

**Note:** Model loading takes 30-60 seconds. Wait for "Server ready" message.

---

## Step 8: Test the Server (5 minutes)

Open a **NEW terminal** (keep the server running in the first terminal):

### 8.1 Local Tests

```bash
# Test health endpoint
curl http://localhost:7777/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_memory_free": "14.5GB",
  "gpu_memory_total": "24.0GB",
  "model_path": "/workspace/models/deepseek-ai/DeepSeek-OCR",
  "backend": "vLLM",
  "uptime_seconds": 42.5
}
```

```bash
# Test status endpoint
curl http://localhost:7777/api/v1/status

# Test root endpoint
curl http://localhost:7777/
```

### 8.2 Test with Sample Image

```bash
# Download test image
wget https://raw.githubusercontent.com/tesseract-ocr/docs/main/images/eurotext.png -O /workspace/test.png

# Test OCR (if endpoint exists)
curl -X POST http://localhost:7777/api/v1/ocr/pdf \
  -F "file=@/workspace/test.png" \
  -F "resolution=base"
```

---

## Step 9: Test External Access (2 minutes)

### 9.1 Find Your External URL

In Runpod dashboard, your external URL will be:
```
http://[your-pod-name]-7777.proxy.runpod.net
```

Example: `http://stale-lavender-junglefowl-7777.proxy.runpod.net`

### 9.2 Test from Local Machine

From your Windows machine (PowerShell or Command Prompt):

```bash
# Test health endpoint
curl http://[your-pod-name]-7777.proxy.runpod.net/health

# Or use PowerShell
Invoke-WebRequest -Uri "http://[your-pod-name]-7777.proxy.runpod.net/health"
```

### 9.3 Configure Go Client

Update your client configuration:

```yaml
# config.yaml
api:
  url: "http://[your-pod-name]-7777.proxy.runpod.net"
  timeout: 300
```

Test with Go client:

```bash
./deepseek-client -workers 4 -scan /path/to/pdfs -api http://[your-pod-name]-7777.proxy.runpod.net
```

---

## Quick Command Reference

```bash
# Start server
bash /workspace/start_server.sh

# Check status
bash /workspace/check_server.sh

# Stop server
bash /workspace/stop_server.sh

# View GPU usage
nvidia-smi

# Check logs (if running in background)
tail -f /workspace/logs/server.log

# Kill stuck processes
pkill -f uvicorn
fuser -k 7777/tcp
```

---

## Troubleshooting

### Server Won't Start

```bash
# Check Python packages
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import vllm; print('vLLM OK')"
python3 -c "import fastapi; print('FastAPI OK')"

# Check model exists
ls -la /workspace/models/deepseek-ai/DeepSeek-OCR/

# Check for port conflicts
netstat -tuln | grep 7777
```

### vLLM Not Working

If vLLM fails, the server will automatically fall back to transformers. To force transformers:

```bash
# Edit ocr_service.py to skip vLLM
# Or set environment variable
export USE_VLLM=false
```

### Model Download Issues

```bash
# If download is incomplete
rm -rf /workspace/models/deepseek-ai/DeepSeek-OCR

# Try without fast transfer
export HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR
```

### Out of Memory

Reduce GPU memory usage:

```bash
# Edit start_server.sh
export GPU_MEMORY_UTILIZATION=0.70  # Instead of 0.85
```

---

## Verification Checklist

- [ ] All 5 Python files uploaded to `/workspace/deepseek-ocr-server/app/`
- [ ] Dependencies installed (torch, vllm, fastapi)
- [ ] Model downloaded (~8-10GB in `/workspace/models/`)
- [ ] Helper scripts created (start, stop, check)
- [ ] Server starts without errors
- [ ] Health endpoint responds with `"status": "healthy"`
- [ ] ASCII banner shows on startup
- [ ] Port 7777 is listening
- [ ] External access works

---

## Summary

You now have DeepSeek-OCR server running on Runpod:
- ✅ Port 7777 exposed
- ✅ vLLM backend for fast inference
- ✅ ASCII banner on startup
- ✅ Manual Python deployment (no Docker)
- ✅ RTX 4090 optimized (~10GB VRAM usage)

The server is ready to process PDFs to Markdown!