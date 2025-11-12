# Terminal-Only Setup Commands for DeepSeek-OCR on Runpod

This guide provides step-by-step terminal commands to set up and run the DeepSeek-OCR server on Runpod without using Jupyter notebooks.

## Prerequisites

- Runpod pod with GPU (RTX 4090 or A100 recommended)
- Port 7777 exposed in Runpod settings
- 50GB+ available disk space
- Server files already uploaded to `/workspace/deepseek-ocr-server/`

---

## Part 1: Quick Setup Commands (If Starting Fresh)

If you're starting completely fresh, run these commands to upload the server files:

```bash
# Step 1: Navigate to workspace
cd /workspace

# Step 2: Create server directory
mkdir -p deepseek-ocr-server/app

# Step 3: Create the server files
# You'll need to copy the Python files from server/app/*.py
# Use the upload_via_jupyter.py script or manual upload
```

---

## Part 2: Main Setup (Since You're at Step 5)

Since you already have the server files uploaded from steps 1-4, continue with these commands:

### Step 5: Run the Complete Setup Script

Copy and paste this entire command to download and run the setup script:

```bash
cd /workspace/deepseek-ocr-server

# Download the setup script (or create it if not present)
cat > runpod_terminal_setup.sh << 'SETUP_EOF'
#!/bin/bash
# [The full runpod_terminal_setup.sh content will be here]
# This is a placeholder - use the actual script content
SETUP_EOF

# Make it executable and run
chmod +x runpod_terminal_setup.sh
bash runpod_terminal_setup.sh
```

**OR** if the setup script is already present:

```bash
cd /workspace/deepseek-ocr-server
bash runpod_terminal_setup.sh
```

**Note:** The setup script will:
- Install all Python dependencies (~5-10 minutes)
- Download the DeepSeek-OCR model (~10GB, 5-15 minutes)
- Create helper scripts
- Set up the environment

---

## Part 3: Manual Setup (Alternative to Setup Script)

If you prefer to run commands manually instead of the setup script:

### Step 5A: Install System Dependencies

```bash
# Update packages
apt-get update

# Install PDF processing tools
apt-get install -y poppler-utils

# Install utilities
apt-get install -y git curl wget
```

### Step 5B: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (optimized inference)
pip install vllm

# Install FastAPI and server dependencies
pip install fastapi uvicorn python-multipart aiofiles

# Install PDF processing
pip install pdf2image PyMuPDF Pillow

# Install ML libraries
pip install transformers accelerate huggingface-hub

# Install fast downloads
pip install hf-transfer
```

### Step 5C: Create Directories

```bash
# Create necessary directories
mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/temp
```

### Step 6: Download the Model

```bash
# Enable fast downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download DeepSeek-OCR model (~10GB)
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR
```

**Note:** This will take 5-15 minutes depending on your connection speed.

### Step 7: Start the Server

```bash
# Navigate to server directory
cd /workspace/deepseek-ocr-server

# Set environment variables
export MODEL_PATH="/workspace/models/deepseek-ai/DeepSeek-OCR"
export PORT=7777
export HOST="0.0.0.0"
export GPU_MEMORY_UTILIZATION=0.85

# Start the server (runs in foreground, shows logs)
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 7777 --log-level info
```

**Expected output:**
```
============================================================
    DEEPSEEK-OCR SERVER
    PDF to Markdown Conversion
    Port: 7777 | Backend: vLLM
============================================================

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Loading model from: /workspace/models/deepseek-ai/DeepSeek-OCR
...
✓ Server ready at http://0.0.0.0:7777
```

**Note:**
- The server will take 30-60 seconds to load the model
- Keep this terminal open to see logs
- Press `Ctrl+C` to stop the server

---

## Part 4: Testing the Server

### Step 8: Open a New Terminal

In Runpod, open a second terminal window/tab to run tests while the server is running.

### Step 9: Test Health Endpoint

```bash
# Check if server is healthy
curl http://localhost:7777/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_memory_free": "20.5GB",
  "gpu_memory_total": "24.0GB"
}
```

### Step 10: Test with a Sample PDF

If you have a test PDF file:

```bash
# Upload a test PDF (replace test.pdf with your file)
curl -X POST \
  -F "file=@test.pdf" \
  -F "resolution=base" \
  http://localhost:7777/api/v1/ocr/pdf
```

---

## Part 5: Using the Helper Scripts

After running the setup script, you'll have these helper scripts available:

### Start Server (Recommended Method)

```bash
bash /workspace/start_server.sh
```

### Check Server Status

```bash
bash /workspace/check_server.sh
```

### Stop Server

```bash
bash /workspace/stop_server.sh
```

---

## Part 6: Connecting from Your Local Machine

### Find Your Runpod URL

1. In Runpod dashboard, find your pod
2. Look for the connection URL (e.g., `greasy-lime-clownfish.runpod.io`)
3. Make sure port 7777 is exposed

### Test from Local Machine

```bash
# Replace with your actual Runpod URL
curl http://greasy-lime-clownfish.runpod.io:7777/health
```

### Use with Go Client

```bash
# On your local machine
cd client
./deepseek-client -workers 4 \
  -scan /path/to/pdfs \
  -api http://greasy-lime-clownfish.runpod.io:7777
```

---

## Troubleshooting

### If Model Download Fails

```bash
# Retry with standard download (slower but more reliable)
export HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR
```

### If vLLM Installation Fails

```bash
# Use transformers backend instead
pip install transformers accelerate
# The server will automatically fallback to transformers
```

### If Port 7777 is Blocked

```bash
# Use a different port (e.g., 8000)
export PORT=8000
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Check GPU Memory

```bash
nvidia-smi
```

### View Server Logs

```bash
# If running in background
tail -f /workspace/logs/deepseek-ocr.log

# If running in foreground
# Logs appear directly in terminal
```

### Server Won't Start

```bash
# Check if another instance is running
pkill -f "uvicorn.*7777"

# Check if model exists
ls -la /workspace/models/deepseek-ai/DeepSeek-OCR/

# Check Python packages
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import fastapi; import vllm"
```

---

## Summary of Key Commands

```bash
# One-time setup (takes ~20 minutes)
bash /workspace/deepseek-ocr-server/runpod_terminal_setup.sh

# Start server (do this each time)
bash /workspace/start_server.sh

# Check status
bash /workspace/check_server.sh

# Stop server
bash /workspace/stop_server.sh

# Test health
curl http://localhost:7777/health

# Process PDF
curl -X POST -F "file=@document.pdf" http://localhost:7777/api/v1/ocr/pdf
```

---

## Notes

- **First time setup**: Takes 15-30 minutes (mainly model download)
- **Server startup**: Takes 30-60 seconds (model loading)
- **Processing speed**: ~1-2 seconds per PDF page on RTX 4090
- **Port**: Using 7777 instead of default 8000 (remember to expose in Runpod)
- **Mode**: Server runs in foreground showing logs (Ctrl+C to stop)

---

## Quick Copy-Paste Commands

For your convenience, here are the essential commands to copy-paste:

```bash
# Complete setup (run once)
cd /workspace/deepseek-ocr-server && bash runpod_terminal_setup.sh

# Start server
cd /workspace && bash start_server.sh

# Test server
curl http://localhost:7777/health
```

That's it! The server should now be running on port 7777 with a simple text banner and vLLM backend.