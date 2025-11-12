# Runpod Docker Deployment Guide - DeepSeek-OCR API

Complete step-by-step guide to deploy the Dockerized DeepSeek-OCR API on your Runpod pod.

**Repository:** https://github.com/Bogdanovich77/DeekSeek-OCR---Dockerized-API

---

## Your Pod Information

- **Pod ID:** stale_lavender_junglefowl
- **GPU:** NVIDIA RTX 4090 (24GB VRAM)
- **Current Port:** 8888 (Jupyter)
- **Target Port:** 7777 (DeepSeek-OCR API)
- **Access Method:** Web Terminal

---

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Connected to Runpod Web Terminal
- [ ] RTX 4090 GPU available (check with `nvidia-smi`)
- [ ] At least 50GB free disk space
- [ ] Docker installed with GPU support

---

## Step 1: Pre-flight Verification (5 minutes)

Connect to your Runpod Web Terminal and run these checks:

```bash
# Check GPU is available
nvidia-smi
```

**Expected:** You should see your RTX 4090 with 24GB memory

```bash
# Check Docker installation
docker --version
```

**Expected:** Docker version 20.10 or higher

```bash
# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Expected:** nvidia-smi output from inside the container

```bash
# Check disk space (need 50GB+)
df -h /workspace
```

**Expected:** At least 50GB available

```bash
# Check Docker Compose
docker-compose --version
```

**Expected:** docker-compose version 2.0+

---

## Step 2: Clone Repository (2 minutes)

```bash
# Navigate to workspace
cd /workspace

# Clone the repository
git clone https://github.com/Bogdanovich77/DeekSeek-OCR---Dockerized-API.git

# Enter the directory
cd DeekSeek-OCR---Dockerized-API

# Verify files
ls -la
```

**Expected files:**
- `Dockerfile`
- `docker-compose.yml`
- `start_server.py`
- `README.md`
- Various Python scripts

```bash
# Create necessary directories
mkdir -p models outputs data

# Verify structure
tree -L 1
```

---

## Step 3: Download DeepSeek-OCR Model (15-30 minutes)

This is the longest step - downloads ~9GB of model files.

```bash
# Install Hugging Face Hub CLI
pip install -U huggingface-hub
```

```bash
# Download the model
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir models/deepseek-ai/DeepSeek-OCR \
  --local-dir-use-symlinks False
```

**Note:** This will take 15-30 minutes depending on your network speed. You'll see progress bars for each file being downloaded.

```bash
# Verify the download completed
ls -lh models/deepseek-ai/DeepSeek-OCR/
```

**Expected:** Multiple `.safetensors` files and config files, totaling ~9GB

```bash
# Check total size
du -sh models/deepseek-ai/DeepSeek-OCR/
```

**Expected:** Around 9-10GB

---

## Step 4: Configure for Port 7777 (2 minutes)

We need to modify the Docker Compose configuration to use port 7777 instead of 8000.

```bash
# Change port mapping from 8000 to 7777
sed -i 's/- "8000:8000"/- "7777:8000"/' docker-compose.yml
```

```bash
# Optimize GPU memory utilization for RTX 4090
sed -i 's/GPU_MEMORY_UTILIZATION=0.85/GPU_MEMORY_UTILIZATION=0.90/' docker-compose.yml
```

```bash
# Increase concurrency for better performance
sed -i 's/MAX_CONCURRENCY=5/MAX_CONCURRENCY=10/' docker-compose.yml
```

```bash
# Verify all changes
grep -E "(ports:|GPU_MEMORY_UTILIZATION|MAX_CONCURRENCY)" docker-compose.yml
```

**Expected output:**
```
    ports:
      - "7777:8000"
      - GPU_MEMORY_UTILIZATION=0.90
      - MAX_CONCURRENCY=10
```

---

## Step 5: Build Docker Image (10-15 minutes)

```bash
# Build the Docker image
docker-compose build
```

**What's happening:**
- Pulling base image vllm/vllm-openai:v0.8.5 (~3GB)
- Installing Python dependencies
- Copying custom files with bug fixes
- Configuring the environment

**Expected:** Build completes without errors. This takes 10-15 minutes.

```bash
# Verify the image was created
docker images | grep deepseek
```

**Expected:** You should see an image with a name containing "deepseek-ocr"

---

## Step 6: Launch the Container (4 minutes)

```bash
# Start the container in detached mode
docker-compose up -d
```

**What's happening:**
- Container starts in the background
- Mounts model directory
- Begins loading the model into GPU memory

```bash
# Monitor the startup logs
docker-compose logs -f deepseek-ocr
```

**Watch for these messages:**
- `Loading model from /app/models/deepseek-ai/DeepSeek-OCR`
- `CUDA available: True`
- `Model loaded successfully`
- `Uvicorn running on http://0.0.0.0:8000`

Press `Ctrl+C` to exit log viewing (container keeps running)

```bash
# Wait for model to fully load (takes ~2 minutes)
echo "Waiting for model to load..."
sleep 120
```

```bash
# Check container status
docker-compose ps
```

**Expected:** Container should be "Up" and healthy

---

## Step 7: Health Check and Verification (5 minutes)

```bash
# Test the health endpoint
curl http://localhost:7777/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "device_count": 1,
  "model_path": "/app/models/deepseek-ai/DeepSeek-OCR"
}
```

```bash
# Check GPU memory usage
nvidia-smi
```

**Expected:** You should see a Python process using ~9-10GB of VRAM

```bash
# Check container logs for any errors
docker-compose logs --tail=50 deepseek-ocr
```

**Expected:** No error messages, just startup logs

---

## Step 8: API Testing (10 minutes)

### Test 1: Download a test image

```bash
# Download a sample image
wget https://raw.githubusercontent.com/tesseract-ocr/docs/main/images/eurotext.png \
  -O test_image.png
```

### Test 2: Process the image

```bash
# Test the image OCR endpoint
curl -X POST "http://localhost:7777/ocr/image" \
  -F "file=@test_image.png" \
  -o response.json
```

```bash
# View the response
cat response.json | python -m json.tool
```

**Expected:** JSON response with `"success": true` and OCR text in the results

### Test 3: Check processing time

```bash
# Extract processing time from response
cat response.json | python -c "import json, sys; data=json.load(sys.stdin); print(f'Processing time: {data.get(\"processing_time_ms\", 0)}ms')"
```

**Expected:** Processing time should be 1000-3000ms for the first request (includes warmup)

---

## Step 9: Configure Runpod Port Exposure

**Important:** You need to expose port 7777 in Runpod's UI to access it externally.

### Option A: Use Runpod Web Interface

1. Go to your Runpod dashboard
2. Click on your pod "stale_lavender_junglefowl"
3. Find "Port Configuration" or "HTTP Services"
4. Add port **7777** with protocol **HTTP**
5. Save changes

Your external URL will be:
```
https://stale-lavender-junglefowl-7777.proxy.runpod.net
```

### Option B: If port 7777 cannot be exposed

Use port 8888 instead (already exposed for Jupyter):

```bash
# Stop the current container
docker-compose down

# Change port back to 8888
sed -i 's/- "7777:8000"/- "8888:8000"/' docker-compose.yml

# Restart
docker-compose up -d
```

Then your external URL will be:
```
https://stale-lavender-junglefowl-8888.proxy.runpod.net
```

---

## Step 10: Test External Access

From your **local machine** (not the Runpod terminal):

```bash
# Test health endpoint (replace with your actual URL)
curl https://stale-lavender-junglefowl-7777.proxy.runpod.net/health
```

**Expected:** Same health check response as before

```bash
# Test with an image from your local machine
curl -X POST "https://stale-lavender-junglefowl-7777.proxy.runpod.net/ocr/image" \
  -F "file=@/path/to/your/image.jpg" \
  | python -m json.tool
```

---

## Common Commands Reference

### Start the service
```bash
cd /workspace/DeekSeek-OCR---Dockerized-API
docker-compose up -d
```

### Stop the service
```bash
docker-compose down
```

### Restart the service
```bash
docker-compose restart deepseek-ocr
```

### View logs
```bash
docker-compose logs -f deepseek-ocr
```

### Check container status
```bash
docker-compose ps
```

### Check GPU usage
```bash
nvidia-smi
```

### Health check
```bash
curl http://localhost:7777/health
```

### Process an image
```bash
curl -X POST "http://localhost:7777/ocr/image" -F "file=@image.jpg"
```

### Process a PDF
```bash
curl -X POST "http://localhost:7777/ocr/pdf" -F "file=@document.pdf"
```

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs deepseek-ocr

# Common issues:
# - Model not downloaded: Check models/deepseek-ai/DeepSeek-OCR/
# - Port conflict: Try a different port
# - GPU not accessible: Check nvidia-smi

# Restart container
docker-compose down && docker-compose up -d
```

### Model loading failed

```bash
# Verify model files exist
ls -lh models/deepseek-ai/DeepSeek-OCR/

# Should see multiple .safetensors files

# If missing, re-download
rm -rf models/deepseek-ai/DeepSeek-OCR/
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir models/deepseek-ai/DeepSeek-OCR \
  --local-dir-use-symlinks False
```

### Out of memory errors

```bash
# Reduce GPU memory usage
sed -i 's/GPU_MEMORY_UTILIZATION=0.90/GPU_MEMORY_UTILIZATION=0.75/' docker-compose.yml

# Reduce concurrency
sed -i 's/MAX_CONCURRENCY=10/MAX_CONCURRENCY=5/' docker-compose.yml

# Restart
docker-compose down && docker-compose up -d
```

### Slow processing

```bash
# Check GPU utilization
nvidia-smi dmon -s u

# If low, increase settings
sed -i 's/GPU_MEMORY_UTILIZATION=0.75/GPU_MEMORY_UTILIZATION=0.90/' docker-compose.yml
sed -i 's/MAX_CONCURRENCY=5/MAX_CONCURRENCY=10/' docker-compose.yml

docker-compose down && docker-compose up -d
```

### Can't access externally

```bash
# Test internal access first
curl http://localhost:7777/health

# Check port is exposed in Runpod UI
# Verify firewall isn't blocking the port

# Check container is listening
netstat -tuln | grep 7777
```

---

## Using the API

### Python Example

```python
import requests

# Health check
response = requests.get("https://stale-lavender-junglefowl-7777.proxy.runpod.net/health")
print(response.json())

# Process an image
url = "https://stale-lavender-junglefowl-7777.proxy.runpod.net/ocr/image"
files = {"file": open("document.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(result["results"][0]["text"])
```

### cURL Examples

```bash
# Image OCR
curl -X POST "https://your-pod-url:7777/ocr/image" \
  -F "file=@image.jpg" \
  | python -m json.tool

# PDF OCR
curl -X POST "https://your-pod-url:7777/ocr/pdf" \
  -F "file=@document.pdf" \
  | python -m json.tool

# Batch processing
curl -X POST "https://your-pod-url:7777/ocr/batch" \
  -F "files=@doc1.pdf" \
  -F "files=@image1.jpg" \
  | python -m json.tool
```

---

## Performance Tuning for RTX 4090

### High Performance Mode
```bash
sed -i 's/GPU_MEMORY_UTILIZATION=0.90/GPU_MEMORY_UTILIZATION=0.95/' docker-compose.yml
sed -i 's/MAX_CONCURRENCY=10/MAX_CONCURRENCY=15/' docker-compose.yml
docker-compose down && docker-compose up -d
```

### Balanced Mode (Recommended)
```bash
sed -i 's/GPU_MEMORY_UTILIZATION=.*/GPU_MEMORY_UTILIZATION=0.90/' docker-compose.yml
sed -i 's/MAX_CONCURRENCY=.*/MAX_CONCURRENCY=10/' docker-compose.yml
docker-compose down && docker-compose up -d
```

### Conservative Mode
```bash
sed -i 's/GPU_MEMORY_UTILIZATION=.*/GPU_MEMORY_UTILIZATION=0.75/' docker-compose.yml
sed -i 's/MAX_CONCURRENCY=.*/MAX_CONCURRENCY=5/' docker-compose.yml
docker-compose down && docker-compose up -d
```

---

## Complete One-Command Deployment

For the fastest deployment, run this single command block:

```bash
cd /workspace && \
git clone https://github.com/Bogdanovich77/DeekSeek-OCR---Dockerized-API.git && \
cd DeekSeek-OCR---Dockerized-API && \
mkdir -p models outputs data && \
pip install -U huggingface-hub && \
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir models/deepseek-ai/DeepSeek-OCR \
  --local-dir-use-symlinks False && \
sed -i 's/- "8000:8000"/- "7777:8000"/' docker-compose.yml && \
sed -i 's/GPU_MEMORY_UTILIZATION=0.85/GPU_MEMORY_UTILIZATION=0.90/' docker-compose.yml && \
sed -i 's/MAX_CONCURRENCY=5/MAX_CONCURRENCY=10/' docker-compose.yml && \
docker-compose build && \
docker-compose up -d && \
echo "Waiting 120 seconds for model to load..." && \
sleep 120 && \
curl http://localhost:7777/health
```

**Note:** This takes ~40-60 minutes total (mostly unattended downloading/building)

---

## Summary

You now have a production-ready DeepSeek-OCR API running on your Runpod pod:

- ✅ Port 7777 exposed
- ✅ vLLM backend for fast inference
- ✅ Health monitoring enabled
- ✅ Auto-restart on failure
- ✅ Optimized for RTX 4090 (24GB)
- ✅ Handles 10 concurrent requests
- ✅ REST API with /ocr/image, /ocr/pdf, /ocr/batch

**Your API URL:**
```
https://stale-lavender-junglefowl-7777.proxy.runpod.net
```

Happy OCR processing! 🚀