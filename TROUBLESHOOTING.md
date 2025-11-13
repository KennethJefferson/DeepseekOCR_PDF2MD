# Troubleshooting Guide - DeepSeek-OCR Server on Runpod

This guide covers common issues and solutions when deploying the DeepSeek-OCR server on Runpod.

---

## Quick Diagnostic Commands

```bash
# Check if server is running
pgrep -f "uvicorn.*7777"

# Check server logs (if running)
ps aux | grep uvicorn

# Check port status
netstat -tuln | grep 7777

# Check GPU status
nvidia-smi

# Check Python packages
pip list | grep -E "torch|vllm|transformers|fastapi"

# Run verification script
bash /workspace/verify_deployment.sh
```

---

## Common Issues and Solutions

### 1. Port 7777 Not Accessible Externally

**Symptom:** Server runs locally but can't access from outside Runpod

**Solutions:**
```bash
# Check if port is exposed in Runpod settings
# Runpod Dashboard → Your Pod → Settings → Exposed Ports → Add 7777

# Verify server is binding to all interfaces
export HOST="0.0.0.0"  # Not "127.0.0.1" or "localhost"

# Check firewall (usually not an issue on Runpod)
iptables -L -n | grep 7777
```

**External URL format:**
```
http://[your-pod-name]-7777.proxy.runpod.net
```

### 2. Model Download Issues

**Symptom:** Model files missing or incomplete

**Solutions:**
```bash
# Check current model size
du -sh /workspace/models/deepseek-ai/DeepSeek-OCR

# If < 8GB, re-download
rm -rf /workspace/models/deepseek-ai/DeepSeek-OCR

# Download with resume support
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR \
  --resume-download

# Alternative: Download without fast transfer if issues persist
export HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR
```

### 3. vLLM Loading Failures

**Symptom:** vLLM fails to load, falls back to transformers

**Solutions:**
```bash
# Check vLLM installation
python3 -c "import vllm; print(vllm.__version__)"

# Reinstall vLLM
pip uninstall vllm -y
pip install vllm==0.8.5 --no-cache-dir

# If CUDA version mismatch
pip install vllm --no-build-isolation

# Force transformers backend (edit ocr_service.py)
# Set self._load_with_vllm() to return False immediately
```

### 4. Out of Memory (OOM) Errors

**Symptom:** Server crashes with CUDA OOM error

**Solutions:**
```bash
# Reduce GPU memory utilization
export GPU_MEMORY_UTILIZATION=0.7  # Default is 0.85

# Check what's using GPU memory
nvidia-smi

# Kill other processes using GPU
fuser -v /dev/nvidia*

# Clear GPU cache before starting
python3 -c "import torch; torch.cuda.empty_cache()"

# Use lower resolution for processing
# In API calls, use resolution="low" instead of "high"
```

### 5. Missing Python Packages

**Symptom:** ImportError for various packages

**Quick fix for all common missing packages:**
```bash
pip install --upgrade \
  torch torchvision \
  transformers \
  vllm \
  fastapi uvicorn python-multipart \
  pdf2image PyMuPDF Pillow \
  accelerate huggingface-hub \
  pydantic pydantic-settings \
  aiohttp aiofiles \
  addict matplotlib einops timm \
  hf-transfer
```

### 6. PDF Processing Errors

**Symptom:** Can't convert PDF to images

**Solutions:**
```bash
# Install poppler-utils
apt-get update
apt-get install -y poppler-utils

# Test poppler
pdftoppm -h

# Alternative: Force PyMuPDF backend
# The server will automatically fall back to PyMuPDF if pdf2image fails
```

### 7. Server Won't Start

**Symptom:** Server fails to start with various errors

**Debug steps:**
```bash
# 1. Check Python version (should be 3.8+)
python3 --version

# 2. Try running directly without script
cd /workspace/deepseek-ocr-server
python3 -m app.main

# 3. Check for import errors
python3 -c "from app.main import app"

# 4. Enable debug logging
export LOG_LEVEL=DEBUG
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 7777 --log-level debug

# 5. Check file permissions
ls -la /workspace/deepseek-ocr-server/app/
chmod 644 /workspace/deepseek-ocr-server/app/*.py
```

### 8. Transformers Version Conflicts

**Symptom:** LlamaFlashAttention2 or other import errors

**Solutions:**
```bash
# Downgrade transformers
pip install transformers==4.37.2

# Or upgrade to latest
pip install --upgrade transformers

# Install flash attention
pip install flash-attn --no-build-isolation

# If flash attention fails, edit ocr_service.py to disable it
# Add use_flash_attention_2=False to model loading
```

### 9. API Response Errors

**Symptom:** 500 errors when calling API

**Debug:**
```bash
# Check server logs
# The terminal running the server will show errors

# Test with curl
curl -X POST http://localhost:7777/api/v1/ocr/pdf \
  -F "file=@test.pdf" \
  -F "resolution=base" \
  -v  # Verbose output

# Check health endpoint
curl http://localhost:7777/health | python3 -m json.tool
```

### 10. Slow Processing Speed

**Symptom:** PDF conversion takes too long

**Optimizations:**
```bash
# 1. Ensure vLLM is being used (check logs for "vLLM backend")

# 2. Use lower resolution
# API call with resolution="low" instead of "high"

# 3. Increase batch size (if processing multiple pages)
# Edit ocr_service.py to process in batches

# 4. Check GPU utilization
watch -n 1 nvidia-smi

# 5. Ensure GPU is being used (not CPU)
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Environment Variable Reference

```bash
# Core settings
export MODEL_PATH="/workspace/models/deepseek-ai/DeepSeek-OCR"
export PORT=7777
export HOST="0.0.0.0"

# Performance tuning
export GPU_MEMORY_UTILIZATION=0.85  # Reduce if OOM
export CUDA_VISIBLE_DEVICES=0        # GPU selection
export OMP_NUM_THREADS=8             # CPU threads

# Download optimization
export HF_HUB_ENABLE_HF_TRANSFER=1   # Fast downloads
export HF_HOME=/workspace/.cache     # Cache location

# Debugging
export LOG_LEVEL=INFO                # or DEBUG
export PYTHONUNBUFFERED=1            # Immediate output
```

---

## Testing Procedures

### Basic Health Check
```bash
curl http://localhost:7777/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_memory_free": "14.5GB",
  "gpu_memory_total": "24.0GB",
  "backend": "vLLM"
}
```

### Test with Sample PDF
```bash
# Download test PDF
wget https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf

# Test conversion
curl -X POST http://localhost:7777/api/v1/ocr/pdf \
  -F "file=@dummy.pdf" \
  -F "resolution=base" \
  -o result.json

# View result
python3 -m json.tool result.json | head -50
```

### Performance Test
```bash
# Time a conversion
time curl -X POST http://localhost:7777/api/v1/ocr/pdf \
  -F "file=@test.pdf" \
  -F "resolution=base" \
  -o /dev/null
```

---

## Recovery Procedures

### Complete Reset
```bash
# 1. Stop everything
bash /workspace/stop_server.sh
pkill -f python
nvidia-smi  # Check no python processes using GPU

# 2. Clear cache
rm -rf /workspace/.cache/huggingface
python3 -c "import torch; torch.cuda.empty_cache()"

# 3. Reinstall critical packages
pip install --upgrade --force-reinstall \
  torch vllm transformers fastapi

# 4. Re-download model if needed
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR \
  --resume-download

# 5. Start fresh
bash /workspace/start_server.sh
```

### Emergency Fallback (CPU Mode)
```bash
# If GPU completely fails, run on CPU (VERY SLOW)
export CUDA_VISIBLE_DEVICES=""
export DEVICE="cpu"

# Edit ocr_service.py
# Change device_map="cuda" to device_map="cpu"

# Start with reduced settings
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 7777 --workers 1
```

---

## Getting Help

### Logs to Collect
When reporting issues, collect:

1. **Server startup log:**
```bash
bash /workspace/start_server.sh 2>&1 | tee server_startup.log
```

2. **Verification output:**
```bash
bash /workspace/verify_deployment.sh > verification.log 2>&1
```

3. **System information:**
```bash
nvidia-smi > gpu_info.log
pip list > packages.log
python3 --version > python_version.log
uname -a > system_info.log
```

4. **Error traceback:**
Copy the full Python error traceback from the terminal

### Quick Fixes Checklist

- [ ] Port 7777 exposed in Runpod settings?
- [ ] Model fully downloaded (~8-10GB)?
- [ ] All Python packages installed?
- [ ] GPU has enough memory (>10GB free)?
- [ ] Server binding to 0.0.0.0 not localhost?
- [ ] poppler-utils installed?
- [ ] Files have correct permissions?
- [ ] Using correct external URL format?

---

## Performance Benchmarks

Expected processing times on RTX 4090:

| PDF Type | Pages | Resolution | vLLM | Transformers |
|----------|-------|------------|------|--------------|
| Text-heavy | 1 | base | ~3s | ~8s |
| Text-heavy | 10 | base | ~25s | ~70s |
| Image-heavy | 1 | high | ~5s | ~12s |
| Mixed content | 5 | base | ~15s | ~40s |

If your times are significantly higher, check:
1. vLLM is being used (not falling back)
2. GPU is being utilized (check nvidia-smi)
3. No other processes competing for GPU

---

## Contact and Support

For issues specific to:
- **Runpod platform:** support@runpod.io
- **DeepSeek model:** Check HuggingFace model card
- **This deployment:** Review the MANUAL_RUNPOD_DEPLOYMENT.md guide

Remember to always check the server logs first - they usually contain the exact error and sometimes the solution!