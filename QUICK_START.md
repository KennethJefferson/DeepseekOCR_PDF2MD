# DeepSeek-OCR Docker - Quick Start Guide

**5-Minute Quick Reference** for deploying the Dockerized DeepSeek-OCR API on Runpod

---

## TL;DR - Complete Deployment in One Command

Copy and paste this into your Runpod terminal:

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
sleep 120 && \
curl http://localhost:7777/health
```

**Time:** 40-60 minutes (mostly unattended)

---

## Essential Commands

### Start/Stop Service
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart deepseek-ocr

# View logs
docker-compose logs -f deepseek-ocr
```

### Health & Status
```bash
# Health check
curl http://localhost:7777/health

# GPU usage
nvidia-smi

# Container status
docker-compose ps
```

### Testing
```bash
# Test with image
curl -X POST "http://localhost:7777/ocr/image" \
  -F "file=@image.jpg" | python -m json.tool

# Test with PDF
curl -X POST "http://localhost:7777/ocr/pdf" \
  -F "file=@document.pdf" | python -m json.tool
```

---

## Your Setup

- **Pod:** stale_lavender_junglefowl
- **GPU:** RTX 4090 (24GB)
- **Port:** 7777
- **External URL:** https://stale-lavender-junglefowl-7777.proxy.runpod.net

---

## Quick Troubleshooting

### Container won't start
```bash
docker-compose logs deepseek-ocr
docker-compose down && docker-compose up -d
```

### Out of memory
```bash
sed -i 's/GPU_MEMORY_UTILIZATION=.*/GPU_MEMORY_UTILIZATION=0.75/' docker-compose.yml
docker-compose down && docker-compose up -d
```

### Model not loaded
```bash
ls -lh models/deepseek-ai/DeepSeek-OCR/
# Should see ~9GB of files
```

---

## API Endpoints

- `GET  /health` - Health check
- `POST /ocr/image` - Process single image
- `POST /ocr/pdf` - Process PDF document
- `POST /ocr/batch` - Batch process multiple files

---

## Performance Settings

**Current (Balanced):**
- GPU Memory: 90%
- Concurrency: 10 requests
- Good for most workloads

**High Performance:**
```bash
sed -i 's/GPU_MEMORY_UTILIZATION=.*/GPU_MEMORY_UTILIZATION=0.95/' docker-compose.yml
sed -i 's/MAX_CONCURRENCY=.*/MAX_CONCURRENCY=15/' docker-compose.yml
docker-compose down && docker-compose up -d
```

**Conservative:**
```bash
sed -i 's/GPU_MEMORY_UTILIZATION=.*/GPU_MEMORY_UTILIZATION=0.75/' docker-compose.yml
sed -i 's/MAX_CONCURRENCY=.*/MAX_CONCURRENCY=5/' docker-compose.yml
docker-compose down && docker-compose up -d
```

---

## File Locations

- **Repository:** `/workspace/DeekSeek-OCR---Dockerized-API/`
- **Model:** `/workspace/DeekSeek-OCR---Dockerized-API/models/deepseek-ai/DeepSeek-OCR/`
- **Outputs:** `/workspace/DeekSeek-OCR---Dockerized-API/outputs/`
- **Test Data:** `/workspace/DeekSeek-OCR---Dockerized-API/data/`

---

## Need More Help?

See the full guide: `RUNPOD_DOCKER_DEPLOYMENT.md`