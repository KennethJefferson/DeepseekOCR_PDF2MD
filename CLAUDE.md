# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A two-part PDF to Markdown conversion system using the DeepSeek-OCR model:
- **Server**: FastAPI server running on Runpod GPU infrastructure with DeepSeek-OCR model
- **Client**: Concurrent Go application for batch PDF processing with worker pool pattern

### Deployment Options

**Option 1: Dockerized Deployment (Recommended for Runpod)**
- Uses pre-built Docker container from [Bogdanovich77/DeekSeek-OCR---Dockerized-API](https://github.com/Bogdanovich77/DeekSeek-OCR---Dockerized-API)
- Production-ready with health monitoring, auto-restart, and bug fixes
- vLLM backend for optimized GPU inference
- See: `RUNPOD_DOCKER_DEPLOYMENT.md` and `QUICK_START.md`

**Option 2: Manual Python Setup**
- Direct Python installation with custom configuration
- More flexibility for development and customization
- See: `COMPLETE_TERMINAL_SETUP.md` and `TERMINAL_COMMANDS.md`

## Architecture

### Client Architecture (Go)
The Go client follows a standard worker pool pattern:
- **Scanner**: Discovers PDF files recursively, creates jobs, sends to channel
- **Worker Pool**: Configurable number of workers process PDFs concurrently
- **Collector**: Single goroutine writes results to disk with progress tracking
- **Shutdown Sequence**: Close jobs channel → wait for workers → close results channel → wait for collector

### Server Architecture (Python/FastAPI)
- FastAPI server with DeepSeek-OCR model loaded via vLLM
- Endpoints: `/health`, `/api/v1/ocr/pdf`, `/api/v1/ocr/pdf-url`, `/api/v1/status`, `/ocr/image`, `/ocr/batch`
- GPU-accelerated inference with configurable memory utilization
- Docker containerized for Runpod deployment
- Default port: 7777 (configurable)
- Production features: health monitoring, auto-restart, concurrency control (10 concurrent requests)

## Building and Running

### Client Commands

**Build the client:**
```bash
cd client
go build -o deepseek-client
# Windows: go build -o deepseek-client.exe
```

**Run the client:**
```bash
# Required flags
./deepseek-client -workers 4 -scan /path/to/pdfs

# With all options
./deepseek-client \
  -workers 4 \
  -scan /path/to/pdfs \
  -api http://runpod-url:8000 \
  -output ./markdown \
  -recursive=true \
  -overwrite=false \
  -verbose
```

**Configuration file** (optional `config.yaml`):
```yaml
api:
  url: "http://localhost:8000"
  timeout: 300
processing:
  workers: 4
  retry_attempts: 3
  retry_delay: 5
output:
  directory: "./output"
  preserve_structure: true
```

### Server Commands

#### Dockerized Deployment (Recommended for Runpod)

**Prerequisites - Install Docker on Runpod:**
```bash
# Quick Docker installation
apt-get update && \
apt-get install -y ca-certificates curl gnupg lsb-release && \
mkdir -p /etc/apt/keyrings && \
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
apt-get update && \
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
systemctl start docker

# Install NVIDIA Container Toolkit (GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
apt-get update && \
apt-get install -y nvidia-container-toolkit && \
nvidia-ctk runtime configure --runtime=docker && \
systemctl restart docker
```

**One-Command Deployment (Port 7777):**
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

**Docker Management Commands:**
```bash
# Start service
docker-compose up -d

# Stop service
docker-compose down

# Restart service
docker-compose restart deepseek-ocr

# View logs
docker-compose logs -f deepseek-ocr

# Health check
curl http://localhost:7777/health
```

#### Manual Python Setup (Original Method)

**Local development:**
```bash
cd server
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 7777
```

**Custom Docker build:**
```bash
cd server
docker build -t deepseek-ocr-server .
docker run -d --gpus all -p 7777:7777 \
  -v ./models:/app/models \
  -e MODEL_PATH=/app/models/deepseek-ai/DeepSeek-OCR \
  -e PORT=7777 \
  deepseek-ocr-server
```

**Testing deployment:**
```bash
# Test health endpoint
curl http://localhost:7777/health

# Test PDF processing
curl -X POST http://localhost:7777/api/v1/ocr/pdf \
  -F "file=@test.pdf" \
  -F "resolution=base"
```

## Code Structure

### Client Internal Packages

**`internal/scanner/scanner.go`**
- `Scanner` struct: Handles PDF discovery with recursive/non-recursive modes
- `ScanForPDFs()`: Streams jobs to channel as files are discovered
- `ScanForPDFsBatch()`: Returns all jobs at once (alternative method)
- Automatically skips already-processed files (checks for existing .md output)

**`internal/api/client.go`**
- `Client` struct: HTTP client with configurable timeout
- `ProcessPDF()`: Main method - uploads PDF, returns combined markdown + page count
- `ProcessPDFWithRetry()`: Wrapper with retry logic for transient failures
- `HealthCheck()`: Validates API availability and model loaded status
- `isRetryableError()`: Determines if errors warrant retry (network/5xx errors)

**`internal/output/writer.go`**
- `Writer` struct: Handles markdown file writing
- `WriteMarkdownFile()`: Writes markdown with metadata header (generated_by, timestamp, source)
- Automatic directory creation with structure preservation
- `addMetadata()`: Prepends YAML frontmatter to output files

### Client Main Flow

1. Parse flags and load config (flags override config file)
2. Health check API before starting
3. Start scanner goroutine (discovers PDFs, sends to jobs channel)
4. Launch N worker goroutines (process PDFs via API)
5. Launch collector goroutine (writes results, updates progress bar)
6. Wait for scanner → workers → collector (sequential shutdown)
7. Print statistics and exit

### Server Structure

**`app/main.py`**
- FastAPI application with CORS middleware
- Health check endpoint with GPU memory stats
- PDF processing endpoints (multipart file upload and URL-based)
- Status endpoint for server statistics

**`app/ocr_service.py`**
- DeepSeek-OCR model loading and inference
- vLLM integration for optimized GPU inference

**`app/pdf_processor.py`**
- PDF to image conversion (per-page)
- Resolution presets: tiny, small, base, large, gundam

## Progress Bar Style

The client uses the green Unicode block style with `github.com/schollz/progressbar/v3`:

```go
bar := progressbar.NewOptions(total,
    progressbar.OptionSetWriter(ansi.NewAnsiStdout()),
    progressbar.OptionEnableColorCodes(true),
    progressbar.OptionSetWidth(40),
    progressbar.OptionShowCount(),
    progressbar.OptionSetDescription("Processing PDFs"),
    progressbar.OptionSetTheme(progressbar.Theme{
        Saucer:        "[green]█[reset]",
        SaucerHead:    "[green]█[reset]",
        SaucerPadding: "░",
        BarStart:      "│",
        BarEnd:        "│",
    }),
)
```

Appearance: `│████████████░░░░│ 75% (75/100)`

## Concurrency Patterns

### Worker Pool Pattern (Client)
- Buffered job channel (size 100) prevents blocking scanner
- Buffered results channel (size 100) prevents blocking workers
- Two WaitGroups: one for workers, one for collector
- Atomic operations for all statistics (`atomic.AddInt64()`)
- Progress bar updated by collector goroutine (single writer)

### Retry Logic
- Configurable retry attempts (default: 3) and delay (default: 5s)
- Only retries network errors and 5xx server errors
- 4xx client errors fail immediately

## Key Configuration

### Client Defaults
- API URL: `http://localhost:7777`
- Timeout: 300 seconds
- Workers: Required flag (no default)
- Recursive: `true`
- Overwrite: `false`
- Output directory: `./output`

### Server Environment Variables (Docker)
- `MODEL_PATH`: Path to DeepSeek-OCR model weights
- `GPU_MEMORY_UTILIZATION`: 0.90 (90% GPU memory, optimized for RTX 4090)
- `MAX_CONCURRENCY`: 10 (concurrent request limit)
- `HOST`: 0.0.0.0
- `PORT`: 7777 (mapped to internal 8000)
- `CUDA_VISIBLE_DEVICES`: 0

### Server Environment Variables (Manual)
- `MODEL_PATH`: Path to DeepSeek-OCR model weights
- `GPU_MEMORY_UTILIZATION`: 0.85 (85% GPU memory)
- `HOST`: 0.0.0.0
- `PORT`: 7777
- `WORKERS`: 1 (uvicorn workers)

## Output Format

Markdown files include:
1. YAML frontmatter with metadata (generator, timestamp, source)
2. Page separators: `<!-- Page N -->`
3. Page content (converted from PDF by OCR model)
4. Page dividers: `---`

## Testing

**Test server health:**
```bash
# Local testing
curl http://localhost:7777/health

# External (Runpod)
curl https://your-pod-id-7777.proxy.runpod.net/health
```

**Test PDF processing (Dockerized API):**
```bash
# Image OCR
curl -X POST http://localhost:7777/ocr/image \
  -F "file=@test_image.jpg"

# PDF OCR
curl -X POST http://localhost:7777/ocr/pdf \
  -F "file=@test.pdf"

# Batch processing
curl -X POST http://localhost:7777/ocr/batch \
  -F "files=@doc1.pdf" \
  -F "files=@image1.jpg"
```

**Test PDF processing (Manual API):**
```bash
curl -X POST http://localhost:7777/api/v1/ocr/pdf \
  -F "file=@test.pdf" \
  -F "resolution=base"
```

**Test client locally:**
```bash
# Create test directory with PDFs
mkdir test_pdfs
# Add some PDF files
./deepseek-client -workers 2 -scan test_pdfs -api http://localhost:7777 -output test_output
```

## Dependencies

### Client (Go)
- `github.com/schollz/progressbar/v3` - Progress bar
- `github.com/k0kubun/go-ansi` - ANSI color support
- `gopkg.in/yaml.v3` - YAML configuration

### Server (Python)
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `vllm` - Optimized LLM inference
- `pdf2image` - PDF to image conversion
- `Pillow` - Image processing
- `torch` - PyTorch for model inference

## Common Issues

**Docker not installed on Runpod:**
- Run the Docker installation commands in the "Server Commands" section
- Verify with: `docker --version` and `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

**Client can't connect to server:**
- Check API URL is correct and includes protocol (`http://`)
- Verify server is running: `curl http://server:7777/health`
- Check firewall/port configuration
- For Runpod: Ensure port 7777 is exposed in pod settings

**Docker container won't start:**
- Check logs: `docker-compose logs deepseek-ocr`
- Verify model downloaded: `ls -lh models/deepseek-ai/DeepSeek-OCR/`
- Check GPU access: `nvidia-smi`
- Restart: `docker-compose down && docker-compose up -d`

**Server out of GPU memory:**
- Dockerized: Edit `docker-compose.yml` and reduce `GPU_MEMORY_UTILIZATION` to 0.75
- Manual: Reduce `GPU_MEMORY_UTILIZATION` environment variable
- Use smaller resolution preset ("tiny" or "small")
- Reduce `MAX_CONCURRENCY` in docker-compose.yml

**Model loading failed:**
- Check transformers version compatibility issues
- Verify all model files present (~9GB total)
- Re-download model if incomplete
- See `RUNPOD_DOCKER_DEPLOYMENT.md` troubleshooting section

**No PDFs found:**
- Verify scan directory path is correct
- Check recursive flag if PDFs are in subdirectories
- File extension matching is case-sensitive on Linux

## Documentation Files

- **`RUNPOD_DOCKER_DEPLOYMENT.md`** - Complete Docker deployment guide for Runpod (recommended)
- **`QUICK_START.md`** - Quick reference for Docker deployment
- **`COMPLETE_TERMINAL_SETUP.md`** - Manual Python setup (all steps from scratch)
- **`TERMINAL_COMMANDS.md`** - Terminal commands for manual setup
- **`README.md`** - Project overview and features
- **`WEB_TERMINAL_COMMANDS.md`** - Web terminal specific commands
- **`RUNPOD_QUICK_SETUP.md`** - Quick setup notes
