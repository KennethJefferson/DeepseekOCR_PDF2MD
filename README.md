# DeepSeek-OCR PDF to Markdown Conversion System

A high-performance, GPU-powered PDF to Markdown conversion system using DeepSeek-OCR model. Features a FastAPI server running on Runpod and a concurrent Go client for batch processing.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Server Setup](#server-setup)
- [Client Usage](#client-usage)
- [API Documentation](#api-documentation)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## ✨ Features

### Server
- **DeepSeek-OCR Integration**: State-of-the-art OCR model for accurate text extraction
- **GPU Acceleration**: Optimized for NVIDIA GPUs with vLLM backend
- **FastAPI Framework**: High-performance async API server
- **Multi-format Support**: PDF to Markdown conversion with structure preservation
- **Configurable Resolution**: Multiple quality presets (tiny, small, base, large, gundam)
- **Health Monitoring**: Built-in health checks and status endpoints
- **Docker Support**: Containerized deployment for easy scaling

### Client
- **Parallel Processing**: Configurable worker pool for concurrent PDF processing
- **Directory Scanning**: Recursive PDF discovery with structure preservation
- **Progress Tracking**: Real-time progress bar with statistics
- **Retry Logic**: Automatic retry on transient failures
- **Skip Processed**: Intelligent detection of already-processed files
- **Cross-platform**: Works on Windows, Linux, and macOS

## 🏗️ Architecture

```
┌─────────────────────┐         ┌────────────────────┐
│                     │         │                    │
│   Go Client         │ ──HTTP──▶   FastAPI Server	 │
│   (Local Machine)   │         │   (Runpod GPU Pod) │
│                     │         │                    │
│  ┌──────────────┐   │         │  ┌──────────────┐  │
│  │ Worker Pool  │   │         │  │ DeepSeek-OCR │  │
│  │  (Parallel)  │   │         │  │    Model     │  │
│  └──────────────┘   │         │  └──────────────┘  │
│                     │         │                    │
│  ┌──────────────┐   │         │  ┌──────────────┐  │
│  │ PDF Scanner  │   │         │  │ PDF Processor│  │
│  └──────────────┘   │         │  └──────────────┘  │
│                     │         │                    │
│  ┌──────────────┐   │         │  ┌──────────────┐  │
│  │ MD Writer    │   │         │  │   vLLM/HF    │  │
│  └──────────────┘   │         │  └──────────────┘  │
└─────────────────────┘         └────────────────────┘
```

## 📋 Requirements

### Server Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 minimum, RTX 4090 recommended)
- **CUDA**: 11.8 or compatible version
- **Python**: 3.10 - 3.12
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for model weights and processing

### Client Requirements
- **Go**: 1.21 or later
- **OS**: Windows, Linux, or macOS
- **Network**: Stable connection to server

## 🚀 Installation

### Quick Start (Runpod)

1. **Create a Runpod Pod**:
   - GPU: RTX 4090 or A100
   - Template: PyTorch or CUDA
   - Expose HTTP Port: 8000

2. **Upload and run setup script on pod**:
   ```bash
   # On your local machine
   runpod send <pod-id>:/workspace/runpod_setup.sh ./runpod_setup.sh
   runpod send <pod-id>:/workspace/deepseek-ocr-server ./server/

   # Connect to pod via web terminal
   bash /workspace/runpod_setup.sh
   ```

3. **Build Go client locally**:
   ```bash
   cd client
   go build -o deepseek-client
   ```

### Manual Installation

#### Server Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/DeepSeekOCR_PDF2Markdown.git
   cd DeepSeekOCR_PDF2Markdown
   ```

2. **Download model weights** (8-10GB):
   ```bash
   pip install huggingface-hub
   huggingface-cli download deepseek-ai/DeepSeek-OCR \
       --local-dir ./server/models/deepseek-ai/DeepSeek-OCR
   ```

3. **Install Python dependencies**:
   ```bash
   cd server
   pip install -r requirements.txt
   ```

4. **Start the server**:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

#### Client Setup

1. **Build the Go client**:
   ```bash
   cd client
   go mod download
   go build -o deepseek-client
   ```

2. **Configure client** (edit `client/config.yaml`):
   ```yaml
   api:
     url: "http://your-runpod-url:8000"
     timeout: 300
   ```

## 📚 Server Setup

### Using Docker

1. **Build image**:
   ```bash
   cd server
   docker build -t deepseek-ocr-server .
   ```

2. **Run container**:
   ```bash
   docker run -d \
     --name deepseek-ocr \
     --gpus all \
     -p 8000:8000 \
     -v ./models:/app/models \
     -e MODEL_PATH=/app/models/deepseek-ai/DeepSeek-OCR \
     deepseek-ocr-server
   ```

### Using Docker Compose

```bash
cd server
docker-compose up -d
```

### Environment Variables

```bash
# Model Configuration
MODEL_PATH=/app/models/deepseek-ai/DeepSeek-OCR
GPU_MEMORY_UTILIZATION=0.85
CUDA_VISIBLE_DEVICES=0

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Runpod (optional)
RUNPOD_POD_ID=your-pod-id
RUNPOD_API_KEY=your-api-key
```

## 💻 Client Usage

### Basic Usage

```bash
# Process PDFs with 4 workers
./deepseek-client -workers 4 -scan /path/to/pdfs

# With custom output directory
./deepseek-client -workers 4 -scan /path/to/pdfs -output /path/to/markdown

# With custom API URL
./deepseek-client -workers 4 -scan /path/to/pdfs -api http://runpod-url:8000

# Non-recursive scanning
./deepseek-client -workers 4 -scan /path/to/pdfs -recursive=false

# Overwrite existing files
./deepseek-client -workers 4 -scan /path/to/pdfs -overwrite
```

### Command-Line Options

| Flag | Description | Required | Default |
|------|-------------|----------|---------|
| `-workers` | Number of parallel workers | Yes | - |
| `-scan` | Directory to scan for PDFs | Yes | - |
| `-api` | DeepSeek-OCR API URL | No | http://localhost:8000 |
| `-output` | Output directory for markdown | No | ./output |
| `-recursive` | Recursively scan directories | No | true |
| `-overwrite` | Overwrite existing files | No | false |
| `-config` | Configuration file path | No | config.yaml |
| `-verbose` | Enable verbose logging | No | false |

### Configuration File

Create `config.yaml`:

```yaml
api:
  url: "http://your-server:8000"
  timeout: 300

processing:
  workers: 4
  retry_attempts: 3
  retry_delay: 5

input:
  scan_directory: "./pdfs"
  recursive: true

output:
  directory: "./markdown"
  preserve_structure: true
  overwrite_existing: false
```

## 📡 API Documentation

### Endpoints

#### Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_memory_free": "20.5GB",
  "gpu_memory_total": "24.0GB"
}
```

#### Process PDF
```http
POST /api/v1/ocr/pdf
Content-Type: multipart/form-data

Form Data:
- file: PDF file (binary)
- resolution: "base" (optional: tiny|small|base|large|gundam)

Response:
{
  "success": true,
  "filename": "document.pdf",
  "total_pages": 5,
  "pages": [
    {
      "page_number": 1,
      "markdown": "# Page Content...",
      "processing_time_ms": 450
    }
  ],
  "total_processing_time_ms": 2250
}
```

#### Process PDF from URL
```http
POST /api/v1/ocr/pdf-url
Content-Type: application/json

Body:
{
  "url": "https://example.com/document.pdf",
  "resolution": "base"
}

Response: Same as /api/v1/ocr/pdf
```

#### Server Status
```http
GET /api/v1/status

Response:
{
  "status": "running",
  "model_loaded": true,
  "stats": {
    "total_requests": 150,
    "successful_requests": 148,
    "failed_requests": 2,
    "total_pages_processed": 1250,
    "average_processing_time_ms": 485.5,
    "average_pages_per_pdf": 8.4,
    "success_rate": 98.7
  },
  "uptime_seconds": 3600
}
```

## ⚡ Performance

### Benchmarks

| GPU | Pages/Min | Avg Time/Page | Daily Capacity |
|-----|-----------|---------------|----------------|
| RTX 3070 (8GB) | 5-10 | 6-12s | 7,200-14,400 |
| RTX 4090 (24GB) | 50-100 | 0.6-1.2s | 72,000-144,000 |
| A100-40G | 200+ | 0.3s | 288,000+ |

### Optimization Tips

1. **Server Optimization**:
   - Adjust `GPU_MEMORY_UTILIZATION` (0.7-0.95)
   - Use appropriate resolution for your documents
   - Enable vLLM prefix caching for similar documents

2. **Client Optimization**:
   - Adjust workers based on server capacity (typically 4-8)
   - Process smaller PDFs first for better throughput
   - Use local SSD for input/output directories

3. **Network Optimization**:
   - Use same region for client and Runpod pod
   - Consider compression for large PDFs
   - Implement connection pooling

## 🔧 Troubleshooting

### Common Issues

#### Server Issues

**CUDA Out of Memory**:
```bash
# Reduce GPU memory utilization
export GPU_MEMORY_UTILIZATION=0.7

# Or reduce batch size
```

**Model Loading Fails**:
```bash
# Verify model path
ls -la /app/models/deepseek-ai/DeepSeek-OCR

# Re-download model
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir ./models
```

**Server Timeout**:
```bash
# Increase timeout in client config
api:
  timeout: 600  # 10 minutes
```

#### Client Issues

**Connection Refused**:
```bash
# Check server is running
curl http://server-url:8000/health

# Verify firewall/port settings
```

**No PDFs Found**:
```bash
# Check file extensions (case-sensitive on Linux)
find /path/to/pdfs -name "*.pdf" -o -name "*.PDF"

# Verify permissions
ls -la /path/to/pdfs
```

### Debugging

Enable verbose logging:

**Server**:
```bash
export LOG_LEVEL=DEBUG
python -m uvicorn app.main:app --log-level debug
```

**Client**:
```bash
./deepseek-client -workers 4 -scan /path -verbose
```

## 📊 Monitoring

### Server Metrics

Monitor GPU usage:
```bash
nvidia-smi -l 1
```

Monitor server logs:
```bash
docker logs -f deepseek-ocr
```

### Client Progress

The client displays real-time progress:
```
Processing PDFs │████████████░░░░│ 75% (75/100)
```

Final statistics:
```
=============================
Processing Complete
=============================
Total PDFs:        100
Successful:        98
Failed:            2
Total Pages:       850
Avg Pages/PDF:     8.7
Avg Time/PDF:      2.3 seconds
Throughput:        35.4 pages/min
Total Time:        4m 32s
=============================
```

## 🔒 Security

- Run server behind reverse proxy (nginx, traefik)
- Implement authentication if exposing to internet
- Sanitize uploaded PDFs
- Set resource limits in production
- Use secrets management for API keys

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/DeepSeekOCR_PDF2Markdown/issues)
- **Documentation**: [Wiki](https://github.com/your-username/DeepSeekOCR_PDF2Markdown/wiki)

## 🙏 Acknowledgments

- [DeepSeek-AI](https://github.com/deepseek-ai) for the OCR model
- [vLLM](https://github.com/vllm-project/vllm) for optimized inference
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Runpod](https://runpod.io/) for GPU infrastructure