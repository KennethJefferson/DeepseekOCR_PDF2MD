# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A two-part PDF to Markdown conversion system using the DeepSeek-OCR model:
- **Server**: FastAPI server running on Runpod GPU infrastructure with DeepSeek-OCR model
- **Client**: Concurrent Go application for batch PDF processing with worker pool pattern

## Architecture

### Client Architecture (Go)
The Go client follows a standard worker pool pattern:
- **Scanner**: Discovers PDF files recursively, creates jobs, sends to channel
- **Worker Pool**: Configurable number of workers process PDFs concurrently
- **Collector**: Single goroutine writes results to disk with progress tracking
- **Shutdown Sequence**: Close jobs channel → wait for workers → close results channel → wait for collector

### Server Architecture (Python/FastAPI)
- FastAPI server with DeepSeek-OCR model loaded via vLLM
- Endpoints: `/health`, `/api/v1/ocr/pdf`, `/api/v1/ocr/pdf-url`, `/api/v1/status`
- GPU-accelerated inference with configurable memory utilization
- Docker containerized for Runpod deployment

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

**Local development:**
```bash
cd server
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Docker build and run:**
```bash
cd server
docker build -t deepseek-ocr-server .
docker run -d --gpus all -p 8000:8000 \
  -v ./models:/app/models \
  -e MODEL_PATH=/app/models/deepseek-ai/DeepSeek-OCR \
  deepseek-ocr-server
```

**Runpod deployment:**
```bash
# Quick setup script for Runpod
bash runpod_quickstart.sh

# Or manual deployment
bash deploy_to_runpod.sh

# Or direct upload via Jupyter
python upload_via_jupyter.py
```

**Testing deployment:**
```bash
bash test_deployment.sh http://your-runpod-url:8000
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
- API URL: `http://localhost:8000`
- Timeout: 300 seconds
- Workers: Required flag (no default)
- Recursive: `true`
- Overwrite: `false`
- Output directory: `./output`

### Server Environment Variables
- `MODEL_PATH`: Path to DeepSeek-OCR model weights
- `GPU_MEMORY_UTILIZATION`: 0.85 (85% GPU memory)
- `HOST`: 0.0.0.0
- `PORT`: 8000
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
curl http://localhost:8000/health
```

**Test PDF processing:**
```bash
curl -X POST http://localhost:8000/api/v1/ocr/pdf \
  -F "file=@test.pdf" \
  -F "resolution=base"
```

**Test client locally:**
```bash
# Create test directory with PDFs
mkdir test_pdfs
# Add some PDF files
./deepseek-client -workers 2 -scan test_pdfs -output test_output
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

**Client can't connect to server:**
- Check API URL is correct and includes protocol (`http://`)
- Verify server is running: `curl http://server:8000/health`
- Check firewall/port configuration

**Server out of GPU memory:**
- Reduce `GPU_MEMORY_UTILIZATION` environment variable
- Use smaller resolution preset ("tiny" or "small")
- Reduce batch size in vLLM configuration

**No PDFs found:**
- Verify scan directory path is correct
- Check recursive flag if PDFs are in subdirectories
- File extension matching is case-sensitive on Linux
