# Complete Terminal Setup Guide for DeepSeek-OCR on Runpod

This guide provides every step from scratch to set up the DeepSeek-OCR server on Runpod using only the terminal - no Jupyter required.

## Prerequisites
- Runpod account with credits
- GPU pod (RTX 4090 or A100 recommended)

---

## Step 1: Create and Configure Runpod Pod

### In Runpod Dashboard:
1. Click "Deploy" → "Pods" → "GPU Pod"
2. Select GPU: RTX 4090 or A100
3. Select Template: PyTorch 2.1.0 or RunPod Pytorch 2.0.1
4. Configure:
   - Container Disk: 50 GB minimum
   - Volume Disk: 100 GB (optional but recommended)
5. Expand "Advanced Options":
   - Expose HTTP Ports: `7777`
   - Expose TCP Ports: `22` (for SSH, optional)
6. Click "Deploy On-Demand" or "Deploy Spot"
7. Wait for pod to start (~2-3 minutes)

### Get Your Pod URL:
Once running, you'll see something like:
- Connection URL: `greasy-lime-clownfish.runpod.io`
- HTTP Access: `https://greasy-lime-clownfish-7777.proxy.runpod.net`

---

## Step 2: Connect to Pod Terminal

### Option A: Web Terminal (Recommended)
1. In Runpod dashboard, click "Connect"
2. Select "Connect to Web Terminal"
3. New browser tab opens with terminal

### Option B: SSH (if port 22 exposed)
```bash
ssh root@greasy-lime-clownfish.runpod.io -p 22
```

---

## Step 3: Initial System Setup

Once connected to terminal, run these commands:

```bash
# Navigate to workspace
cd /workspace

# Check GPU is available
nvidia-smi

# Check disk space (need 50GB+)
df -h /workspace

# Update system packages
apt-get update
apt-get install -y git curl wget unzip poppler-utils
```

---

## Step 4: Create Server Directory Structure

```bash
# Create all necessary directories
mkdir -p /workspace/deepseek-ocr-server/app
mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/temp

# Navigate to server directory
cd /workspace/deepseek-ocr-server
```

---

## Step 5: Create Server Python Files

We need to create 5 Python files. Copy and paste each command:

### 5.1 Create main.py

```bash
cat > /workspace/deepseek-ocr-server/app/main.py << 'EOF'
"""
DeepSeek-OCR FastAPI Server
PDF to Markdown conversion service using DeepSeek-OCR model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import logging
from typing import Optional
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from .ocr_service import DeepSeekOCRService
from .pdf_processor import PDFProcessor
from .models import (
    PDFResponse,
    HealthResponse,
    PDFURLRequest,
    ErrorResponse,
    PageResult,
    ProcessingStats
)

# Create FastAPI app
app = FastAPI(
    title="DeepSeek-OCR Server",
    version="1.0.0",
    description="GPU-powered PDF to Markdown conversion using DeepSeek-OCR"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
ocr_service: Optional[DeepSeekOCRService] = None
pdf_processor: Optional[PDFProcessor] = None

# Statistics tracking
stats = ProcessingStats()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global ocr_service, pdf_processor

    # Display ASCII banner
    print("\n" + "="*60)
    print("    DEEPSEEK-OCR SERVER")
    print("    PDF to Markdown Conversion")
    print("    Port: {} | Backend: vLLM".format(os.getenv("PORT", "7777")))
    print("="*60 + "\n")

    try:
        logger.info("Starting DeepSeek-OCR server...")

        # Initialize OCR service
        model_path = os.getenv("MODEL_PATH", "/workspace/models/deepseek-ai/DeepSeek-OCR")
        gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))

        logger.info(f"Loading model from: {model_path}")
        logger.info(f"GPU memory utilization: {gpu_memory_utilization}")

        ocr_service = DeepSeekOCRService(
            model_path=model_path,
            gpu_memory_utilization=gpu_memory_utilization
        )

        # Initialize PDF processor
        pdf_processor = PDFProcessor()

        logger.info("Server startup complete!")
        print("\n✓ Server ready at http://0.0.0.0:{}\n".format(os.getenv("PORT", "7777")))

    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down DeepSeek-OCR server...")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "DeepSeek-OCR Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "process_pdf": "/api/v1/ocr/pdf",
            "process_pdf_url": "/api/v1/ocr/pdf-url",
            "status": "/api/v1/status"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        health_status = ocr_service.get_health_status()
        health_status["uptime_seconds"] = stats.get_uptime()
        return HealthResponse(**health_status)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/api/v1/ocr/pdf", response_model=PDFResponse, tags=["OCR"])
async def process_pdf(
    file: UploadFile = File(...),
    resolution: str = "base"
):
    """Process a PDF file and convert it to markdown"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    if ocr_service is None or pdf_processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    temp_pdf_path = None
    temp_image_paths = []

    try:
        start_time = datetime.now()

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_pdf_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        logger.info(f"Processing PDF: {file.filename} ({len(content)} bytes)")

        # Convert PDF to images
        temp_image_paths = pdf_processor.pdf_to_images(temp_pdf_path, resolution=resolution)
        logger.info(f"Converted PDF to {len(temp_image_paths)} images")

        # Process each page
        pages = []
        for idx, image_path in enumerate(temp_image_paths, 1):
            page_start = datetime.now()
            markdown = await ocr_service.process_image(
                image_path,
                prompt="<image>\n<|grounding|>Convert the document to markdown."
            )
            page_time = (datetime.now() - page_start).total_seconds() * 1000
            pages.append(PageResult(
                page_number=idx,
                markdown=markdown,
                processing_time_ms=int(page_time)
            ))
            logger.info(f"Processed page {idx}/{len(temp_image_paths)} in {page_time:.0f}ms")

        total_time = (datetime.now() - start_time).total_seconds() * 1000
        stats.record_success(len(pages), total_time)

        return PDFResponse(
            success=True,
            filename=file.filename,
            total_pages=len(pages),
            pages=pages,
            total_processing_time_ms=int(total_time)
        )

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}\n{traceback.format_exc()}")
        stats.record_failure()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        for image_path in temp_image_paths:
            if os.path.exists(image_path):
                os.remove(image_path)

@app.get("/api/v1/status", tags=["Status"])
async def get_status():
    """Get server status and statistics"""
    return {
        "status": "running",
        "model_loaded": ocr_service is not None,
        "stats": stats.get_stats(),
        "uptime_seconds": stats.get_uptime()
    }

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7777"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False, log_level="info")
EOF
```

### 5.2 Create models.py

```bash
cat > /workspace/deepseek-ocr-server/app/models.py << 'EOF'
"""Data models for DeepSeek-OCR API"""

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class PageResult(BaseModel):
    """Result for a single page"""
    page_number: int
    markdown: str
    processing_time_ms: int

class PDFResponse(BaseModel):
    """Response for PDF processing"""
    success: bool
    filename: str
    total_pages: int
    pages: List[PageResult]
    total_processing_time_ms: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    cuda_available: bool
    gpu_memory_free: Optional[str] = None
    gpu_memory_total: Optional[str] = None
    model_path: Optional[str] = None
    uptime_seconds: Optional[float] = None

class PDFURLRequest(BaseModel):
    """Request for processing PDF from URL"""
    url: str
    resolution: str = "base"

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None

class ProcessingStats:
    """Statistics tracking"""
    def __init__(self):
        self.start_time = datetime.now()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_pages = 0
        self.total_time_ms = 0

    def record_success(self, pages: int, time_ms: float):
        self.total_requests += 1
        self.successful_requests += 1
        self.total_pages += pages
        self.total_time_ms += time_ms

    def record_failure(self):
        self.total_requests += 1
        self.failed_requests += 1

    def get_stats(self):
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        avg_time = (self.total_time_ms / self.successful_requests) if self.successful_requests > 0 else 0
        avg_pages = (self.total_pages / self.successful_requests) if self.successful_requests > 0 else 0

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_pages_processed": self.total_pages,
            "average_processing_time_ms": avg_time,
            "average_pages_per_pdf": avg_pages,
            "success_rate": success_rate
        }

    def get_uptime(self):
        return (datetime.now() - self.start_time).total_seconds()
EOF
```

### 5.3 Create ocr_service.py

```bash
cat > /workspace/deepseek-ocr-server/app/ocr_service.py << 'EOF'
"""DeepSeek-OCR Service using vLLM or transformers"""

import os
import torch
import logging
from typing import Optional, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

class DeepSeekOCRService:
    """Service for running DeepSeek-OCR model"""

    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.85):
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.model = None
        self.processor = None
        self.use_vllm = False

        self._load_model()

    def _load_model(self):
        """Load the model using vLLM or transformers"""
        try:
            # Try vLLM first
            from vllm import LLM, SamplingParams
            logger.info("Loading model with vLLM...")
            self.model = LLM(
                model=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True
            )
            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=4096
            )
            self.use_vllm = True
            logger.info("Model loaded with vLLM backend")

        except Exception as e:
            logger.warning(f"vLLM loading failed: {e}")
            logger.info("Falling back to transformers...")

            # Fallback to transformers
            from transformers import AutoModelForVision2Seq, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Model loaded with transformers backend")

    async def process_image(self, image_path: str, prompt: str) -> str:
        """Process an image and return markdown"""
        try:
            if self.use_vllm:
                # vLLM processing
                outputs = self.model.generate(
                    prompts=[{"prompt": prompt, "multi_modal_data": {"image": image_path}}],
                    sampling_params=self.sampling_params
                )
                return outputs[0].outputs[0].text
            else:
                # Transformers processing
                image = Image.open(image_path)
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=4096)

                return self.processor.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the service"""
        cuda_available = torch.cuda.is_available()
        status = {
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "cuda_available": cuda_available,
            "model_path": self.model_path,
            "backend": "vLLM" if self.use_vllm else "transformers"
        }

        if cuda_available:
            gpu_props = torch.cuda.get_device_properties(0)
            status["gpu_memory_total"] = f"{gpu_props.total_memory / 1e9:.1f}GB"
            status["gpu_memory_free"] = f"{(gpu_props.total_memory - torch.cuda.memory_allocated()) / 1e9:.1f}GB"

        return status
EOF
```

### 5.4 Create pdf_processor.py

```bash
cat > /workspace/deepseek-ocr-server/app/pdf_processor.py << 'EOF'
"""PDF processing utilities"""

import os
import tempfile
import logging
from typing import List
from pdf2image import convert_from_path
import aiohttp
import asyncio

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF to image conversion"""

    def __init__(self):
        self.resolution_map = {
            "tiny": 100,
            "small": 150,
            "base": 200,
            "large": 300,
            "gundam": 400
        }

    def pdf_to_images(self, pdf_path: str, resolution: str = "base") -> List[str]:
        """Convert PDF to images"""
        dpi = self.resolution_map.get(resolution, 200)
        logger.info(f"Converting PDF with resolution: {resolution} ({dpi} DPI)")

        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)

        # Save images to temp files
        temp_paths = []
        for i, image in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, "PNG")
                temp_paths.append(tmp.name)
                logger.debug(f"Saved page {i+1} to {tmp.name}")

        return temp_paths

    async def download_pdf(self, url: str) -> str:
        """Download PDF from URL"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download PDF: HTTP {response.status}")

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    content = await response.read()
                    tmp.write(content)
                    logger.info(f"Downloaded PDF to {tmp.name} ({len(content)} bytes)")
                    return tmp.name
EOF
```

### 5.5 Create __init__.py

```bash
cat > /workspace/deepseek-ocr-server/app/__init__.py << 'EOF'
"""DeepSeek-OCR Server"""
__version__ = "1.0.0"
EOF
```

---

## Step 6: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (for fast inference)
pip install vllm

# If vLLM fails, try:
# pip install vllm --no-build-isolation

# Install FastAPI and server dependencies
pip install fastapi uvicorn python-multipart aiofiles

# Install PDF processing libraries
pip install pdf2image PyMuPDF Pillow

# Install ML libraries
pip install transformers accelerate huggingface-hub

# Install fast download support
pip install hf-transfer
```

---

## Step 7: Download the DeepSeek-OCR Model

This downloads ~10GB and takes 5-15 minutes:

```bash
# Enable fast downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download the model
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR

# Verify download (should show ~8-10GB)
du -sh /workspace/models/deepseek-ai/DeepSeek-OCR
```

---

## Step 8: Create Helper Scripts

### Create Start Script

```bash
cat > /workspace/start_server.sh << 'EOF'
#!/bin/bash
cd /workspace/deepseek-ocr-server
export MODEL_PATH="/workspace/models/deepseek-ai/DeepSeek-OCR"
export PORT=7777
export HOST="0.0.0.0"
export GPU_MEMORY_UTILIZATION=0.85

clear
echo "============================================================"
echo "    Starting DEEPSEEK-OCR SERVER"
echo "    Port: 7777 | Backend: vLLM | Mode: Foreground"
echo "============================================================"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 7777 --log-level info
EOF
chmod +x /workspace/start_server.sh
```

### Create Stop Script

```bash
cat > /workspace/stop_server.sh << 'EOF'
#!/bin/bash
echo "Stopping DeepSeek-OCR server..."
pkill -f "uvicorn.*7777" || echo "No server process found"
fuser -k 7777/tcp 2>/dev/null || echo "Port 7777 is free"
echo "Server stopped"
EOF
chmod +x /workspace/stop_server.sh
```

### Create Check Script

```bash
cat > /workspace/check_server.sh << 'EOF'
#!/bin/bash
echo "============================================================"
echo "    DEEPSEEK-OCR SERVER STATUS CHECK"
echo "============================================================"
echo ""
echo "Process Status:"
if pgrep -f "uvicorn.*7777" > /dev/null; then
    echo "  ✓ Server is running (PID: $(pgrep -f 'uvicorn.*7777'))"
else
    echo "  ✗ Server not running"
fi
echo ""
echo "Port Status:"
if netstat -tuln 2>/dev/null | grep -q ":7777 "; then
    echo "  ✓ Port 7777 is listening"
else
    echo "  ✗ Port 7777 not active"
fi
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
echo ""
echo "Health Check:"
if curl -s http://localhost:7777/health > /dev/null 2>&1; then
    echo "  ✓ Server responding"
    curl -s http://localhost:7777/health | python3 -m json.tool
else
    echo "  ✗ Server not responding"
fi
EOF
chmod +x /workspace/check_server.sh
```

---

## Step 9: Start the Server

```bash
# Run the start script
bash /workspace/start_server.sh
```

You should see:
```
============================================================
    Starting DEEPSEEK-OCR SERVER
    Port: 7777 | Backend: vLLM | Mode: Foreground
============================================================

Press Ctrl+C to stop the server

============================================================
    DEEPSEEK-OCR SERVER
    PDF to Markdown Conversion
    Port: 7777 | Backend: vLLM
============================================================

INFO:     Loading model from: /workspace/models/deepseek-ai/DeepSeek-OCR
...
✓ Server ready at http://0.0.0.0:7777

INFO:     Uvicorn running on http://0.0.0.0:7777 (Press CTRL+C to quit)
```

---

## Step 10: Test the Server (New Terminal)

Open a new terminal in Runpod and test:

```bash
# Check health
curl http://localhost:7777/health

# Should return:
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_memory_free": "20.5GB",
  "gpu_memory_total": "24.0GB",
  "backend": "vLLM"
}

# Check status
curl http://localhost:7777/api/v1/status

# Test with a PDF (if you have one)
curl -X POST -F "file=@test.pdf" http://localhost:7777/api/v1/ocr/pdf
```

---

## Step 11: Access from Your Local Machine

1. Find your Runpod URL in the dashboard
2. Test from your local machine:

```bash
# Replace with your actual Runpod URL
curl http://greasy-lime-clownfish.runpod.io:7777/health

# Use with Go client
./deepseek-client -workers 4 -scan /path/to/pdfs \
  -api http://greasy-lime-clownfish.runpod.io:7777
```

---

## Summary

That's the complete setup! The server is now:
- Running on port **7777**
- Using **vLLM** backend for fast inference
- Showing a simple **ASCII banner**
- Running in **foreground mode** (shows logs)

### Quick Reference Commands

```bash
# Start server
bash /workspace/start_server.sh

# Check status
bash /workspace/check_server.sh

# Stop server
bash /workspace/stop_server.sh

# View logs (if running)
# Logs appear directly in terminal when running in foreground

# Test health
curl http://localhost:7777/health
```

### Troubleshooting

If something goes wrong:

```bash
# Check GPU
nvidia-smi

# Check if model downloaded
ls -la /workspace/models/deepseek-ai/DeepSeek-OCR/

# Kill any stuck processes
pkill -f uvicorn

# Check Python packages
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import vllm; print('vLLM OK')"
python3 -c "import fastapi; print('FastAPI OK')"
```

The setup is complete! The server should be running with your custom ASCII banner on port 7777.