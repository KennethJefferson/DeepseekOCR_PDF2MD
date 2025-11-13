# Complete Terminal Setup Commands for Runpod

This file contains all the commands you can copy and paste directly into your Runpod terminal to set up the DeepSeek-OCR server. These are designed to be run sequentially in a terminal session.

**Important:** This assumes you've already uploaded the 5 Python files to `/workspace/deepseek-ocr-server/app/`

---

## Step 1: Initial Setup and Directory Creation

```bash
# Navigate to workspace and create directories
cd /workspace
mkdir -p /workspace/deepseek-ocr-server/app
mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/temp

# Verify directories
ls -la /workspace/
```

---

## Step 2: Create All Server Files via Terminal

If you prefer to create files directly via terminal instead of uploading, use these commands:

### File 1: main.py

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
startup_time: Optional[datetime] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global ocr_service, pdf_processor, startup_time

    # Display ASCII banner
    print("\n" + "="*60)
    print("    DEEPSEEK-OCR SERVER")
    print("    PDF to Markdown Conversion")
    print("    Port: {} | Backend: vLLM".format(os.getenv("PORT", "7777")))
    print("="*60 + "\n")

    try:
        # Record startup time
        startup_time = datetime.now()

        # Get model path from environment
        model_path = os.getenv("MODEL_PATH", "/workspace/models/deepseek-ai/DeepSeek-OCR")

        logger.info(f"Loading model from: {model_path}")

        # Initialize OCR service
        ocr_service = DeepSeekOCRService(model_path)

        # Initialize PDF processor
        pdf_processor = PDFProcessor()

        logger.info("Server startup complete!")
        print(f"\n✓ Server ready at http://0.0.0.0:{os.getenv('PORT', '7777')}\n")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global ocr_service
    if ocr_service:
        ocr_service.cleanup()

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "DeepSeek-OCR Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "convert_pdf": "/api/v1/ocr/pdf",
            "convert_pdf_url": "/api/v1/ocr/pdf-url",
            "status": "/api/v1/status"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_memory_free = torch.cuda.mem_get_info()[0] / (1024**3)
            gpu_memory_total = torch.cuda.mem_get_info()[1] / (1024**3)
        else:
            gpu_memory_free = 0
            gpu_memory_total = 0

        uptime_seconds = (datetime.now() - startup_time).total_seconds() if startup_time else 0

        return HealthResponse(
            status="healthy",
            model_loaded=ocr_service is not None,
            cuda_available=cuda_available,
            gpu_memory_free=f"{gpu_memory_free:.1f}GB",
            gpu_memory_total=f"{gpu_memory_total:.1f}GB",
            model_path=os.getenv("MODEL_PATH", "/workspace/models/deepseek-ai/DeepSeek-OCR"),
            backend=ocr_service.backend if ocr_service else "not_loaded",
            uptime_seconds=uptime_seconds
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            cuda_available=False,
            error=str(e)
        )

@app.get("/api/v1/status")
async def status():
    """Detailed status endpoint"""
    return await health_check()

@app.post("/api/v1/ocr/pdf", response_model=PDFResponse)
async def convert_pdf(
    file: UploadFile = File(...),
    resolution: str = "base"
):
    """
    Convert PDF to Markdown

    Args:
        file: PDF file to convert
        resolution: Image resolution (low/base/high)

    Returns:
        PDFResponse with markdown content and processing stats
    """
    if not ocr_service:
        raise HTTPException(status_code=503, detail="OCR service not initialized")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    temp_path = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        logger.info(f"Processing PDF: {file.filename} ({len(content)} bytes)")

        # Convert PDF to images
        images = pdf_processor.pdf_to_images(temp_path, resolution)

        if not images:
            raise HTTPException(status_code=400, detail="No pages could be extracted from PDF")

        # Process each page
        pages = []
        total_tokens = 0

        for i, img in enumerate(images, 1):
            logger.info(f"Processing page {i}/{len(images)}")

            markdown = ocr_service.process_image(img)
            tokens = len(markdown) // 4  # Rough token estimate

            pages.append(PageResult(
                page_number=i,
                markdown=markdown,
                tokens_used=tokens
            ))
            total_tokens += tokens

        # Combine all pages
        full_markdown = "\n\n---\n\n".join([p.markdown for p in pages])

        response = PDFResponse(
            filename=file.filename,
            markdown=full_markdown,
            pages=pages,
            stats=ProcessingStats(
                total_pages=len(pages),
                total_tokens=total_tokens,
                processing_time=(datetime.now() - startup_time).total_seconds(),
                resolution=resolution
            )
        )

        logger.info(f"Successfully processed {file.filename}: {len(pages)} pages, {total_tokens} tokens")
        return response

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/api/v1/ocr/pdf-url", response_model=PDFResponse)
async def convert_pdf_from_url(request: PDFURLRequest):
    """Convert PDF from URL to Markdown"""
    # Implementation for URL-based conversion
    raise HTTPException(status_code=501, detail="URL conversion not yet implemented")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7777"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
EOF
```

### File 2: models.py

```bash
cat > /workspace/deepseek-ocr-server/app/models.py << 'EOF'
"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class PageResult(BaseModel):
    """Result for a single page"""
    page_number: int
    markdown: str
    tokens_used: int = 0

class ProcessingStats(BaseModel):
    """Processing statistics"""
    total_pages: int
    total_tokens: int
    processing_time: float
    resolution: str = "base"

class PDFResponse(BaseModel):
    """Response for PDF conversion"""
    filename: str
    markdown: str
    pages: List[PageResult]
    stats: ProcessingStats

class PDFURLRequest(BaseModel):
    """Request for URL-based PDF conversion"""
    url: str = Field(..., description="URL to PDF file")
    resolution: str = Field("base", description="Resolution: low/base/high")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    cuda_available: bool
    gpu_memory_free: Optional[str] = None
    gpu_memory_total: Optional[str] = None
    model_path: Optional[str] = None
    backend: Optional[str] = None
    uptime_seconds: Optional[float] = None
    error: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
EOF
```

### File 3: ocr_service.py

```bash
cat > /workspace/deepseek-ocr-server/app/ocr_service.py << 'EOF'
"""
DeepSeek-OCR Service
Handles model loading and inference with vLLM optimization
"""

import os
import logging
import torch
from PIL import Image
from typing import Optional, Union
import traceback

logger = logging.getLogger(__name__)

class DeepSeekOCRService:
    """Service for DeepSeek-OCR model inference"""

    def __init__(self, model_path: str):
        """Initialize OCR service with model"""
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.backend = "not_loaded"
        self.vllm_model = None

        # Try vLLM first, fall back to transformers
        if not self._load_with_vllm():
            self._load_with_transformers()

    def _load_with_vllm(self) -> bool:
        """Try to load model with vLLM for optimized inference"""
        try:
            from vllm import LLM, SamplingParams

            logger.info("Loading model with vLLM...")

            # vLLM configuration
            gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))

            self.vllm_model = LLM(
                model=self.model_path,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=4096,
            )

            # Still need processor for image preprocessing
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.backend = "vLLM"
            logger.info("Model loaded with vLLM backend")
            return True

        except Exception as e:
            logger.warning(f"vLLM loading failed: {e}")
            logger.warning("Falling back to transformers...")
            return False

    def _load_with_transformers(self):
        """Load model with standard transformers"""
        try:
            from transformers import AutoModel, AutoProcessor

            logger.info("Loading model with transformers...")

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cuda"
            )

            self.model.eval()
            self.backend = "transformers"
            logger.info("Model loaded with transformers backend")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            raise

    def process_image(self, image: Union[Image.Image, str]) -> str:
        """
        Process image and extract text as markdown

        Args:
            image: PIL Image or path to image

        Returns:
            Markdown formatted text
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError("Input must be PIL Image or image path")

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Create conversation
            conversation = [{
                "role": "user",
                "content": "<image>\nConvert this image to markdown format. Preserve all text, formatting, tables, and structure."
            }]

            if self.backend == "vLLM":
                return self._process_with_vllm(image, conversation)
            else:
                return self._process_with_transformers(image, conversation)

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            logger.error(traceback.format_exc())
            return f"Error: Failed to process image - {str(e)}"

    def _process_with_vllm(self, image: Image.Image, conversation: list) -> str:
        """Process with vLLM backend"""
        from vllm import SamplingParams

        # Prepare inputs
        inputs = self.processor(
            text=conversation,
            images=image,
            return_tensors="pt"
        )

        # vLLM sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=4096,
            stop=["<|end_of_text|>"],
        )

        # Generate
        prompt = inputs.get("input_ids", inputs.get("prompt", ""))
        outputs = self.vllm_model.generate(prompt, sampling_params)

        # Extract text
        generated_text = outputs[0].outputs[0].text
        return generated_text.strip()

    def _process_with_transformers(self, image: Image.Image, conversation: list) -> str:
        """Process with transformers backend"""
        # Prepare inputs
        inputs = self.processor(
            text=conversation,
            images=image,
            return_tensors="pt"
        ).to("cuda")

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return generated_text.strip()

    def cleanup(self):
        """Cleanup resources"""
        if self.model:
            del self.model
        if self.vllm_model:
            del self.vllm_model
        if self.processor:
            del self.processor
        torch.cuda.empty_cache()
EOF
```

### File 4: pdf_processor.py

```bash
cat > /workspace/deepseek-ocr-server/app/pdf_processor.py << 'EOF'
"""
PDF Processing utilities
Handles PDF to image conversion
"""

import os
import logging
from typing import List, Optional
from PIL import Image
import tempfile
from pdf2image import convert_from_path
import PyMuPDF  # fitz

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing and conversion"""

    def __init__(self):
        """Initialize PDF processor"""
        self.resolution_map = {
            "low": 150,
            "base": 200,
            "high": 300
        }

    def pdf_to_images(
        self,
        pdf_path: str,
        resolution: str = "base"
    ) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images

        Args:
            pdf_path: Path to PDF file
            resolution: Resolution setting (low/base/high)

        Returns:
            List of PIL Images
        """
        try:
            dpi = self.resolution_map.get(resolution, 200)

            # Try pdf2image first (uses poppler)
            try:
                logger.info(f"Converting PDF with pdf2image at {dpi} DPI...")
                images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    fmt='PNG',
                    thread_count=4,
                    use_pdftocairo=True  # Better quality
                )
                logger.info(f"Converted {len(images)} pages")
                return images

            except Exception as e:
                logger.warning(f"pdf2image failed: {e}, trying PyMuPDF...")

                # Fallback to PyMuPDF
                return self._convert_with_pymupdf(pdf_path, dpi)

        except Exception as e:
            logger.error(f"Failed to convert PDF: {e}")
            raise

    def _convert_with_pymupdf(self, pdf_path: str, dpi: int) -> List[Image.Image]:
        """
        Convert PDF using PyMuPDF as fallback

        Args:
            pdf_path: Path to PDF
            dpi: Resolution in DPI

        Returns:
            List of PIL Images
        """
        import fitz  # PyMuPDF

        images = []
        pdf = fitz.open(pdf_path)

        # Calculate zoom factor from DPI
        zoom = dpi / 72.0  # 72 is default PDF DPI
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(pdf)):
            page = pdf[page_num]

            # Render page to pixmap
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            # Convert to PIL Image
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

            logger.info(f"Converted page {page_num + 1}/{len(pdf)}")

        pdf.close()
        return images

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract raw text from PDF (fallback method)

        Args:
            pdf_path: Path to PDF

        Returns:
            Extracted text
        """
        try:
            import fitz
            pdf = fitz.open(pdf_path)
            text = ""

            for page in pdf:
                text += page.get_text() + "\n"

            pdf.close()
            return text.strip()

        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return ""

# Ensure io is imported for BytesIO
import io
EOF
```

### File 5: __init__.py

```bash
cat > /workspace/deepseek-ocr-server/app/__init__.py << 'EOF'
"""
DeepSeek-OCR Server
"""
__version__ = "1.0.0"
EOF
```

---

## Step 3: Install All Dependencies

```bash
# Update package list
apt-get update

# Install system dependencies
apt-get install -y poppler-utils git curl wget

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (this may take a few minutes)
pip install vllm==0.8.5

# Install FastAPI and web server
pip install fastapi uvicorn[standard] python-multipart aiohttp aiofiles

# Install PDF processing libraries
pip install pdf2image PyMuPDF Pillow

# Install ML libraries
pip install transformers accelerate huggingface-hub pydantic pydantic-settings pyyaml

# Install fast download support
pip install hf-transfer

# Install additional dependencies
pip install addict matplotlib einops timm
```

---

## Step 4: Download the Model

```bash
# Enable fast downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download the DeepSeek-OCR model (this will take 15-20 minutes)
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR

# Verify download
du -sh /workspace/models/deepseek-ai/DeepSeek-OCR
ls -la /workspace/models/deepseek-ai/DeepSeek-OCR/
```

---

## Step 5: Create All Helper Scripts

```bash
# Create start script
cat > /workspace/start_server.sh << 'EOF'
#!/bin/bash
cd /workspace/deepseek-ocr-server || exit 1
export MODEL_PATH="/workspace/models/deepseek-ai/DeepSeek-OCR"
export PORT=7777
export HOST="0.0.0.0"
export GPU_MEMORY_UTILIZATION=0.85
export CUDA_VISIBLE_DEVICES=0

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
echo "Loading model... This may take 30-60 seconds"
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info \
    --access-log
EOF

chmod +x /workspace/start_server.sh

# Create check script
cat > /workspace/check_server.sh << 'EOF'
#!/bin/bash
echo ""
echo "============================================================"
echo "    DEEPSEEK-OCR SERVER STATUS CHECK"
echo "============================================================"
echo ""

echo "1. Process Status:"
if pgrep -f "uvicorn.*7777" > /dev/null; then
    echo "   ✓ Server is running (PID: $(pgrep -f 'uvicorn.*7777'))"
else
    echo "   ✗ Server not running"
fi
echo ""

echo "2. Port Status:"
if netstat -tuln 2>/dev/null | grep -q ":7777 "; then
    echo "   ✓ Port 7777 is listening"
else
    echo "   ✗ Port 7777 not active"
fi
echo ""

echo "3. GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo ""

echo "4. Health Check:"
if curl -s -m 5 http://localhost:7777/health > /dev/null 2>&1; then
    echo "   ✓ Server is responding"
    echo ""
    echo "   Health Response:"
    curl -s http://localhost:7777/health | python3 -m json.tool | head -20
else
    echo "   ✗ Server not responding on http://localhost:7777"
fi
echo ""
echo "============================================================"
EOF

chmod +x /workspace/check_server.sh

# Create stop script
cat > /workspace/stop_server.sh << 'EOF'
#!/bin/bash
echo ""
echo "Stopping DeepSeek-OCR server..."
pkill -f "uvicorn.*7777" && echo "✓ Server process stopped" || echo "No server process found"
fuser -k 7777/tcp 2>/dev/null && echo "✓ Port 7777 freed" || echo "Port 7777 already free"
echo "Server stopped"
echo ""
EOF

chmod +x /workspace/stop_server.sh
```

---

## Step 6: Start the Server

```bash
# Run the start script
bash /workspace/start_server.sh
```

The server will start and display:
- ASCII banner
- GPU information
- Loading progress
- Ready message when complete

---

## Step 7: Test the Server (in new terminal)

```bash
# Quick test
curl http://localhost:7777/health

# Full check
bash /workspace/check_server.sh
```

---

## Quick One-Liner Installation

If you want to run everything at once (after uploading files):

```bash
cd /workspace && \
apt-get update && apt-get install -y poppler-utils git curl wget && \
pip install --upgrade pip setuptools wheel && \
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118 && \
pip install vllm==0.8.5 && \
pip install fastapi uvicorn[standard] python-multipart aiohttp aiofiles pdf2image PyMuPDF Pillow transformers accelerate huggingface-hub pydantic pydantic-settings pyyaml hf-transfer addict matplotlib einops timm && \
export HF_HUB_ENABLE_HF_TRANSFER=1 && \
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir /workspace/models/deepseek-ai/DeepSeek-OCR && \
echo "✓ Installation complete! Run: bash /workspace/start_server.sh"
```