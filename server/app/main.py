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

# Import our modules (will be created next)
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
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances (initialized on startup)
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
        model_path = os.getenv("MODEL_PATH", "/app/models/deepseek-ai/DeepSeek-OCR")
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
    # Add any cleanup code here


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
    """
    Process a PDF file and convert it to markdown

    Args:
        file: PDF file to process
        resolution: Image resolution (tiny, small, base, large, gundam)

    Returns:
        PDFResponse with markdown content for each page
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    if ocr_service is None or pdf_processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    temp_pdf_path = None
    temp_image_paths = []

    try:
        start_time = datetime.now()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_pdf_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        logger.info(f"Processing PDF: {file.filename} ({len(content)} bytes)")

        # Convert PDF to images
        temp_image_paths = pdf_processor.pdf_to_images(
            temp_pdf_path,
            resolution=resolution
        )

        logger.info(f"Converted PDF to {len(temp_image_paths)} images")

        # Process each page with OCR
        pages = []
        for idx, image_path in enumerate(temp_image_paths, 1):
            page_start = datetime.now()

            # Process image with DeepSeek-OCR
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

        # Calculate total processing time
        total_time = (datetime.now() - start_time).total_seconds() * 1000

        # Update statistics
        stats.record_success(len(pages), total_time)

        response = PDFResponse(
            success=True,
            filename=file.filename,
            total_pages=len(pages),
            pages=pages,
            total_processing_time_ms=int(total_time)
        )

        logger.info(f"Successfully processed {file.filename}: {len(pages)} pages in {total_time:.0f}ms")

        return response

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}\n{traceback.format_exc()}")
        stats.record_failure()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary files
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        for image_path in temp_image_paths:
            if os.path.exists(image_path):
                os.remove(image_path)


@app.post("/api/v1/ocr/pdf-url", response_model=PDFResponse, tags=["OCR"])
async def process_pdf_url(request: PDFURLRequest):
    """
    Process a PDF from URL and convert it to markdown

    Args:
        request: PDFURLRequest containing URL and resolution

    Returns:
        PDFResponse with markdown content for each page
    """
    if ocr_service is None or pdf_processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    temp_pdf_path = None
    temp_image_paths = []

    try:
        start_time = datetime.now()

        # Download PDF from URL
        logger.info(f"Downloading PDF from: {request.url}")
        temp_pdf_path = await pdf_processor.download_pdf(request.url)

        # Get filename from URL
        filename = request.url.split('/')[-1] or "document.pdf"

        # Convert PDF to images
        temp_image_paths = pdf_processor.pdf_to_images(
            temp_pdf_path,
            resolution=request.resolution
        )

        logger.info(f"Converted PDF to {len(temp_image_paths)} images")

        # Process each page with OCR
        pages = []
        for idx, image_path in enumerate(temp_image_paths, 1):
            page_start = datetime.now()

            # Process image with DeepSeek-OCR
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

        # Calculate total processing time
        total_time = (datetime.now() - start_time).total_seconds() * 1000

        # Update statistics
        stats.record_success(len(pages), total_time)

        response = PDFResponse(
            success=True,
            filename=filename,
            total_pages=len(pages),
            pages=pages,
            total_processing_time_ms=int(total_time)
        )

        logger.info(f"Successfully processed {filename}: {len(pages)} pages in {total_time:.0f}ms")

        return response

    except Exception as e:
        logger.error(f"Error processing PDF URL: {str(e)}\n{traceback.format_exc()}")
        stats.record_failure()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary files
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


# Run the server if executed directly
if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))

    # Run with uvicorn
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )