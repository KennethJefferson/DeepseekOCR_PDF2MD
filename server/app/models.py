"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from datetime import datetime


class PageResult(BaseModel):
    """Result for a single PDF page"""
    page_number: int = Field(..., description="Page number (1-indexed)")
    markdown: str = Field(..., description="Markdown content of the page")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class PDFResponse(BaseModel):
    """Response for PDF processing"""
    success: bool = Field(..., description="Whether processing was successful")
    filename: str = Field(..., description="Name of the processed file")
    total_pages: int = Field(..., description="Total number of pages processed")
    pages: List[PageResult] = Field(..., description="Results for each page")
    total_processing_time_ms: int = Field(..., description="Total processing time in milliseconds")


class PDFURLRequest(BaseModel):
    """Request for processing PDF from URL"""
    url: HttpUrl = Field(..., description="URL of the PDF to process")
    resolution: str = Field(
        default="base",
        description="Image resolution for OCR (tiny, small, base, large, gundam)"
    )


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    cuda_available: bool = Field(..., description="Whether CUDA is available")
    gpu_memory_free: str = Field(..., description="Free GPU memory")
    gpu_memory_total: Optional[str] = Field(None, description="Total GPU memory")
    model_path: Optional[str] = Field(None, description="Path to the loaded model")
    uptime_seconds: Optional[float] = Field(None, description="Server uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class ProcessingStats:
    """Statistics tracking for the server"""

    def __init__(self):
        self.start_time = datetime.now()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_pages = 0
        self.total_processing_time_ms = 0

    def record_success(self, pages: int, processing_time_ms: float):
        """Record a successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_pages += pages
        self.total_processing_time_ms += processing_time_ms

    def record_failure(self):
        """Record a failed request"""
        self.total_requests += 1
        self.failed_requests += 1

    def get_stats(self) -> dict:
        """Get current statistics"""
        avg_time = 0
        if self.successful_requests > 0:
            avg_time = self.total_processing_time_ms / self.successful_requests

        avg_pages = 0
        if self.successful_requests > 0:
            avg_pages = self.total_pages / self.successful_requests

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_pages_processed": self.total_pages,
            "average_processing_time_ms": round(avg_time, 2),
            "average_pages_per_pdf": round(avg_pages, 2),
            "success_rate": round(
                (self.successful_requests / max(self.total_requests, 1)) * 100, 2
            )
        }

    def get_uptime(self) -> float:
        """Get server uptime in seconds"""
        return (datetime.now() - self.start_time).total_seconds()


class StatusResponse(BaseModel):
    """Server status response"""
    status: str = Field(..., description="Server status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    stats: dict = Field(..., description="Processing statistics")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")