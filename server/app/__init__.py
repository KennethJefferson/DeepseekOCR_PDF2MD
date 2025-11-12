"""
DeepSeek-OCR Server Application
"""

__version__ = "1.0.0"
__author__ = "DeepSeek-OCR Server"

from .models import *
from .ocr_service import DeepSeekOCRService
from .pdf_processor import PDFProcessor