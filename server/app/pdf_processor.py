"""
PDF Processor Module
Handles PDF to image conversion and URL downloads
"""

import os
import tempfile
import logging
import aiohttp
import asyncio
from typing import List, Optional
from pathlib import Path
import shutil

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not available - PDF processing will fail")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available - using pdf2image as fallback")

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handles PDF processing operations:
    - PDF to image conversion
    - PDF downloading from URLs
    - Image resolution management
    """

    # Resolution presets for DeepSeek-OCR
    RESOLUTION_PRESETS = {
        "tiny": 150,     # Fast but lower quality
        "small": 200,    # Good balance for simple documents
        "base": 300,     # Standard quality (default)
        "large": 400,    # High quality for detailed documents
        "gundam": 600    # Maximum quality (named after DeepSeek's preset)
    }

    def __init__(
        self,
        default_dpi: int = 300,
        temp_dir: Optional[str] = None,
        use_pymupdf: bool = True
    ):
        """
        Initialize PDF processor

        Args:
            default_dpi: Default DPI for image conversion
            temp_dir: Directory for temporary files (uses system temp if None)
            use_pymupdf: Whether to prefer PyMuPDF over pdf2image
        """
        self.default_dpi = default_dpi
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.use_pymupdf = use_pymupdf and PYMUPDF_AVAILABLE

        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"PDF Processor initialized (backend: {'PyMuPDF' if self.use_pymupdf else 'pdf2image'})")

    def pdf_to_images(
        self,
        pdf_path: str,
        resolution: str = "base",
        output_format: str = "png"
    ) -> List[str]:
        """
        Convert PDF to images

        Args:
            pdf_path: Path to PDF file
            resolution: Resolution preset name or DPI value
            output_format: Output image format (png, jpg)

        Returns:
            List of paths to generated images
        """
        # Get DPI from resolution preset or use as-is if numeric
        if resolution in self.RESOLUTION_PRESETS:
            dpi = self.RESOLUTION_PRESETS[resolution]
        elif resolution.isdigit():
            dpi = int(resolution)
        else:
            dpi = self.default_dpi
            logger.warning(f"Unknown resolution '{resolution}', using default {dpi} DPI")

        logger.info(f"Converting PDF to images at {dpi} DPI (resolution: {resolution})")

        if self.use_pymupdf:
            return self._pdf_to_images_pymupdf(pdf_path, dpi, output_format)
        else:
            return self._pdf_to_images_pdf2image(pdf_path, dpi, output_format)

    def _pdf_to_images_pymupdf(
        self,
        pdf_path: str,
        dpi: int,
        output_format: str
    ) -> List[str]:
        """Convert PDF to images using PyMuPDF"""
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF is not installed")

        image_paths = []

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)

            # Calculate zoom factor from DPI (default PyMuPDF is 72 DPI)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            # Convert each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Render page to pixmap
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)

                # Generate temp file path
                temp_image = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{output_format}",
                    dir=self.temp_dir,
                    prefix=f"page_{page_num + 1}_"
                )
                temp_image.close()

                # Save image
                if output_format.lower() == "png":
                    pixmap.save(temp_image.name)
                else:
                    # For JPEG, we need to convert from PNG
                    pixmap.save(temp_image.name)

                image_paths.append(temp_image.name)

                logger.debug(f"Converted page {page_num + 1}/{len(pdf_document)}")

            pdf_document.close()

            logger.info(f"Successfully converted {len(image_paths)} pages")
            return image_paths

        except Exception as e:
            logger.error(f"Error converting PDF with PyMuPDF: {str(e)}")
            # Clean up any created images
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
            raise

    def _pdf_to_images_pdf2image(
        self,
        pdf_path: str,
        dpi: int,
        output_format: str
    ) -> List[str]:
        """Convert PDF to images using pdf2image"""
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image is not installed")

        image_paths = []

        try:
            # Convert PDF to PIL Images
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt=output_format.upper(),
                thread_count=4,  # Use multiple threads for faster conversion
                use_pdftocairo=True,  # Use pdftocairo if available (faster)
                output_folder=self.temp_dir
            )

            # Save each image
            for idx, image in enumerate(images, 1):
                # Generate temp file path
                temp_image = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{output_format}",
                    dir=self.temp_dir,
                    prefix=f"page_{idx}_"
                )
                temp_image.close()

                # Save image
                image.save(temp_image.name, output_format.upper())
                image_paths.append(temp_image.name)

                logger.debug(f"Converted page {idx}/{len(images)}")

            logger.info(f"Successfully converted {len(image_paths)} pages")
            return image_paths

        except Exception as e:
            logger.error(f"Error converting PDF with pdf2image: {str(e)}")
            # Clean up any created images
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
            raise

    async def download_pdf(self, url: str, timeout: int = 300) -> str:
        """
        Download PDF from URL

        Args:
            url: URL to download PDF from
            timeout: Download timeout in seconds

        Returns:
            Path to downloaded PDF file
        """
        logger.info(f"Downloading PDF from: {url}")

        # Generate temp file for PDF
        temp_pdf = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
            dir=self.temp_dir,
            prefix="download_"
        )
        temp_pdf.close()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    # Check response status
                    response.raise_for_status()

                    # Check content type
                    content_type = response.headers.get('Content-Type', '')
                    if 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                        logger.warning(f"URL may not be a PDF (Content-Type: {content_type})")

                    # Get file size if available
                    file_size = response.headers.get('Content-Length')
                    if file_size:
                        logger.info(f"Downloading {int(file_size) / 1024 / 1024:.2f} MB")

                    # Download in chunks
                    with open(temp_pdf.name, 'wb') as f:
                        downloaded = 0
                        chunk_size = 8192

                        async for chunk in response.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress for large files
                            if file_size and downloaded % (chunk_size * 100) == 0:
                                progress = (downloaded / int(file_size)) * 100
                                logger.debug(f"Download progress: {progress:.1f}%")

            logger.info(f"Successfully downloaded PDF to: {temp_pdf.name}")
            return temp_pdf.name

        except asyncio.TimeoutError:
            os.remove(temp_pdf.name)
            raise Exception(f"Download timeout after {timeout} seconds")
        except Exception as e:
            os.remove(temp_pdf.name)
            logger.error(f"Error downloading PDF: {str(e)}")
            raise

    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validate if file is a valid PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if valid PDF, False otherwise
        """
        try:
            if self.use_pymupdf:
                # Try to open with PyMuPDF
                doc = fitz.open(pdf_path)
                is_valid = doc.page_count > 0
                doc.close()
                return is_valid
            else:
                # Try to get page count with pdf2image
                from pdf2image import pdfinfo_from_path
                info = pdfinfo_from_path(pdf_path)
                return info.get('Pages', 0) > 0

        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False

    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get PDF information

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with PDF information
        """
        info = {}

        try:
            if self.use_pymupdf:
                doc = fitz.open(pdf_path)
                info = {
                    "pages": doc.page_count,
                    "metadata": doc.metadata,
                    "is_encrypted": doc.is_encrypted,
                    "file_size": os.path.getsize(pdf_path)
                }
                doc.close()
            else:
                from pdf2image import pdfinfo_from_path
                pdf_info = pdfinfo_from_path(pdf_path)
                info = {
                    "pages": pdf_info.get('Pages', 0),
                    "title": pdf_info.get('Title', ''),
                    "author": pdf_info.get('Author', ''),
                    "subject": pdf_info.get('Subject', ''),
                    "creator": pdf_info.get('Creator', ''),
                    "producer": pdf_info.get('Producer', ''),
                    "creation_date": pdf_info.get('CreationDate', ''),
                    "file_size": os.path.getsize(pdf_path)
                }

        except Exception as e:
            logger.error(f"Error getting PDF info: {str(e)}")
            info = {"error": str(e)}

        return info

    def cleanup_temp_files(self, age_hours: int = 24):
        """
        Clean up old temporary files

        Args:
            age_hours: Delete files older than this many hours
        """
        import time
        from datetime import datetime, timedelta

        cutoff_time = time.time() - (age_hours * 3600)
        cleaned = 0

        try:
            for file_path in Path(self.temp_dir).glob("*"):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned += 1

            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} old temporary files")

        except Exception as e:
            logger.error(f"Error cleaning temp files: {str(e)}")

    def estimate_processing_time(self, pdf_path: str, pages_per_second: float = 2.0) -> float:
        """
        Estimate processing time for a PDF

        Args:
            pdf_path: Path to PDF file
            pages_per_second: Expected processing rate

        Returns:
            Estimated time in seconds
        """
        try:
            info = self.get_pdf_info(pdf_path)
            pages = info.get("pages", 0)
            return pages / pages_per_second
        except:
            return 0.0