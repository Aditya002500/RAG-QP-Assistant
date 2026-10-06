"""PDF Text extraction module with PyMuPDF and Tesseract OCR fallback.
Logs OCR rate per file as specified in FR-2.
"""

import io
import logging
import os
from pathlib import Path
from typing import Any

import pymupdf
import pytesseract
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure Tesseract path if set in environment
tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd and Path(tesseract_cmd).exists():
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
elif Path("/opt/homebrew/bin/tesseract").exists():
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


class PDFExtractor:
    """Extracts text from PDF files using PyMuPDF, with OCR fallback for scanned pages."""

    def __init__(self, min_char_threshold: int = 50, ocr_dpi: int = 300):
        self.min_char_threshold = min_char_threshold
        self.ocr_dpi = ocr_dpi

    def extract_page_ocr(self, page: pymupdf.Page) -> str:
        """Render page as image and extract text via pytesseract."""
        try:
            zoom = self.ocr_dpi / 72.0
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.warning("OCR extraction failed on page %d: %s", page.number + 1, e)
            return ""

    def extract_document(self, file_path: Path) -> list[dict[str, Any]]:
        """Extract pages from a single PDF document.

        Returns list of page dicts with keys:
        - source_file: filename
        - page: 1-indexed page number
        - text: extracted text
        - extraction: 'text' or 'ocr'
        """
        doc = pymupdf.open(str(file_path))
        total_pages = len(doc)
        ocr_count = 0
        pages_data: list[dict[str, Any]] = []

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            raw_text = page.get_text().strip()

            if len(raw_text) >= self.min_char_threshold:
                extraction_type = "text"
                page_text = raw_text
            else:
                logger.info(
                    "Page %d of %s has only %d chars (< %d). Triggering OCR fallback...",
                    page_num,
                    file_path.name,
                    len(raw_text),
                    self.min_char_threshold,
                )
                ocr_text = self.extract_page_ocr(page)
                extraction_type = "ocr"
                page_text = ocr_text
                ocr_count += 1

            pages_data.append(
                {
                    "source_file": file_path.name,
                    "page": page_num,
                    "text": page_text,
                    "extraction": extraction_type,
                }
            )

        doc.close()
        ocr_rate = (ocr_count / total_pages) if total_pages > 0 else 0.0
        logger.info(
            "Extracted %s: %d pages (OCR count: %d, OCR rate: %.2f%%)",
            file_path.name,
            total_pages,
            ocr_count,
            ocr_rate * 100,
        )
        return pages_data

    def extract_directory(self, dir_path: Path) -> list[dict[str, Any]]:
        """Extract all PDF documents in a directory."""
        all_pages: list[dict[str, Any]] = []
        pdf_files = sorted(dir_path.glob("*.pdf"))
        logger.info("Found %d PDF files in %s", len(pdf_files), dir_path)
        for pdf_file in pdf_files:
            pages = self.extract_document(pdf_file)
            all_pages.extend(pages)
        return all_pages
