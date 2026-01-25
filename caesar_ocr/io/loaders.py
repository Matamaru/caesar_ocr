"""Document loaders for PDFs and images (bytes -> page images + metadata)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

from pdf2image import convert_from_bytes
from PIL import Image


@dataclass
class PageImage:
    """
    Metadata for a page image.
    """
    page: int
    image: Image.Image
    width: int
    height: int


def load_images_from_bytes(file_bytes: bytes, *, dpi: int = 300) -> List[PageImage]:
    """
    Load images from PDF or image bytes and return page metadata.
    
    :param file_bytes: Bytes of the input PDF or image file.
    :param dpi: DPI for PDF to image conversion.
    :return: List of PageImage objects containing page number, image, width, and height.
    """
    # Check if the input is a PDF by looking at the file signature
    if file_bytes[:4] == b"%PDF":
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        if not pages:
            raise ValueError("Empty PDF")
        out: List[PageImage] = []
        # Convert each PDF page to an image
        for idx, page in enumerate(pages, start=1):
            im = page.convert("RGB")
            width, height = im.size
            # Append page image and metadata as PageImage instance 
            out.append(PageImage(page=idx, image=im, width=width, height=height))
        return out

    # Otherwise, treat as a single image
    im = Image.open(io.BytesIO(file_bytes))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    width, height = im.size
    # Return single page image and metadata as PageImage instance
    return [PageImage(page=1, image=im, width=width, height=height)]
