"""
ingesttion/ingest.py
Chunking / PDF preprocessing code kept in a dedicated folder (one-time job).
Functions:
- crop_pdf_footer: crop footer and save cleaned PDF
- chunk_pdf_with_metadata: extract text chunks and write a JSON file
"""
import json
from pathlib import Path
from typing import List, Dict

import pdfplumber
from PyPDF2 import PdfReader, PdfWriter

from core.logging_config import logger
from core.config import PATHS


def crop_pdf_footer(input_pdf: str, output_pdf: str, footer_fraction: float = 0.15):
    """
    Crops the footer from each page of the PDF and saves the cropped PDF.

    :param input_pdf: Path to the input PDF file.
    :param output_pdf: Path to the output cropped PDF file.
    :param footer_fraction: Fraction of the page height to be cropped from the bottom.
    """
    # Read the input PDF
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    for page in reader.pages[8:-3]:
        # Get the dimensions of the current page
        media_box = page.mediabox
        width = float(media_box.width)
        height = float(media_box.height)

        # Calculate the new dimensions after cropping
        new_lower_bound = float(media_box.lower_left[1]) + (footer_fraction * height)

        # Update the mediabox to crop the footer
        page.mediabox.lower_left = (media_box.lower_left[0], new_lower_bound)
        page.mediabox.lower_right = (media_box.lower_right[0], new_lower_bound)

        # Add the cropped page to the writer
        writer.add_page(page)

    return writer



def chunk_pdf_with_metadata(pdf_path, chunk_output_path=PATHS.OUTPUT_CHUNK_PATH):
    """
    Reads a PDF, extracts chunks (Articles or paragraphs), and attaches metadata.
    If Articles are not present we will chunk the text through paragraphs else through articles.
    Args:
        pdf_path (str): Path to the input PDF file.

    Metadata: will contain the page number, PART header, section header and article heading

    Returns:
        list[dict]: List of chunks with metadata.
    """
    chunks = []
    current_part = 0
    current_section = 0
    current_article = 0
    is_content = False

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            lines = text.splitlines()
            current_chunk = []

            for line in lines:
                # Detect Parts
                if line.strip().replace(' ','').startswith("PART") and ('section' not in line.strip().lower() and 'article' not in line.strip().lower()):
                    current_part = line.strip()
                    continue

                # Detect Sections
                if line.strip().replace(' ','').startswith("Section"):
                    current_section = line.strip()
                    continue

                # Detect Articles
                if line.strip().replace(' ','').startswith("Article"):
                    # Save the previous chunk if it exists
                    if current_chunk:
                        chunks.append({
                            "Text": " ".join(current_chunk).strip(),
                            "Metadata": {
                              "PART_DETAILS": current_part,
                              "SECTION_DETAILS": current_section,
                              "ARTICLE_DETAILS": current_article,
                              "PAGE_NO": page_number,
                          }
                        })
                        current_chunk = []

                    # Update current article
                    current_article = line.strip()
                    continue

                # Add content to the current chunk
                current_chunk.append(line)

            # Save any remaining content on the page
            if current_chunk:
                chunks.append({
                    "Text": " ".join(current_chunk).strip(),
                    "Metadata": {
                        "PART_DETAILS": current_part,
                        "SECTION_DETAILS": current_section,
                        "ARTICLE_DETAILS": current_article,
                        "PAGE_NO": page_number,
                    }
                })
                current_chunk = []
    return chunks
