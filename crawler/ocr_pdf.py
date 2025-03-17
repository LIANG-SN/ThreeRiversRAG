import os
import argparse
from pdf2image import convert_from_path
import pytesseract
import re

"""
PDF Text Extraction Script

This script extracts text from a PDF file using OCR (Optical Character Recognition) with Tesseract.
It first converts each page of the PDF into an image and then applies Tesseract OCR to extract text.
The extracted text is cleaned to remove newlines and URLs before being saved to an output text file.

Usage:
    python pdf_text_extraction.py --pdf_path <input_pdf> --output_file <output_txt>

Arguments:
    --pdf_path      Path to the input PDF file.
    --output_file   Path to save the extracted text.
"""

def clean_text(text):
    """Lower the text, remove newlines, and strip URLs."""
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r'http[s]?://\S+', '', text)
    return text


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using OCR."""
    pages = convert_from_path(pdf_path, dpi=300)  # Adjust DPI based on requirements
    total_text = ""
    
    for page_index, page_image in enumerate(pages):
        text = pytesseract.image_to_string(page_image)
        total_text += text
        
        print(f"--- Page {page_index+1} Text ---")
        print(text)
        print("====================================\n")
    
    return clean_text(total_text)


def save_text_to_file(text, output_file):
    """Save extracted text to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Text extraction completed. Output saved to:", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a PDF using OCR.")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the input PDF file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the extracted text.")
    args = parser.parse_args()
    
    extracted_text = extract_text_from_pdf(args.pdf_path)
    save_text_to_file(extracted_text, args.output_file)