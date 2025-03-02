import os
import pdfplumber
import re

"""
PDF Parsing Script

Scans a directory for PDF files, extracts text using pdfplumber, processes the text (convert to lowercase, remove newlines, and strip URLs), and saves each output as a .txt file.

Usage:
    python pdf_parser.py <pdf_directory> <output_directory>

Arguments:
    pdf_directory      - Path to the directory containing PDF files.
    output_directory   - Path to the directory where extracted text files will be saved.

Dependencies:
    - pdfplumber
"""

def list_pdfs(directory):
    """List all PDF files in a given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".pdf")]

def clean_text(text):
    """Process extracted text: convert to lowercase, remove newlines, and strip URLs."""
    text = text.lower()  # Convert to lowercase
    text = text.replace("\n", " ")  # Replace newlines with space
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    return text

def extract_text_pdfplumber(pdf_path):
    """Extract text from a PDF using pdfplumber and clean it."""
    text_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_data.append(page_text)
    raw_text = " ".join(text_data)
    return clean_text(raw_text)  # Process text before returning

def crawl_and_save_text(directory, output_directory):
    """Crawl a directory for PDFs, extract and clean text, then save results to .txt files."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    pdf_files = list_pdfs(directory)
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        extracted_text = extract_text_pdfplumber(pdf_file)
        
        txt_file_name = os.path.splitext(os.path.basename(pdf_file))[0] + ".txt"
        txt_file_path = os.path.join(output_directory, txt_file_name)
        
        with open(txt_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)
        
        print(f"Saved extracted text to {txt_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crawl directory for PDFs, extract and process text.")
    parser.add_argument("directory", type=str, help="Path to the directory containing PDFs.")
    parser.add_argument("output_directory", type=str, help="Directory to save extracted text files.")
    
    args = parser.parse_args()
    crawl_and_save_text(args.directory, args.output_directory)
