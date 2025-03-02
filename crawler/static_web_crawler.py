import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import argparse
import time
import random
import re
import csv
from urllib.parse import urljoin

"""
Static Webpage Crawler

This script reads a CSV file containing a list of URLs and extracts all text content from each webpage.
The extracted text is saved as individual .txt files in the specified output directory.
If duplicate filenames occur, a numeric suffix is added to avoid overwriting files.
Additionally, all sublinks found on each webpage are extracted and saved to a single CSV file named source_data_sublinks.csv.
Failed URLs are logged into failed_to_crawl.csv.

Usage:
    python static_webpage_crawler.py <csv_file> <output_directory>

Arguments:
    csv_file        - Path to the CSV file containing URLs.
    output_directory - Directory to save extracted text files and sublinks.

Dependencies:
    - requests
    - BeautifulSoup
    - pandas
"""

# User-agent list to avoid blocking
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
]

def clean_text(text):
    """Process extracted text: convert to lowercase, remove newlines, and strip URLs."""
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r'http[s]?://\S+', '', text)
    return text

def fetch_page_text_and_sublinks(url, retries=3, timeout=5):
    """Fetch the main text content and sublinks from a webpage with retries and randomized user-agent."""
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            text = soup.get_text(separator='\n', strip=True)
            text = clean_text(text)
            sublinks = [urljoin(url, link['href']) for link in soup.find_all('a', href=True)]
            return text, sublinks
        except requests.exceptions.RequestException:
            attempt += 1
            time.sleep(1)
    return None, []

def save_text_to_file(text, filename, output_dir):
    """Save extracted text to a .txt file with a unique filename."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f"{filename}.txt")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Saved: {file_path}")

def save_all_sublinks_to_csv(sublinks_data, output_dir):
    """Append all extracted sublinks to an existing CSV file."""
    file_path = os.path.join(output_dir, "source_data_sublinks.csv")
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Source URL", "Sublink"])
        writer.writerows(sublinks_data)
    print(f"Appended sublinks to {file_path}")

def log_failed_url(url, output_dir):
    """Log failed URLs to failed_to_crawl.csv."""
    failed_log_path = os.path.join(output_dir, "failed_to_crawl.csv")
    log_exists = os.path.exists(failed_log_path)
    with open(failed_log_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not log_exists:
            writer.writerow(["Failed URL"])
        writer.writerow([url])

def crawl_webpages(csv_file, output_dir):
    """Crawl webpages, extract text and sublinks, and save results."""
    df = pd.read_csv(csv_file)
    filename_counter = {}
    all_sublinks = []
    
    for _, row in df.iterrows():
        source_name = str(row.iloc[0]).strip().replace(" ", "_").replace("/", "_")
        url = row.iloc[1]
        print(f"Crawling: {url}")
        text, sublinks = fetch_page_text_and_sublinks(url)
        
        if text:
            if source_name in filename_counter:
                filename_counter[source_name] += 1
                filename = f"{source_name}_{filename_counter[source_name]}"
            else:
                filename_counter[source_name] = 1
                filename = source_name
            save_text_to_file(text, filename, output_dir)
        
        if sublinks:
            for sublink in sublinks:
                all_sublinks.append((url, sublink))
        
        if not text and not sublinks:
            print(f"Failed to crawl: {url}")
            log_failed_url(url, output_dir)
    
    save_all_sublinks_to_csv(all_sublinks, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl static webpages, extract text and sublinks.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing URLs.")
    parser.add_argument("output_directory", type=str, help="Directory to save extracted text files and sublinks.")
    args = parser.parse_args()
    crawl_webpages(args.csv_file, args.output_directory)
