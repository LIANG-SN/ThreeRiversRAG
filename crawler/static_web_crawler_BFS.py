import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import argparse
import time
import random
import re
import csv
from urllib.parse import urljoin, urlparse
import hashlib
from tqdm import tqdm
import ast

"""
Static Webpage Crawler with BFS

This script reads a CSV file containing a list of URLs and extracts all text content from each webpage
up to 2 levels deep (BFS). The extracted text is saved as individual .txt files in the specified
output directory. If duplicate filenames occur, a numeric suffix is added to avoid overwriting files.

All sublinks found during the BFS are extracted and saved to a single CSV file named source_data_sublinks.csv.
Failed URLs are logged into failed_to_crawl.csv.

Usage:
    python static_webpage_crawler.py <csv_file> <output_directory>

Arguments:
    csv_file        - Path to the CSV file containing URLs.
    output_directory - Directory to save extracted text files and sublinks.

"""

# User-agent list to avoid blocking
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
]

# Keywords to match (case insensitive)
KEYWORDS = [
    "cmu",
    "carnegie",
    "mellon",
    "university",
    "tartans",
    "lti",
    "scotty",
    "pittsburgh",
    "pitts",
    "pit",
    "carnival",
    "trustarts",
    "event"
]

# Global variable that keeps track of all visited links
GLOBAL_VISITED = {} # key: url(str), value: len(GLOBAL_VISITED)
overall_sublinks = []

####### load GLOBAL_VISITED
txt_path = '../data/crawled_data/retrieve_source/total_web_txt/global_visited.txt'
# load GLOBAL_VISITED from txt file
with open(txt_path, 'r', encoding='utf-8') as file:
    file_content = file.read()
# use ast.literal_eval() to convert string to dictionary
try:
    GLOBAL_VISITED = ast.literal_eval(file_content)
    print("len(GLOBAL_VISITED):", len(GLOBAL_VISITED))
except ValueError as e:
    print(f"Error parsing the content: {e}")

def clean_text(text):
    """Convert extracted text to lowercase, remove newline characters and URLs"""
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r'http[s]?://\S+', '', text)
    return text

def sanitize_filename(url):
    """Generate a unique filename for the URL using MD5 hash"""
    return hashlib.md5(url.encode("utf-8")).hexdigest()

def fetch_page_text_and_sublinks(url, retries=3, timeout=5):
    """Fetch page text and all href links using requests, return text and absolute links list"""
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            if response.status_code == 404:
                return None, []
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            text = soup.get_text(separator='\n', strip=True)
            text = clean_text(text)
            raw_links = [link.get('href') for link in soup.find_all('a', href=True)]
            sublinks = [urljoin(url, href) for href in raw_links]
            return text, sublinks
        except requests.exceptions.RequestException:
            attempt += 1
            time.sleep(1)
    return None, []

def filter_sublinks(start_url, sublinks):
    """
    Filter logic:
      1) Only keep links starting with http
      2) If the link does not contain "wiki", keep it directly if the link is in the same domain as start_url
      3) If the link contains "wiki", keep it only if the link contains any keyword in KEYWORDS
    """
    # ###### only used for add on wiki pages ######
    # filtered = []
    # for link in sublinks:
    #     if 'Main_Page' in link or 'Current_events' in link:
    #         continue
    #     if 'About' in link:
    #         continue 
    #     filtered.append(link)
    # return filtered
    # #############################################

    filtered = []
    base_domain = urlparse(start_url).netloc

    for link in sublinks:
        if not link.startswith("http"):
            continue
        parsed = urlparse(link)
        link_domain = parsed.netloc
        link_lower = link.lower()

        if "wiki" in link_lower:
            continue

        if link_domain == base_domain:
            filtered.append(link)
        elif any(kw in link_lower for kw in KEYWORDS):
            filtered.append(link)
    
    return filtered

def save_text_to_file(text, filename, output_dir):
    """Save text to a txt file using the filename"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f"{filename}.txt")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def save_global_visited_to_file(output_dir):
    """Save GLOBAL_VISITED to a txt file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "global_visited.txt")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(str(GLOBAL_VISITED))

def save_all_sublinks_to_csv(sublinks_data, output_dir):
    """Append all sublink records to a CSV file"""
    file_path = os.path.join(output_dir, "source_data_sublinks.csv")
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Source URL", "Sublink"])
        writer.writerows(sublinks_data)
    print(f"Appended sublinks to {file_path}")

def log_failed_url(url, output_dir):
    """Log failed URL to failed_to_crawl.csv"""
    failed_log_path = os.path.join(output_dir, "failed_to_crawl.csv")
    log_exists = os.path.exists(failed_log_path)
    with open(failed_log_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not log_exists:
            writer.writerow(["Failed URL"])
        writer.writerow([url])

def bfs_crawl(start_url, output_dir, max_depth):
    """
    BFS crawl for a single starting URL.
    Save each page as a txt file, filename is the MD5 hash of the URL.
    Return all (source_url, sublink) records found in this BFS.
    """
    all_sublinks = []
    visited = set()
    queue = [(start_url, 0)]
    
    while queue:
        current_url, depth = queue.pop(0)
        # If meet visited link, skip 
        if current_url in visited or current_url in GLOBAL_VISITED:
            continue

        # ###### only used for add on wiki pages ######
        # if current_url in visited:
        #     continue
        # if current_url in GLOBAL_VISITED and depth != 0:
        #     continue
        # #############################################

        visited.add(current_url)
        GLOBAL_VISITED[current_url] = len(GLOBAL_VISITED) + 1
        
        text, sublinks = fetch_page_text_and_sublinks(current_url)
        if not text and not sublinks:
            print(f"Failed to crawl: {current_url}")
            log_failed_url(current_url, output_dir)
            continue

        # use value of GLOBAL_VISITED[current_url] as filename
        filename = str(GLOBAL_VISITED[current_url])
        save_text_to_file(text, filename, output_dir)
        
        filtered = filter_sublinks(current_url, sublinks)
        for sl in filtered:
            all_sublinks.append((current_url, sl))
        
        if depth < max_depth:
            for link in filtered:
                if link not in GLOBAL_VISITED:
                    # find a new link: do web crawl
                    queue.append((link, depth + 1))

    return all_sublinks

def crawl_webpages(csv_file, output_dir, max_depth):
    """Read the starting URL from CSV, crawl each URL using BFS, and save text and sublink records."""
    df = pd.read_csv(csv_file)
    overall_sublinks = []

    for _, row in df.iterrows():
        # source_name = str(row.iloc[0]).strip().replace(" ", "_").replace("/", "_")
        url = str(row.iloc[1]).strip()
        print(f"\n=== Start BFS for {url} ===")
        sublinks_found = bfs_crawl(url, output_dir, max_depth=max_depth)
        overall_sublinks.extend(sublinks_found)

    if overall_sublinks:
        save_all_sublinks_to_csv(overall_sublinks, output_dir)
    
    # save GLOBAL_VISITED to txt file
    save_global_visited_to_file(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl static webpages (BFS up to depth=2), extract text and sublinks.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing URLs.")
    parser.add_argument("output_directory", type=str, help="Directory to save extracted text files and sublinks.")
    args = parser.parse_args()

    # get sublinks up to 2 levels deep
    max_depth = 2
    crawl_webpages(args.csv_file, args.output_directory, max_depth)
    print("Crawling completed.")