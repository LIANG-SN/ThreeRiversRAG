import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import argparse
import random
import re
import csv
import hashlib
from urllib.parse import urljoin, urlparse
import ast

"""
Dynamic Webpage Crawler with Selenium

This script crawls dynamically loaded webpages using Selenium, extracting text content from specified URLs. Extracted text is saved as individual .txt files in the specified output directory, with filenames generated from unique identifiers to prevent overwriting files. The script maintains a global record of visited URLs to avoid redundant crawling, storing this information persistently. Failed URLs during crawling are logged into failed_to_crawl.csv for later review.

Usage:
    python dynamic_webpage_crawler.py --url  --output <output_directory>

Arguments:
    --url              - URL of the webpage to crawl.
    --output           - Directory to save extracted text files and logs.

"""

# User-agent list to avoid blocking
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
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
    """Lower the text, remove newlines, and strip URLs."""
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r'http[s]?://\S+', '', text)
    return text

def save_text_to_file(text, filename, output_dir):
    """Save text to a txt file using the filename provided."""
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

def log_failed_url(url, output_dir):
    """Log failed URLs to failed_to_crawl.csv."""
    failed_log_path = os.path.join(output_dir, "failed_to_crawl.csv")
    log_exists = os.path.exists(failed_log_path)
    with open(failed_log_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not log_exists:
            writer.writerow(["Failed URL"])
        writer.writerow([url])

def dynamic_web_crawler(url):
    """Use selenium to crawl text from a url."""
    try:
        # Initialize Chrome WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless=new") 
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        # Initialize Chrome WebDriver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(url)

        # Ensure page is fully loaded
        time.sleep(5)

        # Get page source and close the browser
        page_source = driver.page_source
        driver.quit()

        # Parse the page content using BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract and clean the text from the page
        page_text = soup.get_text(separator='\n', strip=True)
        print(f"Extracted text for {url}:")

        # clean the text
        page_text = clean_text(page_text)
        return page_text

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def crawl_website(url, output_dir):
    """Crawl a website and save the text content to a file"""
    web_content = dynamic_web_crawler(url)
    if web_content:
        if url not in GLOBAL_VISITED:
            GLOBAL_VISITED[url] = len(GLOBAL_VISITED) + 1
        filename = GLOBAL_VISITED[url]
        save_text_to_file(web_content, filename, output_dir)

        # Save the updated GLOBAL_VISITED to a file
        save_global_visited_to_file(output_dir)
    else:
        log_failed_url(url, output_dir)

    return web_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl dynamic webpage using selenium")
    parser.add_argument("--url", type=str, help="URL to crawl")
    parser.add_argument("--output", type=str, help="Save the directory for extracting text file")
    
    args = parser.parse_args()
    
    url = args.url
    output_dir = args.output
    crawl_website(url, output_dir)
    print("Crawling completed.")

