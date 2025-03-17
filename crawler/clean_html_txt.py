import re
import os
import argparse

"""
Clean HTML text file by removing all HTML/XML tags, <link> and <url> tags and their content,
extra spaces and newlines, and converting the text to lowercase.

Parameters:
    input_file (str): Path to the input text file
    output_file (str): Path to the output text file
"""

def remove_tags_and_links(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # remove <link> and <url> tags and their content
    content = re.sub(r'<(link|url)>.*?</\1>', '', content, flags=re.DOTALL)
    
    # remove all HTML/XML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # remove extra spaces and newlines
    content = re.sub(r'\s+', ' ', content).strip()
    
    # convert to lowercase
    content = content.lower()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean HTML text file")
    parser.add_argument("input_txt", help="Input text file path")
    parser.add_argument("output_txt", help="Output text file path")
    args = parser.parse_args()

    remove_tags_and_links(args.input_txt, args.output_txt)
    print("Cleaning complete. Results saved to", args.output_txt)
