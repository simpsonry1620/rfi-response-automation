import os
import re
import subprocess
import hashlib
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Global variables
MAIN_URL = "https://developer.nvidia.com/developer-tools-catalog"
MAINFILE = "../data/nvlinks.txt"
TXT_DIR = "../data/nvidia-text-archive"
LOGFILE = "../data/script.log"

def log(msg):
    """Logging function for timestamped messages."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {msg}"
    print(log_message)
    with open(LOGFILE, 'a') as f:
        f.write(log_message + '\n')

def clean_text(input_text):
    """Function to clean text using regex."""
    # Remove numeric citations like [1]
    input_text = re.sub(r'\[[0-9]+\]', '', input_text)
    # Remove image references like [image.jpg]
    input_text = re.sub(r'\[.*?\.(jpg|png|gif|bmp)\]', '', input_text)
    # Remove HTML tags
    input_text = re.sub(r'<[^>]*>', '', input_text)
    # Remove session IDs
    input_text = re.sub(r';?jsessionid=[A-Za-z0-9]+', '', input_text)
    # Remove dates (YYYY-MM-DD)
    input_text = re.sub(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', '', input_text)
    # Remove times
    input_text = re.sub(r'[0-9]{1,2}:[0-9]{2}(:[0-9]{2})? ?(AM|PM|am|pm)?', '', input_text)
    # Remove lines starting with #
    input_text = re.sub(r'^\s*#.*$', '', input_text, flags=re.MULTILINE)
    # Normalize whitespace
    input_text = re.sub(r'\s+', ' ', input_text)
    # Remove specific phrases
    phrases_to_remove = ["Contact", "Join", "Learn More", "Quick Links", "Forums", "On-Demand Videos"]
    for phrase in phrases_to_remove:
        input_text = re.sub(rf'\b{re.escape(phrase)}\b', '', input_text, flags=re.IGNORECASE)
    return input_text.strip()

def main():
    os.makedirs(TXT_DIR, exist_ok=True)
    log("Starting script...")

    # Fetch the main page to find the JavaScript URL
    log("Fetching JavaScript URL...")
    response = requests.get(MAIN_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    js_url = soup.find('script', src=re.compile('nv-developer-menu'))['src']

    if not js_url:
        log("Error: Could not find the nv-developer-menu script URL.")
        return

    if not js_url.startswith('http'):
        js_url = f"https://developer.nvidia.com{js_url}"
    log(f"Fetched JS URL: {js_url}")

    # Extract links from the JavaScript file
    log("Extracting links from JavaScript file...")
    js_content = requests.get(js_url).text
    links = set(re.findall(r'href:\s*"(https://developer\.nvidia\.com[^"]*)"', js_content))

    with open(MAINFILE, 'w') as f:
        for link in sorted(links):
            f.write(f"{link}\n")

    # Process each link
    total_links = len(links)
    for current_link, link in enumerate(sorted(links), 1):
        log(f"Processing link {current_link} of {total_links}: {link}")

        # Generate filenames for raw and cleaned content
        raw_file = os.path.join(TXT_DIR, "raw.html")
        cleaned_file = os.path.join(TXT_DIR, re.sub(r'[/?=&]', '_', link.replace('https://', ''))) + '.txt'
        if cleaned_file = 'data/nvidia-text-archive/developer.nvidia.com_.txt':
            cleaned_file = 'data/nvidia-text-archive/developer.nvidia.com_DeveloperHome.txt'
        # Fetch raw HTML using lynx and save it to a temporary file
        try:
            subprocess.run(['lynx', '--dump', '--display_charset=utf-8', link], stdout=open(raw_file, 'w'), check=True)
        except subprocess.CalledProcessError:
            log(f"Error fetching {link}. Skipping...")
            continue

        # Clean the raw HTML
        with open(raw_file, 'r') as f:
            raw_content = f.read()
        cleaned_content = clean_text(raw_content)

        with open(cleaned_file, 'w') as f:
            f.write(cleaned_content)

        os.remove(raw_file)

    log("All links processed.")

if __name__ == "__main__":
    main()
