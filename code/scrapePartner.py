import os
import re
import requests
import hashlib
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from collections import deque
from typing import Set, Optional, List
from datetime import datetime
import logging

class WebScraper:
    def __init__(self, base_url: str, output_dir: str, max_depth: int = 3, 
                 unwanted_phrases: Optional[List[str]] = None):
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.unwanted_phrases = unwanted_phrases or [
            'privacy', 'terms', 'login', 'signup', 'contact', 'legal', 
            'profile', 'register', 'community', 'event', 'help', 'video'
        ]
        self.visited: Set[str] = set()
        self.discovered_links: Set[str] = {base_url}
        self.downloaded_pdfs: Set[str] = set()
        self.to_visit = deque([(base_url, 0)])
        self.session = requests.Session()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize counters
        self.processed_links = 0
        self.pdf_counter = 0
        
    def setup_logging(self):
        import sys
        log_file = os.path.join(self.output_dir, 'scraper.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)  # Use stdout instead of stderr
            ]
        )
        
    def download_pdf(self, url: str) -> bool:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/pdf,*/*'
            }
            response = self.session.get(url, headers=headers, timeout=10, stream=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                    filename = f"os.path.basename(urlparse(url).path)}"
                    if not filename.lower().endswith('.pdf'):
                        filename += '.pdf'
                    filepath = os.path.join(self.output_dir, 'pdfs', filename)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    logging.info(f'Downloaded PDF: {filename}')
                    return True
            return False
        except Exception as e:
            logging.error(f"Error downloading PDF {url}: {str(e)}")
            return False

    def extract_relevant_content(self, soup: BeautifulSoup, url: str) -> str:
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'form', 
                        'button', 'svg', 'img', 'iframe', 'noscript', 'meta']):
            tag.decompose()
            
        # Find main content area
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', {'role': 'main'}) or
            soup.find('section', class_='content') or 
            soup.find('div', id='content') or 
            soup.body or 
            soup
        )

        seen_text = set()
        output_lines = []
        current_bullets = []
        last_heading = None

        def clean_text(text: str) -> str:
            return ' '.join(text.strip().split())

        # Process content
        for tag in main_content.find_all(True):
            if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Only add previous heading and bullets if there were bullets
                if current_bullets:
                    if last_heading:
                        output_lines.append(last_heading)
                    output_lines.extend(current_bullets)
                    current_bullets = []
                
                heading_text = clean_text(tag.get_text())
                if heading_text and heading_text not in seen_text:
                    seen_text.add(heading_text)
                    level = int(tag.name[1])
                    last_heading = f"{'#' * level} {heading_text}"
            
            elif tag.name in ['p', 'li']:
                text = clean_text(tag.get_text())
                if text and text not in seen_text and len(text) > 3:  # Minimum length filter
                    if text != last_heading:  # Avoid redundancy with heading
                        seen_text.add(text)
                        current_bullets.append(f"- {text}")

        # Add final section if it has content
        if current_bullets:
            if last_heading:
                output_lines.append(last_heading)
            output_lines.extend(current_bullets)

        return '\n'.join(output_lines)

    def scrape(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'pdfs'), exist_ok=True)
        
        while self.to_visit:
            current_url, current_depth = self.to_visit.popleft()
            
            if (current_depth > self.max_depth or 
                current_url in self.visited or 
                any(phrase in current_url.lower() for phrase in self.unwanted_phrases)):
                continue
                
            self.visited.add(current_url)
            self.processed_links += 1
            
            try:
                if current_url.lower().endswith('.pdf'):
                    self.download_pdf(current_url)
                    continue

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept-Language': 'en-US,en;q=0.5'
                }
                
                response = self.session.get(current_url, headers=headers, timeout=10)
                logging.info(f'Fetched: {current_url} ({self.processed_links}/{len(self.discovered_links)})')
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                main_content = self.extract_relevant_content(soup, current_url)
                
                if main_content:
                    # # Get just the last part of the URL path
                    # filename = os.path.basename(urlparse(current_url).path)
                    # if not filename:
                    #     filename = 'index'
                    
                    # # Clean the filename and add unique identifier
                    # filename = re.sub(r'\W+', '_', filename)
                    filepath = os.path.join(self.output_dir, f"{current_url.strip('https://')}.txt")
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"URL: {current_url}\n")
                        f.write(f"Scraped: {datetime.now().isoformat()}\n")
                        f.write(f"Title: {soup.title.string if soup.title else 'No Title'}\n\n")
                        f.write(main_content)

                if current_depth < self.max_depth:
                    self.process_links(soup, current_url, current_depth)
                    
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching {current_url}: {str(e)}")  # Changed from error to info
            except Exception as e:
                logging.error(f"Unexpected error processing {current_url}: {str(e)}")  # Changed from error to info

    def process_links(self, soup: BeautifulSoup, current_url: str, current_depth: int):
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(current_url, link['href'])
            parsed_url = urlparse(absolute_url)
            
            if absolute_url.lower().endswith('.pdf'):
                self.pdf_counter += 1
                self.download_pdf(absolute_url)
            elif (parsed_url.netloc == urlparse(self.base_url).netloc and
                  not any(phrase in absolute_url.lower() for phrase in self.unwanted_phrases)):
                clean_url = parsed_url.geturl().split('#')[0]
                if clean_url not in self.discovered_links:
                    self.discovered_links.add(clean_url)
                    self.to_visit.append((clean_url, current_depth + 1))

    def print_statistics(self):
        logging.info("\nFinal Statistics:")
        logging.info(f"Total unique links discovered: {len(self.discovered_links)}")
        logging.info(f"Total links processed: {self.processed_links}")
        logging.info(f"Total PDFs downloaded: {self.pdf_counter}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Company Information Scraper')
    parser.add_argument('url', help='Base URL to scrape')
    parser.add_argument('output_dir', help='Output directory for scraped content')
    parser.add_argument('-d', '--depth', type=int, default=2,
                        help='Maximum recursion depth (default: 2)')
    parser.add_argument('-u', '--unwanted', nargs='+', default=None,
                        help='List of unwanted phrases in URLs')
    
    args = parser.parse_args()
    
    scraper = WebScraper(
        base_url=args.url,
        output_dir=args.output_dir,
        max_depth=args.depth,
        unwanted_phrases=args.unwanted
    )
    scraper.scrape()

if __name__ == "__main__":
    main()
