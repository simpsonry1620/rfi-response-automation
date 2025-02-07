import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime

# Configuration
MAIN_URL = "https://developer.nvidia.com/developer-tools-catalog"
LINK_FILE = "../data/nvlinks.txt"
TXT_DIR = "../data/nvidia-text-archive"
LOG_FILE = "../data/script.log"

UNWANTED_SECTIONS = {
    "Quick Links", "Forums", "On-Demand Videos", "Resources", "Get Started", "Programs for you", "Ready to Get Started?",
    "Technical Training", "Section", "Sign Up", "Subscribe", "Download Now", "Get in Touch"
    "Privacy Policy", "Terms of Use", "Cookie Policy", "Copyright",
    "Feedback", "NGC Catalog Resources", "Latest Updates", "Announcements", 
    "Developer Blogs", "Developer News", "GTC Sessions", "Webinars", "Developer Resources"
}

UNWANTED_PHRASES = {
    "Read More", "Sign Up", "Subscribe", "Download Now",
    "View All", "Share", "Follow Us", "Click here", "Privacy Policy",
    "Terms of Use", "Cookie Policy", "Copyright", "All Rights Reserved", "NGC Catalog Resources"
}

def log(msg):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {msg}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

def sanitize_text(text):
    """Clean whitespace and special characters"""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'&nbsp;', ' ', text)
    return text

def is_filtered_content(text, filter_set):
    """Check if text contains any filtered phrases and log if filtered"""
    for phrase in filter_set:
        if phrase.lower() == text.lower():
            log(f"Filtered content due to '{phrase}': {text[:100]}...")
            return True
    return False

def process_html_content(html_content):
    """Process HTML content into structured text with bullet-point formatting"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove non-content elements
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'form']):
        tag.decompose()
        
    # Remove button-like elements and carat spans first
    for element in soup.find_all(class_=["cta--tert", "has-cta-icon"]):
        element.decompose()

    # Collect all header texts first to prevent duplicates
    header_texts = set()
    for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        header_text = sanitize_text(header.get_text())
        if header_text and not is_filtered_content(header_text, UNWANTED_SECTIONS):
            header_texts.add(header_text)

    output = []
    current_header = None
    seen_sections = set()
    seen_bullets = set()
    
    processed_elements = set()  # Track processed elements
    
    def process_text_element(element):
        """Process individual text elements and return formatted content"""
        # Skip already processed elements
        if element in processed_elements:
            return None
        
        text = sanitize_text(element.get_text())
        if not text:
            return None
            
        # Skip header texts and duplicates
        if text in header_texts or text in seen_bullets:
            return None
            
        # Filter unwanted phrases
        if is_filtered_content(text, UNWANTED_PHRASES):
            return None
            
        seen_bullets.add(text)
        processed_elements.add(element)  # Mark as processed
        return f"• {text}"
    
    # Main loop: process a variety of elements
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'div', 'details']):
        # Skip if this element is already inside a list container to avoid duplicate processing of nested items
        if element.parent and element.parent.name in ['ul', 'ol'] and element.name != 'details':
            continue
        
        if element.name.startswith('h'):
            header_text = sanitize_text(element.get_text())
            if header_text in seen_sections or is_filtered_content(header_text, UNWANTED_SECTIONS):
                current_header = None
                continue
                
            header_level = int(element.name[1])
            current_header = f'{"#" * header_level} {header_text}'
            seen_sections.add(header_text)
            output.append(current_header)
            
        elif current_header:
            if element.name == 'details':
                # Process accordion content recursively
                accordion_content = []
                summary = element.find('summary')
                if summary:
                    summary_text = sanitize_text(summary.get_text())
                    if not is_filtered_content(summary_text, UNWANTED_SECTIONS):
                        accordion_content.append(f'### {summary_text}')
                
                # Process all children except summary
                for child in element.find_all(recursive=False):
                    if child.name == 'summary':
                        continue
                    if child.name in ['ul', 'ol']:
                        for li in child.find_all('li'):
                            content = process_text_element(li)
                            if content:
                                accordion_content.append(content)
                    else:
                        content = process_text_element(child)
                        if content:
                            accordion_content.append(content)
                
                output.extend(accordion_content)
                    
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li'):
                    content = process_text_element(li)
                    if content:
                        output.append(content)
                        
            elif element.name in ['p', 'div']:
                # Avoid nested processing of paragraphs/divs already handled by parents
                if element.find(['p', 'div']) is None:
                    content = process_text_element(element)
                    if content:
                        output.append(content)

    # Clean trailing headers
    while output and output[-1].startswith('#'):
        output.pop()

    return '\n'.join(output)

def main():
    os.makedirs(TXT_DIR, exist_ok=True)
    log("Starting scraping process...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'
    })

    try:
        # Discover content links
        response = session.get(MAIN_URL)
        soup = BeautifulSoup(response.text, 'html.parser')
        js_url = urljoin(MAIN_URL, soup.find('script', src=re.compile('nv-developer-menu'))['src'])
        
        js_content = session.get(js_url).text
        links = set(re.findall(r'href:\s*"(https://developer\.nvidia\.com[^"]*)"', js_content))
        
        # Filter unwanted links
        links = {link for link in links 
                 if not any(excl in link for excl in ['contact', 'download', 'legal', 'email-signup'])}
        
        # Process all links
        with open(LINK_FILE, 'w') as f:
            f.write('\n'.join(sorted(links)))
        
        for idx, url in enumerate(sorted(links), 1):
            try:
                response = session.get(url, timeout=30)
                content = process_html_content(response.text)
                
                filename = re.sub(r'[/?=&]', '_', url.replace('https://', '')) + '.txt'
                filepath = os.path.join(TXT_DIR, filename)

                if filepath.endswith('developer.nvidia.com_.txt'):
                    filepath = filepath.replace(
                        'developer.nvidia.com_.txt', 
                        'developer.nvidia.com_DeveloperHome.txt'
                    )
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Source: {url}\n\n{content}")
                
                log(f"Processed {idx}/{len(links)}: {url}")
            
            except Exception as e:
                log(f"Error processing {url}: {str(e)}")
        
        log("Processing completed successfully")
    
    except Exception as main_error:
        log(f"Critical failure: {str(main_error)}")
        raise

if __name__ == "__main__":
    main()
