from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import subprocess
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

def download_site_as_pdfs_with_js(base_url, output_dir, visited=None):
    if visited is None:
        visited = set()

    if base_url in visited:
        return

    visited.add(base_url)
    driver = None 
    
    print(f'Attempting to scrape: {base_url}')
    
    try:
        # Configure Chrome options
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        driver = webdriver.Chrome(options=options)

        # Load the page and wait for JavaScript content
        driver.get(base_url)
        wait = WebDriverWait(driver, 5)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Get the fully rendered page content
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')

        # Generate filename
        parsed_url = urlparse(base_url)
        filename = f"{output_dir}/{parsed_url.path.strip('/').replace('/', '_')}.pdf"
        if not filename.endswith('.pdf'):
            filename = f"{filename}.pdf"

        # Save rendered HTML and convert to PDF using lynx
        html_file = filename.replace('.pdf', '.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        subprocess.run(["lynx", "-dump", "-nolist", html_file, "|", "ps2pdf", "-", filename], check=True)

        # Process links after JavaScript rendering
        links = soup.find_all('a')
        for link in links:
            href = link.get('href')
            if href:
                full_url = urljoin(base_url, href)
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    download_site_as_pdfs_with_js(full_url, output_dir, visited)

    except Exception as e:
        print(f"Error processing {base_url}: {str(e)}")
    finally:
        if driver:  # Check if driver exists before quitting
            driver.quit()
