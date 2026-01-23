import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def extract_internal_links(url, base_url):
    """
    Extracts all internal links from a page.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        links = set()
        base_domain = urlparse(base_url).netloc

        for a_tag in soup.find_all("a", href=True):
            link = a_tag["href"]
            absolute_link = urljoin(url, link)
            parsed_link = urlparse(absolute_link)

            # Only include internal links (same domain) and exclude fragments/queries
            if parsed_link.netloc == base_domain:
                clean_link = absolute_link.split("#")[0].split("?")[0].rstrip("/")
                if clean_link:
                    links.add(clean_link)
        return links
    except Exception as e:
        print(f"Error extracting links from {url}: {e}")
        return set()

def load_website_text(url, max_depth=1):
    """
    Scrapes text from a website, optionally crawling internal links.
    """
    visited = set()
    to_visit = [(url, 1)] # (url, current_depth)
    all_text = []
    
    headers = {"User-Agent": "Mozilla/5.0"}
    
    while to_visit:
        current_url, depth = to_visit.pop(0)
        if current_url in visited or depth > max_depth:
            continue
            
        visited.add(current_url)
        
        try:
            response = requests.get(current_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove navigation and other junk
            for element in soup(["nav", "footer", "script", "style", "header"]):
                element.decompose()

            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_page_text = "\n".join(lines)
            
            all_text.append(f"--- Source: {current_url} ---\n{clean_page_text}")

            # If we haven't reached max depth, find more links
            if depth < max_depth:
                new_links = extract_internal_links(current_url, url)
                for link in new_links:
                    if link not in visited:
                        to_visit.append((link, depth + 1))
                        
        except Exception as e:
            print(f"Error scraping {current_url}: {e}")

    return "\n\n".join(all_text)
