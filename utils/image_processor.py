import numpy as np
from PIL import Image
import io
import re

from typing import Optional, List, Dict
import requests
from urllib.parse import urlparse, parse_qs, unquote
import platform
import ssl
import certifi
try:
    from bs4 import BeautifulSoup  # optional, used for og:image fallback
except Exception:
    BeautifulSoup = None

# Import configuration
try:
    from config.performance import IMAGE_CONFIG, PLATFORM_CONFIG, URL_PATTERNS
except ImportError:
    # Fallback configuration if config file is not available
    IMAGE_CONFIG = {
        'target_size': (224, 224),
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'supported_formats': ['png', 'jpg', 'jpeg', 'webp'],
        'quality_threshold': 32,  # Minimum image dimensions
        'timeout': 30,  # URL fetch timeout
        'max_retries': 3
    }
    PLATFORM_CONFIG = {
        'ssl_verify': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36',
        'headers': {
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8,application/octet-stream',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    }
    URL_PATTERNS = {
        'bing': [
            r'bing\.com.*[?&]u=([^&]+)',
            r'bing\.com.*[?&]url=([^&]+)',
            r'bing\.com.*[?&]mediaurl=([^&]+)'
        ],
        'google': [
            r'google\.com.*[?&]imgurl=([^&]+)',
            r'google\.com.*[?&]url=([^&]+)',
            r'google\.com.*[?&]mediaurl=([^&]+)'
        ],
        'pinterest': [
            r'pinterest\.com.*[?&]url=([^&]+)',
            r'pinterest\.com.*[?&]media=([^&]+)'
        ],
        'instagram': [
            r'instagram\.com.*[?&]url=([^&]+)',
            r'instagram\.com.*[?&]media=([^&]+)'
        ],
        'facebook': [
            r'facebook\.com.*[?&]url=([^&]+)',
            r'facebook\.com.*[?&]media=([^&]+)'
        ],
        'twitter': [
            r'twitter\.com.*[?&]url=([^&]+)',
            r'twitter\.com.*[?&]media=([^&]+)'
        ],
        'amazon': [
            r'amazon\.com.*[?&]url=([^&]+)',
            r'amazon\.com.*[?&]media=([^&]+)'
        ],
        'ebay': [
            r'ebay\.com.*[?&]url=([^&]+)',
            r'ebay\.com.*[?&]media=([^&]+)'
        ]
    }

class ImageProcessor:
    """Handles image preprocessing for embedding generation with enhanced URL support"""
    
    def __init__(self, target_size=None):
        self.target_size = target_size or IMAGE_CONFIG['target_size']
        self.session = self._create_session()
        self.url_patterns = URL_PATTERNS
    
    def _create_session(self):
        """Create a requests session with cross-platform SSL configuration"""
        session = requests.Session()
        
        # Cross-platform SSL configuration
        if platform.system() == 'Windows':
            # Windows-specific SSL configuration
            session.verify = certifi.where()
        elif platform.system() == 'Darwin':  # macOS
            # macOS-specific SSL configuration
            session.verify = certifi.where()
        else:  # Linux and others
            # Linux-specific SSL configuration
            session.verify = certifi.where()
        
        # Enhanced headers for better compatibility
        session.headers.update(PLATFORM_CONFIG['headers'])
        
        return session
    
    def preprocess_image(self, image):
        """
        Preprocess image for embedding generation
        
        Args:
            image: PIL Image object
            
        Returns:
            numpy array: Preprocessed image ready for model input
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image with high-quality resampling
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Normalize pixel values to [0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def load_image_from_path(self, image_path):
        """
        Load and preprocess image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array: Preprocessed image
        """
        try:
            image = Image.open(image_path)
            return self.preprocess_image(image)
        except Exception as e:
            raise Exception(f"Error loading image from {image_path}: {str(e)}")
    
    def validate_image(self, image):
        """
        Validate image format and size
        
        Args:
            image: PIL Image object
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if image is valid
            if not isinstance(image, Image.Image):
                return False
            
            # Check image size (not too small)
            width, height = image.size
            if width < IMAGE_CONFIG['quality_threshold'] or height < IMAGE_CONFIG['quality_threshold']:
                return False
            
            # Check file size (assume reasonable limits)
            if width * height > IMAGE_CONFIG['max_file_size']:
                return False
            
            return True
            
        except Exception:
            return False

    def _extract_direct_image_url(self, url: str) -> str:
        """Enhanced URL resolution for multiple sources"""
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            
            # Try to extract from query parameters
            preferred_keys = ['u', 'imgurl', 'url', 'mediaurl', 'media', 'image', 'src', 'link']
            for key in preferred_keys:
                if key in qs and qs[key]:
                    value = qs[key][0]
                    try:
                        # Handle double-encoded URLs
                        value = unquote(unquote(value))
                    except Exception:
                        value = unquote(value)
                    
                    # Validate the extracted URL
                    if value.startswith(('http://', 'https://')):
                        return value
            
            # Try pattern matching for specific platforms
            for platform_name, patterns in self.url_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, url, re.IGNORECASE)
                    if match:
                        extracted_url = match.group(1)
                        try:
                            extracted_url = unquote(extracted_url)
                            if extracted_url.startswith(('http://', 'https://')):
                                return extracted_url
                        except Exception:
                            continue
            
            return url
            
        except Exception:
            return url

    def _fetch_image_bytes(self, url: str) -> Optional[bytes]:
        """Fetch image bytes with enhanced error handling"""
        try:
            # Add referer for better compatibility
            headers = self.session.headers.copy()
            headers['Referer'] = url
            
            # Use session for better connection handling
            resp = self.session.get(url, timeout=IMAGE_CONFIG['timeout'], headers=headers, allow_redirects=True)
            resp.raise_for_status()
            
            # Check if response is actually an image
            content_type = resp.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                # Try to extract image from HTML if it's a webpage
                return self._extract_image_from_html(resp.text, url)
            
            return resp.content
            
        except requests.exceptions.Timeout:
            raise Exception("Request timeout - the server took too long to respond")
        except requests.exceptions.ConnectionError:
            raise Exception("Connection error - unable to reach the server")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise Exception("Image not found (404 error)")
            elif e.response.status_code == 403:
                raise Exception("Access denied (403 error) - the image may be protected")
            else:
                raise Exception(f"HTTP error {e.response.status_code}: {e.response.reason}")
        except Exception as e:
            raise Exception(f"Failed to fetch image: {str(e)}")

    def _extract_image_from_html(self, html_content: str, page_url: str) -> Optional[bytes]:
        """Extract image from HTML content using multiple methods"""
        if BeautifulSoup is None:
            return None
            
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try multiple image extraction methods
            image_selectors = [
                'meta[property="og:image"]',
                'meta[name="og:image"]',
                'meta[property="twitter:image"]',
                'meta[name="twitter:image"]',
                'link[rel="image_src"]',
                'link[rel="preload"][as="image"]',
                'img[src*="http"]'
            ]
            
            for selector in image_selectors:
                elements = soup.select(selector)
                for element in elements:
                    if element.name == 'meta':
                        image_url = element.get('content')
                    elif element.name == 'link':
                        image_url = element.get('href')
                    elif element.name == 'img':
                        image_url = element.get('src')
                    else:
                        continue
                    
                    if image_url and image_url.startswith(('http://', 'https://')):
                        try:
                            # Try to fetch the extracted image
                            return self._fetch_image_bytes(image_url)
                        except Exception:
                            continue
            
            return None
            
        except Exception:
            return None

    def load_image_from_url(self, url: str) -> Image.Image:
        """Load a PIL Image from various URL sources with enhanced error handling"""
        # Normalize and resolve redirect params
        direct = self._extract_direct_image_url(url.strip())
        
        # Try multiple fallback methods
        fallback_methods = [
            lambda: self._try_direct_fetch(direct),
            lambda: self._try_og_image_fallback(direct),
            lambda: self._try_redirect_follow(direct),
            lambda: self._try_platform_specific(direct)
        ]
        
        last_error = None
        for method in fallback_methods:
            try:
                return method()
            except Exception as e:
                last_error = e
                continue
        
        # If all methods fail, provide detailed error information
        raise Exception(f"Unable to load image from URL after trying multiple methods. Last error: {str(last_error)}")
    
    def _try_direct_fetch(self, url: str) -> Image.Image:
        """Try to fetch image directly from URL"""
        content = self._fetch_image_bytes(url)
        if content:
            img = Image.open(io.BytesIO(content))
            img.verify()
            return Image.open(io.BytesIO(content))
        raise Exception("Direct fetch failed")
    
    def _try_og_image_fallback(self, url: str) -> Image.Image:
        """Try to extract og:image from webpage"""
        og_image = self._scrape_og_image(url)
        if og_image:
            content = self._fetch_image_bytes(og_image)
            img = Image.open(io.BytesIO(content))
            img.verify()
            return Image.open(io.BytesIO(content))
        raise Exception("OG image fallback failed")
    
    def _try_redirect_follow(self, url: str) -> Image.Image:
        """Follow redirects manually"""
        try:
            with self.session as s:
                r = s.get(url, timeout=IMAGE_CONFIG['timeout'], allow_redirects=True)
                final = r.url
                if final != url:
                    content = self._fetch_image_bytes(final)
                    img = Image.open(io.BytesIO(content))
                    img.verify()
                    return Image.open(io.BytesIO(content))
        except Exception:
            pass
        raise Exception("Redirect following failed")
    
    def _try_platform_specific(self, url: str) -> Image.Image:
        """Try platform-specific image extraction"""
        # This could be extended with platform-specific logic
        raise Exception("Platform-specific extraction not implemented")

    def _scrape_og_image(self, page_url: str) -> Optional[str]:
        """Enhanced og:image scraping"""
        if BeautifulSoup is None:
            return None
        try:
            html = self.session.get(page_url, timeout=IMAGE_CONFIG['timeout']).text
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try multiple meta tag patterns
            meta_patterns = [
                'meta[property="og:image"]',
                'meta[name="og:image"]',
                'meta[property="twitter:image"]',
                'meta[name="twitter:image"]',
                'link[rel="image_src"]'
            ]
            
            for pattern in meta_patterns:
                tag = soup.select_one(pattern)
                if tag and tag.get('content'):
                    return tag['content']
                elif tag and tag.get('href'):
                    return tag['href']
            
            return None
            
        except Exception:
            return None
    
    def get_supported_platforms(self) -> List[str]:
        """Get list of supported platforms for URL extraction"""
        return list(self.url_patterns.keys())
    
    def add_custom_pattern(self, platform_name: str, pattern: str):
        """Add custom URL pattern for a platform"""
        if platform_name not in self.url_patterns:
            self.url_patterns[platform_name] = []
        self.url_patterns[platform_name].append(pattern)
