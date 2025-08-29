import numpy as np
from PIL import Image
import io
import re
from typing import Optional
import requests
from urllib.parse import urlparse, parse_qs, unquote
try:
    from bs4 import BeautifulSoup  # optional, used for og:image fallback
except Exception:
    BeautifulSoup = None

class ImageProcessor:
    """Handles image preprocessing for embedding generation"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
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
            
            # Resize image
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
            if width < 32 or height < 32:
                return False
            
            # Check file size (assume reasonable limits)
            if width * height > 10000000:  # 10MP limit
                return False
            
            return True
            
        except Exception:
            return False

    def _extract_direct_image_url(self, url: str) -> str:
        """Resolve common redirect/query patterns to a direct image URL.
        - Bing ads/click URLs: extract `u=` param
        - Google Images: extract `imgurl=` param
        - URL-encoded values are unquoted
        """
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if 'u' in qs and qs['u']:
                return unquote(qs['u'][0])
            if 'imgurl' in qs and qs['imgurl']:
                return unquote(qs['imgurl'][0])
        except Exception:
            pass
        return url

    def _fetch_image_bytes(self, url: str) -> Optional[bytes]:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        resp = requests.get(url, timeout=20, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        return resp.content

    def _scrape_og_image(self, page_url: str) -> Optional[str]:
        """If a page is not a direct image, try to find og:image via HTML."""
        if BeautifulSoup is None:
            return None
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36'
            }
            html = requests.get(page_url, timeout=15, headers=headers).text
            soup = BeautifulSoup(html, 'html.parser')
            tag = soup.find('meta', property='og:image') or soup.find('meta', attrs={'name': 'og:image'})
            if tag and tag.get('content'):
                return tag['content']
        except Exception:
            return None
        return None

    def load_image_from_url(self, url: str) -> Image.Image:
        """Load a PIL Image from potentially indirect URLs (Bing/Google redirects).
        Falls back to scraping og:image if needed.
        """
        # Normalize and resolve redirect params
        direct = self._extract_direct_image_url(url.strip())
        try:
            content = self._fetch_image_bytes(direct)
            img = Image.open(io.BytesIO(content))
            img.verify()
            return Image.open(io.BytesIO(content))
        except Exception:
            # Try og:image from the page
            og = self._scrape_og_image(direct)
            if og:
                content = self._fetch_image_bytes(og)
                img = Image.open(io.BytesIO(content))
                img.verify()
                return Image.open(io.BytesIO(content))
            raise Exception("Unable to load a valid image from the provided URL")
