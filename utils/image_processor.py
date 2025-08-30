import numpy as np
from PIL import Image
import io
import re
from typing import Optional, Dict, Any
import requests
from urllib.parse import urlparse, parse_qs, unquote
import os
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from bs4 import BeautifulSoup  # optional, used for og:image fallback
except Exception:
    BeautifulSoup = None

class ImageProcessor:
    """Handles image preprocessing for embedding generation with enhanced URL support"""
    
    def __init__(self, target_size=(224, 224), cache_dir='./cache'):
        self.target_size = target_size
        self.cache_dir = cache_dir
        self.image_cache_dir = os.path.join(cache_dir, 'images')
        self.session = requests.Session()
        
        # Create cache directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.image_cache_dir, exist_ok=True)
        
        # Configure session for better compatibility
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cached_image_path(self, url: str) -> str:
        """Get cached image path for URL"""
        cache_key = self._get_cache_key(url)
        return os.path.join(self.image_cache_dir, f"{cache_key}.jpg")
    
    def _is_image_url_cached(self, url: str) -> bool:
        """Check if image URL is cached"""
        cache_path = self._get_cached_image_path(url)
        return os.path.exists(cache_path)
    
    def _cache_image(self, url: str, image: Image.Image) -> None:
        """Cache image to disk"""
        try:
            cache_path = self._get_cached_image_path(url)
            image.save(cache_path, 'JPEG', quality=85)
        except Exception as e:
            print(f"Warning: Could not cache image: {e}")
    
    def _load_cached_image(self, url: str) -> Optional[Image.Image]:
        """Load cached image from disk"""
        try:
            cache_path = self._get_cached_image_path(url)
            if os.path.exists(cache_path):
                return Image.open(cache_path)
        except Exception:
            pass
        return None
    
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
        Enhanced to handle more platforms and services.
        """
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            
            # Extended list of common image URL parameters
            preferred_keys = [
                'u', 'imgurl', 'url', 'mediaurl', 'media', 'image',
                'src', 'source', 'img', 'photo', 'picture', 'file',
                'attachment', 'download', 'link', 'href'
            ]
            
            for key in preferred_keys:
                if key in qs and qs[key]:
                    # Handle multiple levels of URL encoding
                    value = qs[key][0]
                    for _ in range(3):  # Try up to 3 levels of decoding
                        try:
                            decoded = unquote(value)
                            if decoded == value:
                                break
                            value = decoded
                        except Exception:
                            break
                    return value
            
            # Handle special cases for common platforms
            if 'pinterest.com' in parsed.netloc:
                # Pinterest URLs often have image IDs
                if '/pin/' in parsed.path:
                    pin_id = parsed.path.split('/pin/')[-1].split('/')[0]
                    return f"https://i.pinimg.com/originals/{pin_id}.jpg"
            
            elif 'instagram.com' in parsed.netloc:
                # Instagram URLs
                if '/p/' in parsed.path:
                    post_id = parsed.path.split('/p/')[-1].split('/')[0]
                    return f"https://www.instagram.com/p/{post_id}/media/?size=l"
            
            elif 'facebook.com' in parsed.netloc:
                # Facebook URLs might need special handling
                if 'photo' in parsed.path:
                    return url
            
        except Exception:
            pass
        return url

    def _fetch_image_bytes(self, url: str, timeout: int = 30) -> Optional[bytes]:
        """Enhanced image fetching with better error handling and retries"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Add referer for better compatibility
        headers['Referer'] = url
        
        # Retry logic
        for attempt in range(3):
            try:
                resp = self.session.get(
                    url, 
                    timeout=timeout, 
                    headers=headers, 
                    allow_redirects=True,
                    stream=True
                )
                resp.raise_for_status()
                
                # Check content type
                content_type = resp.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'webp', 'gif']):
                    raise Exception(f"URL does not point to an image (content-type: {content_type})")
                
                return resp.content
                
            except requests.exceptions.Timeout:
                if attempt == 2:
                    raise Exception("Request timed out after multiple attempts")
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise Exception(f"Failed to fetch image: {str(e)}")
                time.sleep(1)
        
        return None

    def _scrape_og_image(self, page_url: str) -> Optional[str]:
        """Enhanced og:image scraping with better HTML parsing"""
        if BeautifulSoup is None:
            return None
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36'
            }
            html = self.session.get(page_url, timeout=15, headers=headers).text
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try multiple meta tag patterns
            og_image = None
            
            # Check og:image
            og_tag = soup.find('meta', property='og:image') or soup.find('meta', attrs={'name': 'og:image'})
            if og_tag and og_tag.get('content'):
                og_image = og_tag['content']
            
            # Check twitter:image as fallback
            if not og_image:
                twitter_tag = soup.find('meta', attrs={'name': 'twitter:image'})
                if twitter_tag and twitter_tag.get('content'):
                    og_image = twitter_tag['content']
            
            # Check for any image in meta tags
            if not og_image:
                for meta in soup.find_all('meta'):
                    if 'image' in meta.get('content', '').lower():
                        og_image = meta['content']
                        break
            
            return og_image
            
        except Exception:
            return None

    def load_image_from_url(self, url: str, use_cache: bool = True) -> Image.Image:
        """Enhanced image loading from URL with caching and better error handling"""
        # Check cache first
        if use_cache and self._is_image_url_cached(url):
            cached_image = self._load_cached_image(url)
            if cached_image:
                return cached_image
        
        # Normalize and resolve redirect params
        direct = self._extract_direct_image_url(url.strip())
        
        # Try multiple approaches
        approaches = [
            lambda: self._try_direct_fetch(direct),
            lambda: self._try_og_image_fallback(direct),
            lambda: self._try_manual_redirect(direct)
        ]
        
        for approach in approaches:
            try:
                image = approach()
                if image:
                    # Cache the successful result
                    if use_cache:
                        self._cache_image(url, image)
                    return image
            except Exception:
                continue
        
        raise Exception("Unable to load a valid image from the provided URL")
    
    def _try_direct_fetch(self, url: str) -> Optional[Image.Image]:
        """Try direct image fetch"""
        content = self._fetch_image_bytes(url)
        if content:
            img = Image.open(io.BytesIO(content))
            img.verify()
            return Image.open(io.BytesIO(content))
        return None
    
    def _try_og_image_fallback(self, url: str) -> Optional[Image.Image]:
        """Try og:image fallback"""
        og_url = self._scrape_og_image(url)
        if og_url:
            return self._try_direct_fetch(og_url)
        return None
    
    def _try_manual_redirect(self, url: str) -> Optional[Image.Image]:
        """Try manual redirect following"""
        try:
            resp = self.session.head(url, timeout=15, allow_redirects=True)
            final_url = resp.url
            if final_url != url:
                return self._try_direct_fetch(final_url)
        except Exception:
            pass
        return None
    
    def batch_load_images_from_urls(self, urls: list, max_workers: int = 4) -> Dict[str, Image.Image]:
        """Load multiple images from URLs concurrently"""
        results = {}
        
        def load_single_image(url):
            try:
                image = self.load_image_from_url(url)
                return url, image
            except Exception as e:
                return url, None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(load_single_image, url): url for url in urls}
            
            for future in as_completed(future_to_url):
                url, image = future.result()
                if image:
                    results[url] = image
        
        return results
    
    def clear_cache(self) -> None:
        """Clear all cached images"""
        try:
            import shutil
            if os.path.exists(self.image_cache_dir):
                shutil.rmtree(self.image_cache_dir)
            os.makedirs(self.image_cache_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache"""
        try:
            if os.path.exists(self.image_cache_dir):
                files = os.listdir(self.image_cache_dir)
                total_size = sum(
                    os.path.getsize(os.path.join(self.image_cache_dir, f)) 
                    for f in files
                )
                return {
                    "cached_images": len(files),
                    "total_size_bytes": total_size,
                    "total_size_mb": total_size / (1024 * 1024)
                }
        except Exception:
            pass
        return {"cached_images": 0, "total_size_bytes": 0, "total_size_mb": 0}
