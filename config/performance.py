"""
Performance configuration for the Visual Product Matcher application.
This file contains settings to optimize loading times and cross-platform compatibility.
"""

import os
from pathlib import Path

# Model Configuration
MODEL_CONFIG = {
    'default_model': 'mobilenet',  # Faster loading, good accuracy
    'alternative_model': 'efficientnet_small',  # Slower loading, better accuracy
    'use_cache': True,
    'cache_dir': 'cache',
    'model_weights_dir': 'models'
}

# Image Processing Configuration
IMAGE_CONFIG = {
    'target_size': (224, 224),
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'supported_formats': ['png', 'jpg', 'jpeg', 'webp'],
    'quality_threshold': 32,  # Minimum image dimensions
    'timeout': 30,  # URL fetch timeout
    'max_retries': 3
}

# Caching Configuration
CACHE_CONFIG = {
    'data_ttl': 3600,  # 1 hour for product data
    'embeddings_ttl': 3600,  # 1 hour for embeddings
    'model_cache_enabled': True,
    'image_cache_enabled': True,
    'max_cache_size': 100 * 1024 * 1024  # 100MB
}

# Cross-Platform Configuration
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

# Performance Optimization Settings
PERFORMANCE_CONFIG = {
    'lazy_loading': True,
    'batch_processing': True,
    'max_batch_size': 8,
    'progress_indicators': True,
    'memory_optimization': True,
    'gpu_acceleration': False  # Set to True if GPU is available
}

# URL Source Patterns (extensible)
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

def get_cache_dir():
    """Get cache directory path, creating it if it doesn't exist"""
    cache_dir = Path(MODEL_CONFIG['cache_dir'])
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def get_model_weights_dir():
    """Get model weights directory path, creating it if it doesn't exist"""
    weights_dir = Path(MODEL_CONFIG['model_weights_dir'])
    weights_dir.mkdir(exist_ok=True)
    return weights_dir

def is_gpu_available():
    """Check if GPU acceleration is available"""
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except:
        return False

def get_optimal_batch_size():
    """Get optimal batch size based on available resources"""
    if PERFORMANCE_CONFIG['gpu_acceleration'] and is_gpu_available():
        return min(PERFORMANCE_CONFIG['max_batch_size'], 16)
    else:
        return min(PERFORMANCE_CONFIG['max_batch_size'], 4)

def get_platform_specific_settings():
    """Get platform-specific configuration"""
    import platform
    
    settings = PLATFORM_CONFIG.copy()
    
    if platform.system() == 'Windows':
        settings['ssl_verify'] = True
        settings['timeout'] = 45
    elif platform.system() == 'Darwin':  # macOS
        settings['ssl_verify'] = True
        settings['timeout'] = 30
    else:  # Linux and others
        settings['ssl_verify'] = True
        settings['timeout'] = 30
    
    return settings
