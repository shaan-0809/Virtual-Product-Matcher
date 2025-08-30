#!/usr/bin/env python3
"""
Cross-platform startup script for Visual Product Matcher
This script handles platform-specific configurations and optimizations
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'tensorflow', 'pillow', 'numpy', 
        'pandas', 'scikit-learn', 'requests', 'beautifulsoup4'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages are installed")
    return True

def setup_platform_specific():
    """Setup platform-specific configurations"""
    system = platform.system()
    print(f"\n🖥️  Detected platform: {system}")
    
    if system == "Windows":
        setup_windows()
    elif system == "Darwin":  # macOS
        setup_macos()
    elif system == "Linux":
        setup_linux()
    else:
        print(f"⚠️  Unknown platform: {system}")
        print("Using default configuration")

def setup_windows():
    """Setup Windows-specific configurations"""
    print("🔧 Configuring for Windows...")
    
    # Set environment variables for better performance
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available
    
    # Create cache directory
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    print(f"📁 Cache directory: {cache_dir.absolute()}")

def setup_macos():
    """Setup macOS-specific configurations"""
    print("🔧 Configuring for macOS...")
    
    # Set environment variables for better performance
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Check for Metal Performance Shaders
    try:
        import tensorflow as tf
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print("🚀 Metal GPU acceleration available")
        else:
            print("💻 Using CPU for inference")
    except:
        print("💻 Using CPU for inference")
    
    # Create cache directory
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    print(f"📁 Cache directory: {cache_dir.absolute()}")

def setup_linux():
    """Setup Linux-specific configurations"""
    print("🔧 Configuring for Linux...")
    
    # Set environment variables for better performance
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Check for GPU availability
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            print(f"🚀 GPU acceleration available: {len(gpus)} device(s)")
            # Enable memory growth to avoid GPU memory issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("💻 Using CPU for inference")
    except:
        print("💻 Using CPU for inference")
    
    # Create cache directory
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    print(f"📁 Cache directory: {cache_dir.absolute()}")

def check_data_files():
    """Check if required data files exist"""
    print("\n📊 Checking data files...")
    
    required_files = [
        'data/products.json',
        'data/product_embeddings.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    if missing_files:
        print(f"\n⚠️  Missing data files: {', '.join(missing_files)}")
        print("Run the embedding generation script first:")
        print("python scripts/generate_embeddings.py")
        return False
    
    print("✅ All required data files are present")
    return True

def optimize_performance():
    """Apply performance optimizations"""
    print("\n⚡ Applying performance optimizations...")
    
    # Set TensorFlow optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
    os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = '1'  # Enable MKL optimizations
    
    # Set NumPy optimizations
    os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent NumPy from using all cores
    os.environ['MKL_NUM_THREADS'] = '1'
    
    print("✅ Performance optimizations applied")

def start_streamlit(port=5000, host='localhost'):
    """Start the Streamlit application"""
    print(f"\n🚀 Starting Visual Product Matcher on {host}:{port}")
    print("📱 Open your browser and navigate to the URL above")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        # Start Streamlit with optimized settings
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', str(port),
            '--server.address', host,
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            '--global.developmentMode', 'false'
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description='Start Visual Product Matcher')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the application on')
    parser.add_argument('--host', default='localhost', help='Host to bind the application to')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and data file checks')
    
    args = parser.parse_args()
    
    print("🎯 Visual Product Matcher - Cross-Platform Startup")
    print("=" * 50)
    
    if not args.skip_checks:
        # Perform startup checks
        check_python_version()
        
        if not check_dependencies():
            sys.exit(1)
        
        if not check_data_files():
            sys.exit(1)
    
    # Setup platform-specific configurations
    setup_platform_specific()
    
    # Apply performance optimizations
    optimize_performance()
    
    # Start the application
    start_streamlit(args.port, args.host)

if __name__ == "__main__":
    main()
