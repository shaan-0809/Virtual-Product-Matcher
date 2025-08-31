# Visual Product Matcher

A powerful Streamlit-based web application that uses machine learning embeddings to find visually similar products based on uploaded images or image URLs.

##  Performance Optimizations

This application has been optimized for **faster loading times** and **better cross-platform compatibility**:

###  Loading Time Improvements
- **Lazy Model Loading**: AI model loads only when needed
- **Model Caching**: Pre-trained models are cached for instant reuse
- **Lighter Model Options**: Choose between MobileNet (fast) and EfficientNet (accurate)
- **Enhanced Caching**: Product data and embeddings cached with TTL
- **Batch Processing**: Optimized image processing for multiple images

###  Cross-Platform Support
- **Windows**: Optimized SSL configuration and GPU detection
- **macOS**: Metal Performance Shaders support
- **Linux**: GPU memory management and optimization
- **Universal Startup Scripts**: Easy deployment on any platform

###  Multi-Source URL Support
- **Social Media**: Instagram, Facebook, Twitter, Pinterest
- **Search Engines**: Google Images, Bing Images
- **E-commerce**: Amazon, eBay
- **Direct URLs**: Any image URL with automatic fallback methods

##  Features

* **Image Upload**: Drag-and-drop interface for uploading product images
* **URL Input**: Support for external image URLs from multiple sources
* **Visual Similarity Search**: AI-powered similarity matching using TensorFlow embeddings
* **Advanced Filtering**: Filter by category, brand, price range, and similarity threshold
* **Sorting Options**: Sort results by similarity, price, or name
* **Product Database**: 50+ diverse products across 8 categories
* **Responsive Design**: Mobile-friendly interface with loading states
* **Real-time Results**: Interactive product cards with similarity scores

##  Tech Stack

* **Frontend**: Streamlit (Python web framework)
* **ML/AI**: TensorFlow with MobileNetV2/EfficientNet for image embeddings
* **Image Processing**: PIL/Pillow for image manipulation
* **Similarity Search**: Scikit-learn for cosine similarity calculations
* **Data Management**: Pandas for data manipulation, JSON for storage
* **Cross-Platform**: Platform-specific optimizations and SSL handling

##  Quick Start

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Installation

1. **Clone or download the project files**
2. **Install required dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Generate product embeddings** (required first-time setup):  
   ```bash
   python scripts/generate_embeddings.py
   ```
   This script will:
   * Download product images from URLs  
   * Generate ML embeddings for each product  
   * Save embeddings to `data/product_embeddings.json`  
   * Takes approximately 5-10 minutes depending on internet speed

### 🖥️ Cross-Platform Startup

#### Windows Users
```bash
# Double-click the batch file
start_app.bat

# Or use the startup script
python start_app.py --host 0.0.0.0
```

#### macOS/Linux Users
```bash
# Make script executable and run
chmod +x start_app.sh
./start_app.sh

# Or use the startup script directly
python3 start_app.py --host 0.0.0.0
```

#### Universal Method
```bash
# Use the Python startup script on any platform
python start_app.py --port 5000 --host 0.0.0.0
```

### Access the Application

* Open your browser and navigate to `http://localhost:5000`
* Upload an image or enter an image URL to find similar products

## ⚙️ Performance Configuration

### Model Selection
- **MobileNetV2**: Fastest loading, good accuracy (default)
- **EfficientNet-B0**: Slower loading, better accuracy

### Caching Options
- **Model Cache**: Saves pre-trained models for instant reuse
- **Data Cache**: Caches product data and embeddings
- **Image Cache**: Caches processed images

### Platform-Specific Optimizations
The application automatically detects your platform and applies optimizations:

- **Windows**: GPU detection, optimized SSL
- **macOS**: Metal GPU support, optimized memory
- **Linux**: GPU memory management, batch processing

## 🔧 Advanced Configuration

### Custom URL Patterns
Add support for new platforms by editing `config/performance.py`:

```python
URL_PATTERNS['new_platform'] = [
    r'newplatform\.com.*[?&]url=([^&]+)',
    r'newplatform\.com.*[?&]media=([^&]+)'
]
```

### Performance Tuning
Modify `config/performance.py` for your hardware:

```python
PERFORMANCE_CONFIG = {
    'lazy_loading': True,
    'batch_processing': True,
    'max_batch_size': 16,  # Increase for GPU
    'gpu_acceleration': True  # Enable if GPU available
}
```

### Environment Variables
Set these for additional optimizations:

```bash
# Reduce TensorFlow logging
export TF_CPP_MIN_LOG_LEVEL=2

# Enable GPU memory growth (Linux)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Optimize NumPy threading
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```



*Times may vary based on hardware specifications and internet connection*

##  Troubleshooting

### Common Issues

#### Model Loading Fails
```bash
# Clear model cache
rm -rf cache/
# Restart application
python start_app.py
```

#### URL Loading Issues
- Check internet connection
- Verify URL is accessible
- Try different image formats
- Use direct image URLs when possible

#### Performance Issues
- Enable GPU acceleration if available
- Reduce batch size in configuration
- Clear cache directories
- Check available memory

### Platform-Specific Issues

#### Windows
- Ensure Python is in PATH
- Install Visual C++ Redistributable
- Check Windows Defender exclusions

#### macOS
- Install Xcode Command Line Tools
- Check Security & Privacy settings
- Verify Python installation

#### Linux
- Install system dependencies
- Check GPU drivers
- Verify CUDA installation (if using GPU)

## 🔄 Updates and Maintenance

### Regular Maintenance
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Clear old caches
rm -rf cache/
rm -rf __pycache__/

# Regenerate embeddings if needed
python scripts/generate_embeddings.py
```

### Performance Monitoring
The application includes built-in performance monitoring:
- Model loading times
- Search response times
- Memory usage tracking
- Platform-specific optimizations

##  Project Structure

```
PythonChatbot/
├── app.py                          # Main Streamlit application
├── start_app.py                    # Cross-platform startup script
├── start_app.bat                   # Windows startup script
├── start_app.sh                    # Unix/Linux/macOS startup script
├── config/
│   └── performance.py             # Performance configuration
├── data/
│   ├── products.json              # Product database
│   └── product_embeddings.json    # Precomputed embeddings
├── scripts/
│   └── generate_embeddings.py     # Embedding generation script
├── utils/
│   ├── embedding_generator.py     # AI model management
│   ├── image_processor.py         # Image processing and URL handling
│   └── similarity_search.py       # Similarity search algorithms
├── cache/                          # Model and data cache
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on multiple platforms
5. Submit a pull request

##  License

This project is open source and available under the [MIT License](LICENSE).

##  Support

For issues and questions:
1. Check the troubleshooting section
2. Review platform-specific configurations
3. Check GitHub Issues
4. Create a new issue with platform details

---

**Performance Tip**: The first run will be slower due to model downloading and caching. Subsequent runs will be significantly faster!

MADE BY SHAAN DUBEY-2201641530157