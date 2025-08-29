# Visual Product Matcher

## Overview

Visual Product Matcher is a Streamlit-based machine learning application that enables users to find visually similar products by uploading images or providing image URLs. The system uses TensorFlow's MobileNetV2 model to generate embeddings from product images and performs similarity searches using cosine similarity calculations. The application includes a curated database of 50+ products across multiple categories with advanced filtering and sorting capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for rapid web application development
- **Interface Design**: Drag-and-drop file upload with URL input alternative
- **State Management**: Streamlit session state for caching loaded models and search results
- **Responsive Design**: Mobile-friendly interface with loading states and interactive product cards

### Backend Architecture
- **Application Server**: Streamlit's built-in server handles routing and static file serving
- **Data Processing Pipeline**: 
  - Image preprocessing through PIL/Pillow
  - Feature extraction using TensorFlow MobileNetV2
  - Similarity calculation via scikit-learn's cosine similarity
- **Modular Design**: Utility classes for image processing, embedding generation, and similarity search

### Machine Learning Pipeline
- **Feature Extraction**: MobileNetV2 pre-trained model with global average pooling
- **Embedding Generation**: 1280-dimensional feature vectors from images
- **Similarity Calculation**: Cosine similarity for finding visually similar products
- **Preprocessing**: Image resizing to 224x224, RGB conversion, and normalization

### Data Storage Architecture
- **Product Database**: JSON file (`data/products.json`) storing product metadata
- **Embedding Storage**: JSON file (`data/product_embeddings.json`) for precomputed embeddings
- **Image Handling**: External URLs for product images with local caching during processing
- **Data Schema**: Products include id, name, category, price, brand, imageUrl, tags, and description

### Performance Optimizations
- **Precomputed Embeddings**: Product embeddings generated offline to reduce runtime processing
- **Caching Strategy**: Streamlit decorators cache loaded data and models
- **Batch Processing**: Embedding generation script processes all products in batches

## External Dependencies

### Machine Learning Libraries
- **TensorFlow**: Core ML framework for MobileNetV2 model and preprocessing
- **scikit-learn**: Cosine similarity calculations and ML utilities
- **NumPy**: Numerical computing for array operations and embedding manipulation

### Image Processing
- **PIL/Pillow**: Image loading, conversion, and preprocessing
- **OpenCV** (referenced): Computer vision operations (currently unused)

### Web Framework
- **Streamlit**: Complete web application framework with built-in components
- **Pandas**: Data manipulation and DataFrame operations for product filtering

### External Services
- **Unsplash**: Image hosting service providing product image URLs
- **HTTP Requests**: Image downloading from external URLs via requests library

### Data Management
- **JSON**: Product metadata and embedding storage format
- **File System**: Local storage for product database and precomputed embeddings

### Development Tools
- **Python 3.8+**: Runtime environment with pip package management
- **Scripts**: Standalone embedding generation for initial setup and updates