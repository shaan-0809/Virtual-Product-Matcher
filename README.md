# Visual Product Matcher

A powerful Streamlit-based web application that uses machine learning embeddings to find visually similar products based on uploaded images or image URLs.

## Features

- **Image Upload**: Drag-and-drop interface for uploading product images
- **URL Input**: Support for external image URLs
- **Visual Similarity Search**: AI-powered similarity matching using TensorFlow embeddings
- **Advanced Filtering**: Filter by category, brand, price range, and similarity threshold
- **Sorting Options**: Sort results by similarity, price, or name
- **Product Database**: 50+ diverse products across 8 categories
- **Responsive Design**: Mobile-friendly interface with loading states
- **Real-time Results**: Interactive product cards with similarity scores

## Tech Stack

- **Frontend**: Streamlit (Python web framework)
- **ML/AI**: TensorFlow with MobileNetV2 for image embeddings
- **Image Processing**: PIL/Pillow for image manipulation
- **Similarity Search**: Scikit-learn for cosine similarity calculations
- **Data Management**: Pandas for data manipulation, JSON for storage

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**

2. **Install required dependencies**:
   ```bash
   pip install streamlit tensorflow pillow numpy pandas scikit-learn requests
   ```

3. **Generate product embeddings** (required first-time setup):
   ```bash
   python scripts/generate_embeddings.py
   ```
   
   This script will:
   - Download product images from URLs
   - Generate ML embeddings for each product
   - Save embeddings to `data/product_embeddings.json`
   - Takes approximately 5-10 minutes depending on internet speed

4. **Run the application**:
   ```bash
   streamlit run app.py --server.port 5000
   ```

5. **Access the app**:
   - Open your browser and navigate to `http://localhost:5000`
   - Upload an image or enter an image URL to find similar products

## Project Structure

