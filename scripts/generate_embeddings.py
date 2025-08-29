#!/usr/bin/env python3
"""
Script to generate embeddings for product images and save them to JSON files.
This script should be run once to precompute all product embeddings.
"""

import json
import numpy as np
import requests
from PIL import Image
import io
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processor import ImageProcessor
from utils.embedding_generator import EmbeddingGenerator

def download_image(url, timeout=10):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None

def generate_all_embeddings():
    """Generate embeddings for all products and save to JSON"""
    
    print("Loading products...")
    try:
        with open('data/products.json', 'r') as f:
            products = json.load(f)
    except FileNotFoundError:
        print("Error: products.json not found. Please ensure the file exists.")
        return
    
    print("Initializing components...")
    image_processor = ImageProcessor()
    embedding_generator = EmbeddingGenerator()
    
    print("Loading embedding model...")
    try:
        embedding_generator.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    embeddings_data = {}
    successful_count = 0
    failed_count = 0
    
    print(f"Processing {len(products)} products...")
    
    for i, product in enumerate(products):
        product_id = product['id']
        image_url = product['imageUrl']
        product_name = product['name']
        
        print(f"Processing {i+1}/{len(products)}: {product_name}")
        
        try:
            # Download image
            image = download_image(image_url)
            if image is None:
                print(f"  Failed to download image for product {product_id}")
                failed_count += 1
                continue
            
            # Validate image
            if not image_processor.validate_image(image):
                print(f"  Invalid image for product {product_id}")
                failed_count += 1
                continue
            
            # Preprocess image
            processed_image = image_processor.preprocess_image(image)
            
            # Generate embedding
            embedding = embedding_generator.generate_embedding(processed_image)
            
            # Convert to list for JSON serialization
            embeddings_data[str(product_id)] = embedding.tolist()
            
            successful_count += 1
            print(f"  ✓ Successfully processed embedding (shape: {embedding.shape})")
            
        except Exception as e:
            print(f"  ✗ Error processing product {product_id}: {str(e)}")
            failed_count += 1
            continue
    
    # Save embeddings to file
    print(f"\nSaving embeddings to data/product_embeddings.json...")
    try:
        with open('data/product_embeddings.json', 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        print("✓ Embeddings saved successfully!")
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")
        return
    
    print(f"\nSummary:")
    print(f"  Total products: {len(products)}")
    print(f"  Successfully processed: {successful_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Success rate: {successful_count/len(products)*100:.1f}%")
    
    if successful_count > 0:
        # Test loading the saved embeddings
        print(f"\nTesting saved embeddings...")
        try:
            with open('data/product_embeddings.json', 'r') as f:
                loaded_embeddings = json.load(f)
            
            sample_embedding = np.array(list(loaded_embeddings.values())[0])
            print(f"✓ Embeddings loaded successfully!")
            print(f"  Sample embedding shape: {sample_embedding.shape}")
            print(f"  Embedding dimension: {len(sample_embedding)}")
            
        except Exception as e:
            print(f"✗ Error testing saved embeddings: {str(e)}")
    
    print(f"\nEmbedding generation complete!")
    print(f"You can now run the Streamlit app: streamlit run app.py")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("=== Product Embedding Generator ===")
    print("This script will generate embeddings for all products in the database.")
    print("Make sure you have a stable internet connection to download product images.")
    print()
    
    # Check if embeddings already exist
    if os.path.exists('data/product_embeddings.json'):
        response = input("Embeddings file already exists. Regenerate? (y/N): ")
        if response.lower() != 'y':
            print("Skipping embedding generation.")
            sys.exit(0)
    
    generate_all_embeddings()
