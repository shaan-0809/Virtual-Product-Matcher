import streamlit as st
import numpy as np
import json
import requests
from PIL import Image
import io
import pandas as pd
from utils.image_processor import ImageProcessor
from utils.embedding_generator import EmbeddingGenerator
from utils.similarity_search import SimilaritySearch

# Page configuration
st.set_page_config(
    page_title="Visual Product Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'embeddings_loaded' not in st.session_state:
    st.session_state.embeddings_loaded = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'query_image' not in st.session_state:
    st.session_state.query_image = None

# Load data and initialize components
@st.cache_data
def load_products():
    """Load product data from JSON file"""
    try:
        with open('data/products.json', 'r') as f:
            products = json.load(f)
        return pd.DataFrame(products)
    except FileNotFoundError:
        st.error("Product database not found. Please run the embedding generation script first.")
        return pd.DataFrame()

@st.cache_data
def load_embeddings():
    """Load precomputed product embeddings"""
    try:
        with open('data/product_embeddings.json', 'r') as f:
            embeddings_data = json.load(f)
        return embeddings_data
    except FileNotFoundError:
        st.error("Product embeddings not found. Please run the embedding generation script first.")
        return {}

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize image processor and embedding generator"""
    image_processor = ImageProcessor()
    embedding_generator = EmbeddingGenerator()
    similarity_search = SimilaritySearch()
    return image_processor, embedding_generator, similarity_search

def main():
    st.title("🔍 Visual Product Matcher")
    st.markdown("Find visually similar products by uploading an image or providing a URL")
    
    # Load data
    products_df = load_products()
    embeddings_data = load_embeddings()
    
    if products_df.empty or not embeddings_data:
        st.warning("Please run the embedding generation script to populate the product database.")
        st.code("python scripts/generate_embeddings.py")
        return
    
    # Initialize components
    image_processor, embedding_generator, similarity_search = initialize_components()
    
    # Load model with progress
    if not st.session_state.model_loaded:
        with st.spinner("Loading embedding model..."):
            embedding_generator.load_model()
            st.session_state.model_loaded = True
        st.success("Model loaded successfully!")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Category filter
    categories = sorted(products_df['category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Categories",
        categories,
        default=categories
    )
    
    # Brand filter
    brands = sorted(products_df['brand'].unique())
    selected_brands = st.sidebar.multiselect(
        "Brands",
        brands,
        default=brands
    )
    
    # Price range filter
    min_price = float(products_df['price'].min())
    max_price = float(products_df['price'].max())
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )
    
    # Similarity threshold
    similarity_threshold = st.sidebar.slider(
        "Minimum Similarity (%)",
        min_value=0,
        max_value=100,
        value=20
    ) / 100.0
    
    # Sort options
    sort_option = st.sidebar.selectbox(
        "Sort by",
        ["Highest Similarity", "Price (Low to High)", "Price (High to Low)", "Name"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Image")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload a product image to find similar items"
        )
        
        st.subheader("Or Enter Image URL")
        image_url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
        
        # Process image
        query_image = None
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file)
            st.session_state.query_image = query_image
        elif image_url:
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                query_image = Image.open(io.BytesIO(response.content))
                st.session_state.query_image = query_image
            except Exception as e:
                st.error(f"Failed to load image from URL: {str(e)}")
        
        # Display query image
        if query_image:
            st.image(query_image, caption="Query Image", use_column_width=True)
            
            # Search button
            if st.button("🔍 Find Similar Products", type="primary"):
                with st.spinner("Computing image embedding..."):
                    try:
                        # Process image and compute embedding
                        processed_image = image_processor.preprocess_image(query_image)
                        query_embedding = embedding_generator.generate_embedding(processed_image)
                        
                        # Perform similarity search
                        results = similarity_search.find_similar_products(
                            query_embedding, 
                            embeddings_data, 
                            products_df,
                            threshold=similarity_threshold
                        )
                        
                        st.session_state.search_results = results
                        st.success(f"Found {len(results)} similar products!")
                        
                    except Exception as e:
                        st.error(f"Error during search: {str(e)}")
    
    with col2:
        st.subheader("Search Results")
        
        if st.session_state.search_results is not None:
            results = st.session_state.search_results
            
            # Filter results
            filtered_results = results[
                (results['category'].isin(selected_categories)) &
                (results['brand'].isin(selected_brands)) &
                (results['price'] >= price_range[0]) &
                (results['price'] <= price_range[1]) &
                (results['similarity'] >= similarity_threshold)
            ]
            
            # Sort results
            if sort_option == "Highest Similarity":
                filtered_results = filtered_results.sort_values(by='similarity', ascending=False)
            elif sort_option == "Price (Low to High)":
                filtered_results = filtered_results.sort_values(by='price', ascending=True)
            elif sort_option == "Price (High to Low)":
                filtered_results = filtered_results.sort_values(by='price', ascending=False)
            elif sort_option == "Name":
                filtered_results = filtered_results.sort_values(by='name', ascending=True)
            
            if filtered_results.empty:
                st.info("No products match the current filters. Try adjusting your criteria.")
            else:
                # Display results in a grid
                cols_per_row = 3
                rows = len(filtered_results) // cols_per_row + (1 if len(filtered_results) % cols_per_row else 0)
                
                for row in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        idx = row * cols_per_row + col_idx
                        if idx < len(filtered_results):
                            product = filtered_results.iloc[idx]
                            with cols[col_idx]:
                                display_product_card(product)
        else:
            st.info("Upload an image or enter an image URL to start searching for similar products.")

def display_product_card(product):
    """Display a product card with image and details"""
    try:
        # Create a container for the product card
        with st.container():
            # Display product image
            if product['imageUrl'].startswith('http'):
                st.image(product['imageUrl'], use_column_width=True)
            else:
                # Local image path
                try:
                    st.image(product['imageUrl'], use_column_width=True)
                except:
                    st.write("🖼️ Image not available")
            
            # Product details
            st.write(f"**{product['name']}**")
            st.write(f"Brand: {product['brand']}")
            st.write(f"Category: {product['category']}")
            st.write(f"Price: ${product['price']:.2f}")
            
            # Similarity score
            if 'similarity' in product:
                similarity_percent = product['similarity'] * 100
                st.write(f"**Similarity: {similarity_percent:.1f}%**")
                st.progress(product['similarity'])
            
            # View details button
            if st.button(f"View Details", key=f"btn_{product['id']}"):
                show_product_details(product)
            
            st.divider()
    
    except Exception as e:
        st.error(f"Error displaying product: {str(e)}")

@st.dialog("Product Details")
def show_product_details(product):
    """Show detailed product information in a modal"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if product['imageUrl'].startswith('http'):
            st.image(product['imageUrl'], use_column_width=True)
        else:
            try:
                st.image(product['imageUrl'], use_column_width=True)
            except:
                st.write("🖼️ Image not available")
    
    with col2:
        st.write(f"**Name:** {product['name']}")
        st.write(f"**Brand:** {product['brand']}")
        st.write(f"**Category:** {product['category']}")
        st.write(f"**Price:** ${product['price']:.2f}")
        
        if 'similarity' in product:
            similarity_percent = product['similarity'] * 100
            st.write(f"**Similarity:** {similarity_percent:.1f}%")
        
        if 'tags' in product and product['tags']:
            st.write(f"**Tags:** {', '.join(product['tags'])}")
        
        if 'description' in product:
            st.write(f"**Description:** {product['description']}")

if __name__ == "__main__":
    main()
