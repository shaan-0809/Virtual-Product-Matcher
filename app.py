import streamlit as st
import numpy as np
import json
import requests
from PIL import Image
import io
import pandas as pd
import time
from utils.image_processor import ImageProcessor
from utils.embedding_generator import EmbeddingGenerator
from utils.similarity_search import SimilaritySearch

# Page configuration
st.set_page_config(
    page_title="Visual Product Matcher",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better mobile experience and performance
st.markdown("""
<style>
    /* Mobile responsiveness improvements */
    @media (max-width: 768px) {
        .stSelectbox > div > div {
            font-size: 14px;
        }
        .stButton > button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
        .stProgress > div {
            margin: 0.25rem 0;
        }
    }
    
    /* Better product card styling */
    .stContainer {
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Performance optimizations */
    .stImage img {
        max-width: 100%;
        height: auto;
    }
    
    /* Loading states */
    .loading-overlay {
        position: relative;
        opacity: 0.7;
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with performance optimizations
if 'embeddings_loaded' not in st.session_state:
    st.session_state.embeddings_loaded = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'query_image' not in st.session_state:
    st.session_state.query_image = None
if 'components_initialized' not in st.session_state:
    st.session_state.components_initialized = False
if 'model_loading_started' not in st.session_state:
    st.session_state.model_loading_started = False

# Load data and initialize components with enhanced caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_products():
    """Load product data from JSON file with caching"""
    try:
        with open('data/products.json', 'r') as f:
            products = json.load(f)
        return pd.DataFrame(products)
    except FileNotFoundError:
        st.error("Product database not found. Please run the embedding generation script first.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_embeddings():
    """Load precomputed product embeddings with caching"""
    try:
        with open('data/product_embeddings.json', 'r') as f:
            embeddings_data = json.load(f)
        return embeddings_data
    except FileNotFoundError:
        st.error("Product embeddings not found. Please run the embedding generation script first.")
        return {}

# Initialize components with lazy loading
@st.cache_resource
def initialize_components():
    """Initialize image processor and embedding generator with lazy loading"""
    if not st.session_state.components_initialized:
        image_processor = ImageProcessor()
        embedding_generator = EmbeddingGenerator(model_type='mobilenet', use_cache=True)
        similarity_search = SimilaritySearch()
        st.session_state.components_initialized = True
        return image_processor, embedding_generator, similarity_search
    else:
        # Return cached components if already initialized
        return st.session_state.get('cached_components', (None, None, None))

def load_model_with_progress(embedding_generator):
    """Load model with detailed progress tracking"""
    if st.session_state.model_loaded:
        return True
    
    if not st.session_state.model_loading_started:
        st.session_state.model_loading_started = True
        
        # Create progress container
        progress_container = st.container()
        with progress_container:
            st.markdown("### Loading AI Model...")
            
            # Progress bar for model loading
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress steps
            steps = [
                "Initializing TensorFlow...",
                "Downloading model weights...",
                "Loading MobileNetV2...",
                "Compiling model...",
                "Model ready!"
            ]
            
            for i, step in enumerate(steps):
                status_text.text(step)
                progress_bar.progress((i + 1) / len(steps))
                time.sleep(0.5)  # Simulate loading time
            
            # Actually load the model
            try:
                embedding_generator.load_model()
                st.session_state.model_loaded = True
                st.success("✓ Model loaded successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")
                return False
            finally:
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
    
    return st.session_state.model_loaded

def main():
    st.title("Visual Product Matcher")
    st.markdown("Find visually similar products by uploading an image or providing a URL")
    
    # Performance optimization: Load data in parallel
    with st.spinner("Loading product database..."):
        products_df = load_products()
        embeddings_data = load_embeddings()
    
    if products_df.empty or not embeddings_data:
        st.warning("Please run the embedding generation script to populate the product database.")
        st.code("python scripts/generate_embeddings.py")
        return
    
    # Initialize components (lazy loading)
    image_processor, embedding_generator, similarity_search = initialize_components()
    
    # Cache components for reuse
    if not st.session_state.get('cached_components'):
        st.session_state.cached_components = (image_processor, embedding_generator, similarity_search)
    
    # Load model with progress tracking
    if not load_model_with_progress(embedding_generator):
        st.error("Failed to load the AI model. Please refresh the page and try again.")
        return
    
    # Mobile layout toggle (at the top for better UX)
    mobile_layout = st.checkbox("Mobile Layout", value=False, help="Check this for better mobile experience")
    
    # Sidebar filters - mobile responsive
    if mobile_layout:
        st.header("Filters")
        with st.expander("Filter Options", expanded=False):
            filter_container = st.container()
    else:
        st.sidebar.header("Filters")
        filter_container = st.sidebar
    
    # Filters in responsive container
    with filter_container:
        # Category filter
        categories = sorted(products_df['category'].unique())
        selected_categories = st.multiselect(
            "Categories",
            categories,
            default=categories
        )
        
        # Brand filter
        brands = sorted(products_df['brand'].unique())
        selected_brands = st.multiselect(
            "Brands",
            brands,
            default=brands
        )
        
        # Price range filter
        min_price = float(products_df['price'].min())
        max_price = float(products_df['price'].max())
        price_range = st.slider(
            "Price Range ($)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )
        
        # Similarity threshold
        similarity_threshold = st.slider(
            "Minimum Similarity (%)",
            min_value=0,
            max_value=100,
            value=20
        ) / 100.0
        
        # Sort options
        sort_option = st.selectbox(
            "Sort by",
            ["Highest Similarity", "Price (Low to High)", "Price (High to Low)", "Name"]
        )
        
        # Model selection for advanced users
        if st.checkbox("Advanced Options", help="Show advanced model settings"):
            model_type = st.selectbox(
                "Model Type",
                ["mobilenet", "efficientnet_small"],
                help="MobileNet is faster, EfficientNet is more accurate"
            )
            
            if st.button("Switch Model"):
                if embedding_generator.switch_model(model_type):
                    st.session_state.model_loaded = False
                    st.session_state.model_loading_started = False
                    st.success(f"Switched to {model_type} model. Please wait for reload.")
                    st.rerun()
        
        # Show model info
        if embedding_generator:
            model_info = embedding_generator.get_model_info()
            st.info(f"Model: {model_info['type']} ({model_info['status']})")
    
    # Main content area - responsive layout
    if mobile_layout:
        # Mobile: Stack vertically
        st.subheader("Upload Image")
    else:
        # Desktop: Side by side
        col1, col2 = st.columns([1, 2])
    
    # Upload section
    upload_container = st.container() if mobile_layout else col1
    with upload_container:
        if not mobile_layout:
            st.subheader("Upload Image")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload a product image to find similar items"
        )
        
        st.subheader("Or Enter Image URL")
        image_url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
        
        # Show supported platforms
        with st.expander("Supported URL Sources", expanded=False):
            st.write("This app can extract images from various platforms:")
            platforms = image_processor.get_supported_platforms()
            for platform in platforms:
                st.write(f"• {platform.title()}")
            st.info("Tip: Right-click on any image and select 'Copy image address' for best results")
        
        # Process image
        query_image = None
        image_source = None
        
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file)
            st.session_state.query_image = query_image
            image_source = "upload"
        elif image_url.strip():
            # Sanitize pasted URLs (handle leading '@', spaces)
            sanitized = image_url.strip().lstrip('@').strip()
            if not sanitized.startswith(('http://', 'https://')):
                st.error("Please enter a valid URL starting with http:// or https://")
            else:
                try:
                    with st.spinner("Loading image from URL..."):
                        url = sanitized
                        try:
                            query_image = image_processor.load_image_from_url(url)
                            st.session_state.query_image = query_image
                            image_source = "url"
                            st.success("✓ Image loaded successfully!")
                        except Exception as img_error:
                            st.error(f"The URL does not contain a valid image. {img_error}")
                            st.info("Please try:\n- Right-click on an image and select 'Copy image address'\n- Use direct image URLs ending with .jpg, .png, .webp, etc.\n- Some redirect/tracking links are supported (Bing/Google), otherwise paste the direct image URL")
                            
                except Exception as e:
                    st.error(f"Failed to load image from URL: {str(e)}")
                    st.info("**Tips for image URLs:**\n- Right-click on any image and select 'Copy image address'\n- Use direct links to image files (.jpg, .png, .webp, etc.)\n- Avoid search result or gallery URLs")
        
        # Display query image
        if query_image:
            st.image(query_image, caption=f"Query Image ({'Uploaded' if image_source == 'upload' else 'From URL'})", use_container_width=True)
            
            # Search button
            if st.button("Find Similar Products", type="primary"):
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
    
    # Results section
    results_container = st.container() if mobile_layout else col2
    with results_container:
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
                # Display results in a responsive grid
                if mobile_layout:
                    cols_per_row = 1  # Single column on mobile
                else:
                    cols_per_row = 3  # Three columns on desktop
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
                st.image(product['imageUrl'], use_container_width=True)
            else:
                # Local image path
                try:
                    st.image(product['imageUrl'], use_container_width=True)
                except:
                    st.write("Image not available")
            
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
            st.image(product['imageUrl'], use_container_width=True)
        else:
            try:
                st.image(product['imageUrl'], use_container_width=True)
            except:
                st.write("Image not available")
    
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
