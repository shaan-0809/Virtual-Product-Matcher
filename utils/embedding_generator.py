import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import os
import pickle
from typing import Optional

class EmbeddingGenerator:
    """Generates embeddings using optimized pre-trained models with caching"""
    
    def __init__(self, model_type='efficientnet', cache_dir='./cache'):
        self.model = None
        self.feature_extractor = None
        self.model_type = model_type
        self.cache_dir = cache_dir
        self.model_cache_path = os.path.join(cache_dir, f'{model_type}_model.pkl')
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_model(self, force_reload=False):
        """Load pre-trained model with caching for faster loading"""
        try:
            # Check if cached model exists and load if not forcing reload
            if not force_reload and os.path.exists(self.model_cache_path):
                with open(self.model_cache_path, 'rb') as f:
                    cached_model = pickle.load(f)
                    self.feature_extractor = cached_model
                    self.model = cached_model
                    return True
            
            # Load model based on type
            if self.model_type == 'efficientnet':
                # Use EfficientNetB0 - lighter and faster than MobileNetV2
                base_model = EfficientNetB0(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg'
                )
                self.preprocess_func = efficientnet_preprocess
            else:
                # Fallback to MobileNetV2
                base_model = MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg'
                )
                self.preprocess_func = preprocess_input
            
            # Freeze the base model
            base_model.trainable = False
            
            self.feature_extractor = base_model
            self.model = base_model
            
            # Cache the model for future use
            try:
                with open(self.model_cache_path, 'wb') as f:
                    pickle.dump(base_model, f)
            except Exception as e:
                print(f"Warning: Could not cache model: {e}")
            
            return True
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def generate_embedding(self, preprocessed_image):
        """
        Generate embedding for preprocessed image
        
        Args:
            preprocessed_image: Numpy array of preprocessed image
            
        Returns:
            numpy array: Embedding vector
        """
        try:
            if self.feature_extractor is None:
                self.load_model()
            
            # Apply appropriate preprocessing
            processed_image = self.preprocess_func(preprocessed_image * 255.0)
            
            # Generate embedding with optimized settings
            embedding = self.feature_extractor.predict(
                processed_image, 
                verbose=0,
                batch_size=1
            )
            
            # Flatten if necessary and normalize
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)  # L2 normalization
            
            return embedding
            
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def generate_embeddings_batch(self, image_batch, batch_size=32):
        """
        Generate embeddings for a batch of images with optimized batching
        
        Args:
            image_batch: Numpy array of preprocessed images
            batch_size: Batch size for processing
            
        Returns:
            numpy array: Batch of embedding vectors
        """
        try:
            if self.feature_extractor is None:
                self.load_model()
            
            # Apply appropriate preprocessing
            processed_batch = self.preprocess_func(image_batch * 255.0)
            
            # Generate embeddings with optimized batch processing
            embeddings = self.feature_extractor.predict(
                processed_batch, 
                verbose=0,
                batch_size=batch_size
            )
            
            # Normalize each embedding
            embeddings_normalized = []
            for embedding in embeddings:
                embedding = embedding.flatten()
                embedding = embedding / np.linalg.norm(embedding)
                embeddings_normalized.append(embedding)
            
            return np.array(embeddings_normalized)
            
        except Exception as e:
            raise Exception(f"Error generating batch embeddings: {str(e)}")
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.feature_extractor is None:
            return {"status": "not_loaded"}
        
        return {
            "model_type": self.model_type,
            "input_shape": self.feature_extractor.input_shape,
            "output_shape": self.feature_extractor.output_shape,
            "parameters": self.feature_extractor.count_params()
        }
