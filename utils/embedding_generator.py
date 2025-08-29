import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class EmbeddingGenerator:
    """Generates embeddings using pre-trained MobileNetV2 model"""
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
    
    def load_model(self):
        """Load pre-trained MobileNetV2 model for feature extraction"""
        try:
            # Load MobileNetV2 without top classification layer
            base_model = MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet',
                pooling='avg'  # Global average pooling
            )
            
            # Freeze the base model
            base_model.trainable = False
            
            self.feature_extractor = base_model
            self.model = base_model
            
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
                raise Exception("Model not loaded. Call load_model() first.")
            
            # Apply MobileNetV2 preprocessing
            processed_image = preprocess_input(preprocessed_image * 255.0)
            
            # Generate embedding
            embedding = self.feature_extractor.predict(processed_image, verbose=0)
            
            # Flatten if necessary and normalize
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)  # L2 normalization
            
            return embedding
            
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def generate_embeddings_batch(self, image_batch):
        """
        Generate embeddings for a batch of images
        
        Args:
            image_batch: Numpy array of preprocessed images
            
        Returns:
            numpy array: Batch of embedding vectors
        """
        try:
            if self.feature_extractor is None:
                raise Exception("Model not loaded. Call load_model() first.")
            
            # Apply MobileNetV2 preprocessing
            processed_batch = preprocess_input(image_batch * 255.0)
            
            # Generate embeddings
            embeddings = self.feature_extractor.predict(processed_batch, verbose=0)
            
            # Normalize each embedding
            embeddings_normalized = []
            for embedding in embeddings:
                embedding = embedding.flatten()
                embedding = embedding / np.linalg.norm(embedding)
                embeddings_normalized.append(embedding)
            
            return np.array(embeddings_normalized)
            
        except Exception as e:
            raise Exception(f"Error generating batch embeddings: {str(e)}")
