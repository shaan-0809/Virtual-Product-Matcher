import tensorflow as tf
import numpy as np
import os
import pickle
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# Import configuration
try:
    from config.performance import MODEL_CONFIG, PERFORMANCE_CONFIG
except ImportError:
    # Fallback configuration if config file is not available
    MODEL_CONFIG = {
        'default_model': 'mobilenet',
        'alternative_model': 'efficientnet_small',
        'use_cache': True,
        'cache_dir': 'cache',
        'model_weights_dir': 'models'
    }
    PERFORMANCE_CONFIG = {
        'lazy_loading': True,
        'batch_processing': True,
        'max_batch_size': 8,
        'progress_indicators': True,
        'memory_optimization': True,
        'gpu_acceleration': False
    }

class EmbeddingGenerator:
    """Generates embeddings using optimized pre-trained models with caching"""
    
    def __init__(self, model_type=None, use_cache=None):
        self.model = None
        self.feature_extractor = None
        self.model_type = model_type or MODEL_CONFIG['default_model']
        self.use_cache = use_cache if use_cache is not None else MODEL_CONFIG['use_cache']
        self.cache_dir = Path(MODEL_CONFIG['cache_dir'])
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model configurations for different sizes
        self.model_configs = {
            'mobilenet': {
                'class': MobileNetV2,
                'preprocess': mobilenet_preprocess,
                'input_shape': (224, 224, 3),
                'weights': 'imagenet',
                'pooling': 'avg'
            },
            'efficientnet_small': {
                'class': EfficientNetB0,
                'preprocess': efficientnet_preprocess,
                'input_shape': (224, 224, 3),
                'weights': 'imagenet',
                'pooling': 'avg'
            }
        }
    
    def _get_model_config(self):
        """Get configuration for the selected model type"""
        return self.model_configs.get(self.model_type, self.model_configs['mobilenet'])
    
    def _load_cached_model(self):
        """Load cached model if available"""
        if not self.use_cache:
            return None
            
        cache_file = self.cache_dir / f"{self.model_type}_model.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def _save_cached_model(self, model):
        """Save model to cache"""
        if not self.use_cache:
            return
            
        try:
            cache_file = self.cache_dir / f"{self.model_type}_model.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f)
        except Exception:
            pass  # Silently fail if caching fails
    
    def load_model(self, force_reload=False):
        """Load pre-trained model with caching and lazy loading"""
        if self.feature_extractor is not None and not force_reload:
            return True
            
        try:
            # Try to load from cache first
            if not force_reload:
                cached_model = self._load_cached_model()
                if cached_model is not None:
                    self.feature_extractor = cached_model
                    self.model = cached_model
                    return True
            
            # Load model configuration
            config = self._get_model_config()
            
            # Load base model without top classification layer
            base_model = config['class'](
                input_shape=config['input_shape'],
                include_top=False,
                weights=config['weights'],
                pooling=config['pooling']
            )
            
            # Freeze the base model for inference
            base_model.trainable = False
            
            # Compile model for better inference performance
            base_model.compile(optimizer='adam', loss='mse')
            
            self.feature_extractor = base_model
            self.model = base_model
            
            # Cache the model
            self._save_cached_model(base_model)
            
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
            
            # Get preprocessing function
            config = self._get_model_config()
            preprocess_func = config['preprocess']
            
            # Apply preprocessing
            if self.model_type == 'mobilenet':
                processed_image = preprocess_func(preprocessed_image * 255.0)
            else:
                processed_image = preprocess_func(preprocessed_image)
            
            # Generate embedding with optimized settings
            embedding = self.feature_extractor.predict(
                processed_image, 
                verbose=0,
                batch_size=1
            )
            
            # Flatten and normalize
            embedding = embedding.flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # Add epsilon for stability
            
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
                self.load_model()
            
            # Get preprocessing function
            config = self._get_model_config()
            preprocess_func = config['preprocess']
            
            # Apply preprocessing
            if self.model_type == 'mobilenet':
                processed_batch = preprocess_func(image_batch * 255.0)
            else:
                processed_batch = preprocess_func(image_batch)
            
            # Generate embeddings with optimized batch processing
            batch_size = min(len(image_batch), PERFORMANCE_CONFIG['max_batch_size'])
            embeddings = self.feature_extractor.predict(
                processed_batch, 
                verbose=0,
                batch_size=batch_size
            )
            
            # Normalize each embedding
            embeddings_normalized = []
            for embedding in embeddings:
                embedding = embedding.flatten()
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                embeddings_normalized.append(embedding)
            
            return np.array(embeddings_normalized)
            
        except Exception as e:
            raise Exception(f"Error generating batch embeddings: {str(e)}")
    
    def switch_model(self, model_type):
        """Switch to a different model type"""
        if model_type in self.model_configs and model_type != self.model_type:
            self.model_type = model_type
            self.feature_extractor = None
            self.model = None
            return True
        return False
    
    def get_model_info(self):
        """Get information about the current model"""
        if self.feature_extractor is None:
            return {"status": "Not loaded", "type": self.model_type}
        
        config = self._get_model_config()
        return {
            "status": "Loaded",
            "type": self.model_type,
            "input_shape": config['input_shape'],
            "cached": self.use_cache
        }
