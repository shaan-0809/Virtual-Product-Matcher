import numpy as np
from PIL import Image

class ImageProcessor:
    """Handles image preprocessing for embedding generation"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def preprocess_image(self, image):
        """
        Preprocess image for embedding generation
        
        Args:
            image: PIL Image object
            
        Returns:
            numpy array: Preprocessed image ready for model input
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Normalize pixel values to [0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def load_image_from_path(self, image_path):
        """
        Load and preprocess image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array: Preprocessed image
        """
        try:
            image = Image.open(image_path)
            return self.preprocess_image(image)
        except Exception as e:
            raise Exception(f"Error loading image from {image_path}: {str(e)}")
    
    def validate_image(self, image):
        """
        Validate image format and size
        
        Args:
            image: PIL Image object
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if image is valid
            if not isinstance(image, Image.Image):
                return False
            
            # Check image size (not too small)
            width, height = image.size
            if width < 32 or height < 32:
                return False
            
            # Check file size (assume reasonable limits)
            if width * height > 10000000:  # 10MP limit
                return False
            
            return True
            
        except Exception:
            return False
