import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SimilaritySearch:
    """Handles similarity search using cosine similarity"""
    
    def __init__(self):
        pass
    
    def calculate_cosine_similarity(self, query_embedding, product_embeddings):
        """
        Calculate cosine similarity between query and product embeddings
        
        Args:
            query_embedding: Query image embedding (1D array)
            product_embeddings: Product embeddings (2D array)
            
        Returns:
            numpy array: Similarity scores
        """
        try:
            # Reshape query embedding for sklearn
            query_embedding = query_embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, product_embeddings)
            
            return similarities.flatten()
            
        except Exception as e:
            raise Exception(f"Error calculating similarity: {str(e)}")
    
    def find_similar_products(self, query_embedding, embeddings_data, products_df, 
                            top_k=100, threshold=0.0):
        """
        Find similar products based on embedding similarity
        
        Args:
            query_embedding: Query image embedding
            embeddings_data: Dictionary containing product embeddings
            products_df: DataFrame with product information
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            pandas DataFrame: Similar products with similarity scores
        """
        try:
            # Extract embeddings and product IDs
            product_ids = []
            embeddings_list = []
            
            for product_id, embedding in embeddings_data.items():
                product_ids.append(int(product_id))
                embeddings_list.append(embedding)
            
            if not embeddings_list:
                return pd.DataFrame()
            
            # Convert to numpy array
            product_embeddings = np.array(embeddings_list)
            
            # Calculate similarities
            similarities = self.calculate_cosine_similarity(query_embedding, product_embeddings)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'id': product_ids,
                'similarity': similarities
            })
            
            # Filter by threshold
            results_df = results_df[results_df['similarity'] >= threshold]
            
            # Sort by similarity
            results_df = results_df.sort_values(by='similarity', ascending=False)
            
            # Take top k results
            results_df = results_df.head(top_k)
            
            # Merge with product information
            results_df = results_df.merge(products_df, on='id', how='left')
            
            return results_df
            
        except Exception as e:
            raise Exception(f"Error finding similar products: {str(e)}")
    
    def batch_similarity_search(self, query_embeddings, embeddings_data, products_df):
        """
        Perform similarity search for multiple query embeddings
        
        Args:
            query_embeddings: Multiple query embeddings (2D array)
            embeddings_data: Dictionary containing product embeddings
            products_df: DataFrame with product information
            
        Returns:
            list: List of DataFrames with results for each query
        """
        try:
            results = []
            
            for query_embedding in query_embeddings:
                result = self.find_similar_products(
                    query_embedding, embeddings_data, products_df
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            raise Exception(f"Error in batch similarity search: {str(e)}")
