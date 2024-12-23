# embeddings.py
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


# embeddings.py
def get_embeddings(text: str, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> List[float]:
    """Generate embeddings for a given text with error handling."""
    try:
        model = SentenceTransformer(model_name)
        embedding = model.encode(text).tolist()
        if not embedding or len(embedding) == 0:
            raise ValueError("Generated embedding is empty")
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise