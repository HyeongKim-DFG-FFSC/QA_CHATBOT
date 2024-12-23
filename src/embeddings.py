# embeddings.py
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


def get_embeddings(text: str, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> List[float]:
    """Generate embeddings for a given text."""
    model = SentenceTransformer(model_name)
    return model.encode(text).tolist()