from .model import *
from typing import Dict
import logging
logger = logging.getLogger(__name__)

# Registry dictionary mapping names to image embedding models
_IMAGE_EMBEDDING_MODEL_REGISTRY: Dict[str, callable] = {
    # Image Embedding Models
    "dinov3": dinov3,
    "dinov2": dinov2,
    "google_vit": google_vit,
    "ibot": ibot,
    "infloat_e5": infloat_e5,
}

_TEXT_EMBEDDING_MODEL_REGISTRY: Dict[str, callable] = {
    # Text Embedding Models
    "sentence_t5": sentence_t5,
    "gtr_t5": gtr_t5,
    "all_mpnet_base_v2": all_mpnet_base_v2,
    "alibaba_gte_en_v1_5": alibaba_gte_en_v1_5,
    "baai_bge_en_v1_5": baai_bge_en_v1_5,
}

_IMAGE_EMBEDDING_MODEL_VARIANTS_REGISTRY: Dict[str, callable] = {
    # Image Embedding Models
    "dinov3": ['dinov3-vit7b16-pretrain-lvd1689m', 'dinov3-vith16plus-pretrain-lvd1689m', 'dinov3-vitl16-pretrain-lvd1689m', 'dinov3-vitb16-pretrain-lvd1689m', 'dinov3-vits16-pretrain-lvd1689m'],
    "dinov2": ['dinov2-giant', 'dinov2-base', 'dinov2-small', 'dinov2-large'],
}

_TEXT_EMBEDDING_MODEL_VARIANTS_REGISTRY: Dict[str, callable] = {
    # Text Embedding Models
    "sentence_t5": ['sentence-t5-base', 'sentence-t5-large', 'sentence-t5-xl', 'sentence-t5-xxl'],
    "gtr_t5": ['gtr-t5-base', 'gtr-t5-large', 'gtr-t5-xl', 'gtr-t5-xxl'],
    "all_mpnet_base_v2": ['all-mpnet-base-v2'],
    "alibaba_gte_en_v1_5": ['gte-base-en-v1.5', 'gte-large-en-v1.5'],
    "baai_bge_en_v1_5": ['bge-base-en-v1.5', 'bge-large-en-v1.5'],
}

def get_image_embedding_model(name: str):
    """
    Retrieve an embedding model by name.
    
    Args:
        name: The name of the embedding model to retrieve.
        
    Returns:
        The embedding model class.
        
    Raises:
        ValueError: If the embedding model name is not found in the registry.
    """
    embedding_model_class = _IMAGE_EMBEDDING_MODEL_REGISTRY.get(name.lower())
    if embedding_model_class is None:
        available = list(_IMAGE_EMBEDDING_MODEL_REGISTRY.keys())
        raise ValueError(f"Image Embedding Model '{name}' not found. Available models: {available}")
    return embedding_model_class

def get_text_embedding_model(name: str):
    """
    Retrieve an embedding model by name.
    
    Args:
        name: The name of the embedding model to retrieve.                
    Returns:
        The embedding model class.
        
    Raises:
        ValueError: If the embedding model name is not found in the registry.
    """
    embedding_model_class = _TEXT_EMBEDDING_MODEL_REGISTRY.get(name.lower())
    if embedding_model_class is None:
        available = list(_TEXT_EMBEDDING_MODEL_REGISTRY.keys())
        raise ValueError(f"Text Embedding Model '{name}' not found. Available models: {available}")
    return embedding_model_class

def get_image_embedding_model_variant(name: str):
    """
    Retrieve an embedding model variant by name.
    
    Args:
        name: The name of the embedding model variant to retrieve.
        
    Returns:
        The embedding model variant class.
        
    Raises:
        ValueError: If the embedding model variant name is not found in the registry.
    """
    embedding_model_variant = _IMAGE_EMBEDDING_MODEL_VARIANTS_REGISTRY.get(name.lower())
    if embedding_model_variant is None:
        available = list(_IMAGE_EMBEDDING_MODEL_VARIANTS_REGISTRY.keys())
        raise ValueError(f"Image Embedding Model Variant '{name}' not found. Available variants: {available}")
    return embedding_model_variant

def get_text_embedding_model_variant(name: str):
    """
    Retrieve an embedding model variant by name.
    
    Args:
        name: The name of the embedding model variant to retrieve.
        
    Returns:
        The embedding model variant class.
        
    Raises:
        ValueError: If the embedding model variant name is not found in the registry.
    """
    embedding_model_variant = _TEXT_EMBEDDING_MODEL_VARIANTS_REGISTRY.get(name.lower())
    if embedding_model_variant is None:
        available = list(_TEXT_EMBEDDING_MODEL_VARIANTS_REGISTRY.keys())
        raise ValueError(f"Text Embedding Model Variant '{name}' not found. Available variants: {available}")
    return embedding_model_variant


def list_image_embedding_models():
    """List all available registered datasets."""
    return list(_IMAGE_EMBEDDING_MODEL_REGISTRY.keys())

def list_text_embedding_models():
    """List all available registered datasets."""
    return list(_TEXT_EMBEDDING_MODEL_REGISTRY.keys())
