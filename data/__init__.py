from typing import Type, Dict
import logging

from data.dataset_base.dataset_base import DatasetBase

# Import dataset classes to register them
# Using absolute imports to ensure reliability
from data.imagenet1k.imagenet1k_zeroshot_classif_dataset import (
    Imagenet1k
)
from data.flickr30k.flickr30k_retrieval_dataset import (
    Flickr30kRetrievalDataset
)
from data.mscoco.mscoco_multilabel_classification_dataset import (
    MScocoMultiLabelClassificationDataset, 
    MScocoRetrievalDataset
)

logger = logging.getLogger(__name__)

# Registry dictionary mapping names to classes
_DATASET_REGISTRY: Dict[str, Type[DatasetBase]] = {
    
    # Embedding Generation Datasets
    "imagenet-1k-classification-embedding": Imagenet1kZeroshotClassificationDataset,
    "flickr30k-retrieval-embedding": Flickr30kRetrievalDataset,
    "mscoco-retrieval-embedding": MScocoRetrievalDataset,
    "mscoco-classification-embedding": MScocoMultiLabelClassificationDataset,
}

def get_dataset_class(name: str) -> Type[DatasetBase]:
    """
    Retrieve a dataset class by name.
    
    Args:
        name: The name of the dataset to retrieve.
        
    Returns:
        The dataset class.
        
    Raises:
        ValueError: If the dataset name is not found in the registry.
    """
    dataset_class = _DATASET_REGISTRY.get(name.lower())
    if dataset_class is None:
        available = list(_DATASET_REGISTRY.keys())
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}")
    return dataset_class

def list_datasets() -> list[str]:
    """List all available registered datasets."""
    return list(_DATASET_REGISTRY.keys())
