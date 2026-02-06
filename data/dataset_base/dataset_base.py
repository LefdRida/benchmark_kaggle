from abc import abstractmethod
import numpy as np
from typing import Tuple, Any, Dict
from data.dataset_base.dataset_utils import load_embeddings_from_hf

class DatasetBase:
    """Base class for all datasets."""
    def __init__(self):
        self.image_paths = None
        self.captions = None
        self.labels_descriptions = None
        self.clsidx_to_labels = {}
        self.labels = None
        self.num_caption_per_image = None
        self.num_image_per_caption = None
        self.gt_caption_doc_ids = None
        self.gt_img_doc_ids = None
        self.data = None

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Load raw dataset (paths, labels, captions)."""
        pass

    def get_test_data(self) -> Any:
        """Return test data. Override in subclass."""
        raise NotImplementedError("Subclass must implement get_test_data()")


class EmbeddingDataset(DatasetBase):
    """Dataset class to hold pre-computed embeddings.
    
    Inherits from DatasetBase to provide a unified interface.
    """
    def __init__(self, 
            img_encoder: str, 
            text_encoder: str, 
            hf_img_embedding_name: str, 
            hf_text_embedding_name: str, 
            hf_repo_id: str, 
            train_test_ratio: float = 0.8, 
            seed: int = 42,
            split: str = "large"
            ):
        super().__init__()

        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.hf_img_embedding_name = hf_img_embedding_name
        self.hf_text_embedding_name = hf_text_embedding_name
        self.hf_repo_id = hf_repo_id
        self.train_test_ratio = train_test_ratio
        self.seed = seed
        self.split = split

        self.support_embeddings = {}
        self.image_embeddings = None
        self.text_embeddings = None
        self.labels = None
        self.train_idx = None
        self.val_idx = None
        
    def load_data(self, *args, **kwargs):
        """EmbeddingDataset loads via load_two_encoder_data instead."""
        pass

    def load_two_encoder_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load embeddings for two modalities from HuggingFace Hub.

        Returns:
            Tuple of (image_embeddings, text_embeddings)
        """
        self.image_embeddings = load_embeddings_from_hf(
            hf_file_name=self.hf_img_embedding_name, 
            repo_id=self.hf_repo_id
        )
        self.text_embeddings = load_embeddings_from_hf(
            hf_file_name=self.hf_text_embedding_name, 
            repo_id=self.hf_repo_id
        )
        return self.image_embeddings, self.text_embeddings

    def get_train_test_split_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the index of the training and validation set.

        Returns:
            Tuple of (train_idx, val_idx)
        """
        assert self.image_embeddings is not None and self.text_embeddings is not None, \
            "Please load the data first."
        assert self.split == "large", "Split must be 'large' to create train/test split."
        
        n = self.image_embeddings.shape[0]
        arange = np.arange(n)
        np.random.seed(self.seed)
        np.random.shuffle(arange)
        self.train_idx = arange[:int(n * self.train_test_ratio)]
        self.val_idx = arange[int(n * self.train_test_ratio):]
        return self.train_idx, self.val_idx


    def get_training_paired_embeddings(self) -> None:
        """Get paired embeddings for training. Override in subclass."""
        raise NotImplementedError("Subclass must implement get_training_paired_embeddings()")
    
    def get_test_data(self) -> None:
        """Get test data. Override in subclass."""
        raise NotImplementedError("Subclass must implement get_test_data()")

    def get_support_data(self):
        assert self.support_embeddings is not None, "Please load the data first."
        return self.support_embeddings