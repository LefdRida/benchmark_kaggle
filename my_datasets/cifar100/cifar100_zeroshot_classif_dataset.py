from data.dataset_base import EmbeddingDataset, DatasetBase
import numpy as np
from typing import List, Dict, Tuple
from omegaconf import DictConfig
import polars as pl

class Cifar100(DatasetBase):
    def __init__(self,  csv_data_path: str, **kwargs):
        DatasetBase.__init__(self)
        self.load_data(csv_data_path)

    def load_data(
        self, csv_data_path: str
    ) -> None:
        """
        Load the ImageNet dataset from Hugging Face Arrow format and
        corresponding precomputed embeddings.

        Args:
            cfg_dataset: configuration file (expects cfg_dataset.paths.dataset_path)
        
        Returns:
            img_path: dummy list of image identifiers (or placeholder paths)
            mturks_idx: array of image labels (same as orig_idx, kept for compatibility)
            orig_idx: ground truth class indices (int)
            clsidx_to_labels: a dict of class idx to str.
        """
        """
        Args:
            cfg_dataset: configuration file

        Returns:
            img_paths: list of image absolute paths
            text_descriptions: list of text descriptions
        """
        #'/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv'
        #"/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt"
       
                
        table = pl.read_csv(csv_data_path)
        self.table = table.sort(["label_id", "image_id"])

        

class Cifar100ZeroshotClassificationDataset(Cifar100, EmbeddingDataset):
    def __init__(self, task_config: DictConfig):
        Cifar100.__init__(
            self, 
            csv_data_path=task_config.csv_data_path
            )
        
        self.image_paths = self.table.select("image_path").to_series().to_list()
        if task_config.generate_embedding:
            self.labels_descriptions = self.table.select("caption")
            self.labels_descriptions = self.labels_descriptions.unique(maintain_order=True).to_numpy().flatten().tolist()
        else:
            EmbeddingDataset.__init__(
                self,
                split=task_config.split
            )
            self.labels = self.table.select("label_id").to_numpy().flatten()
            self.metatask = task_config.metatask
            self.load_two_encoder_data(
                hf_repo_id=task_config.hf_repo_id, 
                hf_img_embedding_name=task_config.hf_img_embedding_name, 
                hf_text_embedding_name=task_config.hf_text_embedding_name
            )
            self.set_labels_emb()
            if task_config.split == "train" or task_config.split == "large":
                self.set_train_test_split_index(
                    train_test_ratio=task_config.train_test_ratio, 
                    seed=task_config.seed
                )
                self.set_training_paired_embeddings()
            
    def set_training_paired_embeddings(self) -> None:        

        """Get the paired embeddings for both modalities."""
        #if self.text_embeddings.shape[0] != len(self.clsidx_to_labels):
        #    self.text_embeddings = np.unique(self.text_embeddings, axis=0)
        #    print(self.text_embeddings.shape)
        #, "To pair embeddings, the text embeddings should contain only all possible labels."
        assert self.image_embeddings.shape[0] == len(self.labels), "Each image should have a corresponding list of labels."
        assert self.train_idx is not None or self.split=="train", "Please get the train/test split index first."
        assert self.support_embeddings.get('labels_emb', None) is not None, "Please get the labels embeddings first."
        
        text_emb = []
        self.labels_emb = self.support_embeddings['labels_emb']
        if self.split == "train":
            for idx, label in enumerate(self.labels):
                label_emb = self.labels_emb[label].reshape(-1)
                text_emb.append(label_emb)
            train_text_embeddings = np.array(text_emb)
            train_labels = self.labels
        elif self.split == "large" and self.train_idx is not None:
            train_image_embeddings =  self.image_embeddings[self.train_idx] #[]#
            #seen_labels = []
            for idx in self.train_idx:
                label = self.labels[idx]
                #if label not in seen_labels:
                #ims_embedding = self.image_embeddings[idx].reshape(-1)
                label_emb = self.labels_emb[label].reshape(-1)
                #train_image_embeddings.append(ims_embedding)
                text_emb.append(label_emb)
                #seen_labels.append(label)
                    
            train_text_embeddings = np.array(text_emb)
            train_image_embeddings = np.array(train_image_embeddings)
            train_labels = self.labels[self.train_idx]  #seen_labels #
            
        else:
            raise ValueError("Please set split to 'train' or get the train/test split index first.")
        assert train_image_embeddings.shape[0] == train_text_embeddings.shape[0]

        self.support_embeddings["train_image"] = train_image_embeddings
        self.support_embeddings["train_text"] = train_text_embeddings
        self.support_embeddings["train_labels"] = train_labels
    
    def set_labels_emb(self) -> None:
        """Get the text embeddings for all possible labels."""
        if self.text_embeddings.shape[0] == len(self.clsidx_to_labels):
            labels_emb = self.text_embeddings
        else:
            label_emb = []
            for label_idx in self.clsidx_to_labels:
                # find where the label is in the train_idx
                label_idx_in_ds = np.where(self.labels == label_idx)[0]
                label_emb.append(self.text_embeddings[label_idx_in_ds[0]])
            labels_emb = np.array(label_emb)
            assert labels_emb.shape[0] == len(self.clsidx_to_labels)
        self.labels_emb = labels_emb
        self.support_embeddings["labels_emb"] = labels_emb
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.image_embeddings is not None and self.text_embeddings is not None, "Please load the data first."
        if self.split == "large":
            assert self.train_idx is not None and self.val_idx is not None, "Please get the train/test split index first."
            val_image_embeddings = self.image_embeddings[self.val_idx]
            val_labels = self.labels[self.val_idx] 
        elif self.split == "val":
            val_image_embeddings = self.image_embeddings
            val_labels = self.labels
        else:   
            raise ValueError("Please set split to 'train', 'val' or 'large'.")
        return val_image_embeddings, val_labels
