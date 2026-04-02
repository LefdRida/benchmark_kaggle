from data.dataset_base import EmbeddingDataset, DatasetBase
import numpy as np
from typing import List, Dict, Tuple
from omegaconf import DictConfig
import polars as pl
import re
from sklearn.model_selection import StratifiedShuffleSplit


class Places365(DatasetBase):
    def __init__(self, root: str, filelist_places: str,categories_places: str, **kwargs):
        DatasetBase.__init__(self)
        self.load_data(root, filelist_places,categories_places)

    def load_data(
        self, root: str, filelist_places: str,categories_places: str
    ) -> None:
        """
        Load the Places365 dataset from Hugging Face Arrow format and
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
        mapping = {}
        with open(categories_places, "r") as f:
            for _, l in enumerate(f.readlines()):
                s = l.strip().split(" ")
                class_id = s[-1]
                class_name = s[0].split("/")[-1]
                class_name = re.sub(r'[^a-zA-Z0-9\s]', ' ', class_name)
                mapping[int(class_id)] = (class_name, int(class_id))
                
        table = pl.read_csv(filelist_places, separator = " ", has_header=False)
        table.columns = ["image_fname", "label_id"]
        table = table.with_columns(
            pl.col('label_id')
            .map_elements(lambda x: mapping[x][0], return_dtype=pl.String)
            .alias('label')
            )
        table = table.with_columns(
            pl.col('image_fname')
            .map_elements(lambda x: f"{root}/{x}", return_dtype=pl.String)
            .alias('image_path')
            )
        
        self.table = table.sort("label_id")

        self.clsidx_to_labels = {}
        for sample in self.table.select(['label_id', 'label']).unique(maintain_order=True).to_dicts():
            if sample["label_id"] not in self.clsidx_to_labels:
                self.clsidx_to_labels[sample["label_id"]] = sample["label"]
                
        print(len(self.clsidx_to_labels))

        

class Places365ZeroshotClassificationDataset(Places365, EmbeddingDataset):
    def __init__(self, task_config: DictConfig):
        Places365.__init__(
            self, 
            root=task_config.root, 
            filelist_places=task_config.filelist_places, 
            categories_places=task_config.categories_places
            )
        
        self.image_paths = self.table.select("image_path").to_series().to_list()
        if task_config.generate_embedding:
            self.labels_descriptions = ["This is an image of " + self.clsidx_to_labels[x] for x in self.clsidx_to_labels.keys()]
        else:
            EmbeddingDataset.__init__(
                self,
                split=task_config.split
            )
            self.labels = self.table.select("label_id").to_numpy()
            #print(self.labels.shape)
            #print(self.image_embeddings.shape[0])
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

        elif self.split == "large" and self.train_idx is not None:
            train_image_embeddings =  self.image_embeddings[self.train_idx]
            train_labels = self.labels[self.train_idx]
            sss = StratifiedShuffleSplit(n_splits=1, train_size=400000, random_state=42)
            for i, (train_index, _) in enumerate(sss.split(train_image_embeddings, train_labels)):
                self.train_idx = train_index
                
            train_image_embeddings =  self.image_embeddings[self.train_idx]
            train_labels = self.labels[self.train_idx]
            for idx in self.train_idx:
                label = self.labels[idx]
                label_emb = self.labels_emb[label].reshape(-1)
                text_emb.append(label_emb)
            train_text_embeddings = np.array(text_emb)
             
            
        else:
            raise ValueError("Please set split to 'train' or get the train/test split index first.")
        assert train_image_embeddings.shape[0] == train_text_embeddings.shape[0]
        print(train_image_embeddings.shape)
        print(train_text_embeddings.shape)
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