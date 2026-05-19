from data.dataset_base import EmbeddingDataset, DatasetBase
import numpy as np
from typing import List, Dict, Tuple
from omegaconf import DictConfig
from pathlib import Path
import polars as pl
import json
from datasets import Dataset
import polars as pl
from datasets import load_dataset
from PIL import Image
import io


def closest_to_centroid(embeddings):
    """Return the embedding closest to the centroid of the set."""
    embeddings = np.array(embeddings)
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return embeddings[np.argmin(distances)]

class NoCaps(DatasetBase):
    def __init__(self, dataset_path: str):
        super().__init__()
        self.captions = None
        self.num_caption_per_image = None
        self.num_image_per_caption = None
        self.gt_caption_doc_ids = None
        self.gt_img_doc_ids = None
        self.load_data(dataset_path)
        
            
    def load_data(
    self, dataset_path:str
    ) -> tuple[list[str], list[str], np.ndarray, list[str]]:
        """Load the NoCaps dataset.

        Args:
            dataset_path: configuration file

        Returns:
            data table
        """
        ds = load_dataset("lmms-lab/NoCaps", split="validation", cache_dir="/home/rida.lefdali/work/dataset/nocaps")
        df = pl.from_arrow(ds.data.table)

        # with open(Path(dataset_path) / "nocap_val_4500_captions.json", "r") as f:
        #     data = json.load(f)

        
        
        # images = data["images"]
        # images_table = pl.DataFrame(images)
        # images_table = images_table.rename({"id": "image_id"})
        
        # annotations = data["annotations"]
        # annotations_table = pl.DataFrame(annotations)
        # annotations_table = annotations_table.rename({"id": "caption_id"})

        #self.nocaps = images_table.join(annotations_table, left_on="image_id", right_on="image_id", how="inner")
        self.nocaps = df.sort("image_id", descending=False)




class NoCapsRetrievalDataset(NoCaps, EmbeddingDataset):
    def __init__(self, task_config: DictConfig):
        NoCaps.__init__(self, dataset_path=task_config.dataset_path)
        
        self.image_paths = self.nocaps["image"].to_list() #self.nocaps.select("coco_url").to_series().to_list()
        if task_config.generate_embedding:
            self.nocaps = self.nocaps.explode("annotations_captions").sort("annotations_ids", descending=False)
            self.captions = self.nocaps.select("annotations_captions").to_series().to_list()
        else:
            EmbeddingDataset.__init__(
                self,
                split=task_config.split
            )
            self.captions = self.nocaps.select("annotations_captions").to_series().to_list()
            self.num_caption_per_image = 10
            self.num_image_per_caption = 1
            self.img_caption_mapping = self.nocaps.select("image_id", "annotations_ids").to_dicts()
            self.load_two_encoder_data(
                hf_repo_id=task_config.hf_repo_id, 
                hf_img_embedding_name=task_config.hf_img_embedding_name, 
                hf_text_embedding_name=task_config.hf_text_embedding_name
            )
            if self.split == "train" or self.split == "large":
                self.set_train_test_split_index(
                    train_test_ratio=task_config.train_test_ratio, 
                    seed=task_config.seed
                )
                self.get_training_paired_embeddings()
    
    def get_training_paired_embeddings(self) -> None:
        """Get the paired embeddings for both modalities."""
        print(self.text_embeddings.shape, self.image_embeddings.shape)
        print(len(self.captions), self.num_caption_per_image, len(self.captions)*self.num_caption_per_image)
        assert self.text_embeddings.shape[0] == len(self.captions)*10, \
            "To pair embeddings, the text embeddings should contain only all possible labels."
        assert self.image_embeddings.shape[0] == len(self.captions), \
            "Each image should have a corresponding list of labels."
        assert self.train_idx is not None or self.split=="train", \
            "Please get the train/test split index first."
        representatif_caption = True
        text_emb = []
        image_emb = []
        if self.split == "train":
            for item in self.img_caption_mapping:
                image_ids = item["image_id"]
                caption_ids = item["annotations_ids"]
                caption_emb = self.text_embeddings[caption_ids].reshape(len(caption_ids), -1)
                text_emb.append(caption_emb)
                image_emb.append(
                    np.repeat(self.image_embeddings[image_ids].reshape(1, -1), len(caption_ids), axis=0 )
                    )
            train_image_embeddings = np.concatenate(image_emb, axis=0)
            train_text_embeddings = np.concatenate(text_emb, axis=0)
            
        elif self.split == "large" and self.train_idx is not None:
            for idx in self.train_idx:
                image_ids = self.img_caption_mapping[idx]["image_id"]
                caption_ids = self.img_caption_mapping[idx]["annotations_ids"]
                caption_emb = self.text_embeddings[caption_ids].reshape(len(caption_ids), -1)
                if representatif_caption:
                    closest_to_centroid_emb = closest_to_centroid(caption_emb)
                    text_emb.append(closest_to_centroid_emb.reshape(1, -1))
                    image_emb.append(self.image_embeddings[image_ids].reshape(1, -1))
                else:
                    text_emb.append(caption_emb)
                    image_emb.append(
                        np.repeat(self.image_embeddings[image_ids].reshape(1, -1), len(caption_ids), axis=0)
                        )
            train_image_embeddings = np.concatenate(image_emb, axis=0)
            train_text_embeddings = np.concatenate(text_emb, axis=0)
        else:
            raise ValueError("Please set split to 'train' or get the train/test split index first.")
        
        assert train_image_embeddings.shape[0] == train_text_embeddings.shape[0]
        self.support_embeddings["train_image"] = train_image_embeddings
        self.support_embeddings["train_text"] = train_text_embeddings

    def get_test_data(self):
        assert self.image_embeddings is not None and self.text_embeddings is not None, \
            "Please load the data first."
        if self.split == "large":
            assert self.train_idx is not None and self.val_idx is not None, \
                "Please get the train/test split index first."
            self.text_to_image_gt_ids = {}
            self.image_to_text_gt_ids = {}
            val_text_idx = []
            for idx, image_id in enumerate(self.val_idx):
                caption_ids = self.img_caption_mapping[image_id]["annotations_ids"]
                if len(caption_ids) != self.num_caption_per_image:
                    continue
                val_text_idx.extend(caption_ids)
                val_caption_ids = list(
                    range(idx*self.num_caption_per_image, (idx+1)*self.num_caption_per_image)
                    )
                self.image_to_text_gt_ids[idx] = val_caption_ids
                for val_caption_id in val_caption_ids:
                    self.text_to_image_gt_ids[val_caption_id] = [idx]
            val_image_embeddings = self.image_embeddings[self.val_idx]
            val_text_embeddings = self.text_embeddings[val_text_idx]

        elif self.split == "val":
            val_image_embeddings = self.image_embeddings
            val_text_embeddings = self.text_embeddings
            for item in enumerate(self.img_caption_mapping):
                image_ids = item["image_id"]
                caption_ids = item["annotations_ids"]
                self.image_to_text_gt_ids[image_ids] = caption_ids
                for caption_id in caption_ids:
                    self.text_to_image_gt_ids[caption_id] = [image_ids] 
        else:   
            raise ValueError("Please set split to 'train', 'val' or 'large'.")

        return val_image_embeddings, val_text_embeddings, self.image_to_text_gt_ids, self.text_to_image_gt_ids

    def get_support_embeddings(self):
        return self.support_embeddings
