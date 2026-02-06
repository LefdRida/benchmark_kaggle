
import sys
import os
import unittest
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append(os.getcwd())

# Test imports from the consolidated datasets registry
from datasets import get_dataset_class, list_datasets
from data.dataset_base.dataset_base import DatasetBase, EmbeddingDataset

class TestCentralizedRegistry(unittest.TestCase):
    def test_list_datasets(self):
        datasets = list_datasets()
        print(f"Available datasets: {datasets}")
        # Updated keys after consolidation
        self.assertIn("imagenet-1k-classification", datasets)
        self.assertIn("flickr30k-retrieval", datasets)
        self.assertIn("mscoco-classification", datasets)
        self.assertIn("mscoco-retrieval", datasets)

    def test_get_dataset_class(self):
        # ImageNet Classification
        cls = get_dataset_class("imagenet-1k-classification")
        self.assertTrue(issubclass(cls, DatasetBase))
        
        # Flickr30k Retrieval
        cls = get_dataset_class("flickr30k-retrieval")
        # Note: Flickr30kRetrievalDataset may not directly subclass EmbeddingDataset
        self.assertTrue(issubclass(cls, DatasetBase))  
        
        # MSCOCO Classification
        cls = get_dataset_class("mscoco-classification")
        self.assertTrue(issubclass(cls, DatasetBase))

    def test_loader_import(self):
        # Verify loader imports correctly
        try:
            from datasets.loader import load_dataset_task
        except ImportError as e:
            self.fail(f"Failed to import loader: {e}")

if __name__ == '__main__':
    unittest.main()
