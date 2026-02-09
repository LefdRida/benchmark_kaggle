from typing import Union, List, Optional
from base import AbsTask
from metatasks.classification import ClassificationTask
from metatasks.retrieval import RetrievalTask
from omegaconf import DictConfig

# Use registry for plug-and-play dataset loading
from data import get_dataset_class, list_datasets


def _load_embeddings_and_split(dataset, split: str):
    """Helper function to load embeddings and optionally create train/test split.
    
    Args:
        dataset: Dataset instance
        split: Dataset split type ('large', 'train', or 'val')
        
    Returns:
        support_embeddings dict or None
    """
    dataset.load_two_encoder_data()
    if dataset.metatask == "classification":
        dataset.get_labels_emb()

    if split == 'large' or split == 'train':
        dataset.get_train_test_split_index()
        dataset.get_training_paired_embeddings()

    support_embeddings = dataset.get_support_embeddings()
    return support_embeddings


def _create_classification_task(
    dataset_name: str,
    dataset,
    support_embeddings: Optional[dict]
) -> ClassificationTask:
    """Helper function to create a classification task.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset instance with classification data
        support_embeddings: Optional training embeddings
        
    Returns:
        ClassificationTask instance
    """


    test_img, test_labels = dataset.get_test_data()
    
    return ClassificationTask(
        name=f"{dataset_name}-Classification",
        test_images=test_img,
        train_image = support_embeddings.get("train_image", None),
        train_text = support_embeddings.get("train_text", None),
        labels_emb=support_embeddings.get("labels_emb", None),
        ground_truth=test_labels,
    )


def _create_retrieval_task(
    dataset_name: str,
    dataset,
    config: DictConfig,
    support_embeddings: Optional[dict]
) -> RetrievalTask:
    """Helper function to create a retrieval task.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset instance with retrieval data
        config: Configuration object
        support_embeddings: Optional training embeddings
        
    Returns:
        RetrievalTask instance
    """
    val_img, val_txt, gt_caption_ids, gt_img_ids = dataset.get_test_data()
    
    return RetrievalTask(
        name=f"{dataset_name}-Retrieval",
        queries=val_img,
        documents=val_txt,
        gt_ids=gt_img_ids,
        support_embeddings=support_embeddings,
        topk=config.retrieval.topk,
        num_gt=config.retrieval.num_gt
    )


def load_dataset_task(dataset_name: str, config: DictConfig) -> AbsTask:
    """Load a dataset and wrap it into a Task.
    
    Args:
        dataset_name: Name of the dataset (imagenet-1k, flickr30k, mscoco-*)
        config: Configuration dictionary/object for the dataset
        
    Returns:
        Task instance (ClassificationTask or RetrievalTask)
        
    Raises:
        ValueError: if dataset name is not supported
    """
    dataset_name_lower = dataset_name.lower()
    try:
        DatasetClass = get_dataset_class(f"{dataset_name_lower}-{config.metatask}")
    except ValueError:
        print(f"Dataset {dataset_name} not found. Available: {list(dataset_mapping.values())}")
        exit(1)
    
    ds = DatasetClass(config)
    support_embeddings = _load_embeddings_and_split(ds, ds.split)
    # ImageNet-1K Classification
    if config.metatask == "classification":
        return _create_classification_task(f"{dataset_name_lower}-{config.metatask}", ds, support_embeddings)
    # Flickr30k Retrieval
    elif config.metatask == "retrieval":
        return _create_retrieval_task(f"{dataset_name_lower}-{config.metatask}", ds, config, support_embeddings)
    else:
        raise ValueError(
            f"Task '{config.metatask}' not supported. "
            f"Available: classification, retrieval"
        )

