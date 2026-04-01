import numpy as np
from sklearn import metrics
from typing import Any, Dict, List
from base.base import AbsTask, AbsModel, AbsMethod

class IsoScoreMeasure(AbsTask):
    """Task for zero-shot classification evaluation."""
    
    def __init__(self, name: str, test_images: np.ndarray, support_embeddings: Dict[str, np.ndarray], ground_truth: np.ndarray):
        super().__init__(name, "IsoScoreMeasure")
        self.test_images = test_images         # Images embeddings or raw images
        self.support_embeddings = support_embeddings       # Support images embeddings
        self.ground_truth = ground_truth   # Ground truth labels

    def run(self, support_embeddings: Dict[str, np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Run classification using the provided alignment method."""
        if support_embeddings is None:
            support_embeddings = self.support_embeddings
            
        # Generic flow for alignment-based classification:
        train_features = support_embeddings["train_image"]
        train_labels = support_embeddings["train_labels"]

        image_isoscore = IsoScore(train_features)
        text_isoscore = IsoScore(labels_emb)

        
    
    
