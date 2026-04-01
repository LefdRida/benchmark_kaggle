import numpy as np
from sklearn import metrics
from typing import Any, Dict, List
from base.base import AbsTask, AbsModel, AbsMethod
import torch
from IsoScore.IsoScore import *

class EmbeddingSpaceAnalysisTask(AbsTask):
    """Task for zero-shot classification evaluation."""
    
    def __init__(self, name: str, test_images: np.ndarray, support_embeddings: Dict[str, np.ndarray], ground_truth: np.ndarray):
        super().__init__(name, "EmbeddingSpaceAnalysis")
        self.test_images = test_images         # Images embeddings or raw images
        self.support_embeddings = support_embeddings       # Support images embeddings
        self.ground_truth = ground_truth   # Ground truth labels

    def run(self, method: AbsMethod, support_embeddings: Dict[str, np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Run classification using the provided alignment method."""
        if support_embeddings is None:
            support_embeddings = self.support_embeddings
            
        train_image = support_embeddings["train_image"]
        train_text = support_embeddings["train_text"]

        image_isoscore = IsoScore(torch.Tensor(train_image).float())
        text_isoscore = IsoScore(torch.Tensor(train_text).float())

        geo_preserve_metric = (1/train_image.shape[0]**2) * np.linalg.norm(train_image@train_image.T  - train_text@train_text.T)**2

        return {"geo_preserve_metric": geo_preserve_metric, "image_isoscore": image_isoscore, "text_isoscore": text_isoscore}



    
    
