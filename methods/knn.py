import torch
import numpy as np
from typing import Dict, Any, Optional
from base.base import AbsMethod
# Importing from local module
from tqdm import tqdm
import torch.nn.functional as F

class KNNMethod(AbsMethod):
    """ASIF alignment technique."""
    
    def __init__(self, num_classes: int = 1000, k: int = 20, T: float = 0.07):
        super().__init__("KNN")
        self.num_classes = num_classes
        self.k = k
        self.T = T

    def align(self):
        pass
    
    
    def classify(
        self, 
        data: np.ndarray, 
        labels_emb: np.ndarray,
        support_embeddings: Dict[str, np.ndarray]
        ) -> np.ndarray:

        top1, top5, total = 0.0, 0.0, 0
        train_features = support_embeddings["train_image"]
        train_labels = support_embeddings["train_labels"]
        train_labels = torch.Tensor(train_labels)
        train_features = torch.Tensor(train_features)
        train_features = F.normalize(train_features, p=2, dim=1)

        num_test_images, num_chunks = data.shape[0], 100
        imgs_per_chunk = num_test_images // num_chunks
        retrieval_one_hot = torch.zeros(self.k, self.num_classes).to(train_features.device)
        predictions = []
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images
            features = data[
                idx : min((idx + imgs_per_chunk), num_test_images), :
            ]

            #targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
            batch_size = features.shape[0]
            features = torch.Tensor(features)
            features = F.normalize(features, p=2, dim=1)

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features.T)
            distances, indices = similarity.topk(self.k, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices).to(torch.int32)

            retrieval_one_hot.resize_(batch_size * self.k, self.num_classes).zero_()
            retrieval_one_hot = retrieval_one_hot.to(torch.int32)
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            distances_transform = distances.clone().div_(self.T).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, self.num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            _, pred = probs.sort(1, True)
            predictions.append(pred)
        return np.concatenate(predictions, axis=0)
    

    