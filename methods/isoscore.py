import torch
from IsoScore.IsoScore import *


class IsoScoreMethod(AbsMethod):
    """ASIF alignment technique."""
    
    def __init__(self):
        super().__init__("IsoScore")


    def align(self):
        pass
    
    
    def classify(
        self, 
        data: np.ndarray, 
        labels_emb: np.ndarray,
        support_embeddings: Dict[str, np.ndarray]
        ) -> np.ndarray:

        train_features = support_embeddings["train_image"]
        train_labels = support_embeddings["train_labels"]

        image_isoscore = IsoScore(train_features)
        text_isoscore = IsoScore(labels_emb)

        
        return image_isoscore, text_isoscore
    

    