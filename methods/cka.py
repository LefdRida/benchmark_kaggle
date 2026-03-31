import numpy as np
import torch
from base.base import AbsMethod
from .cka_core import linear_local_CKA
from tqdm import tqdm

import numpy as np
import torch
from base.base import AbsMethod
from .cka_core import linear_local_CKA



### working base from train and query set from test for one to one retrieval 
class CKAMethod(AbsMethod):

    def __init__(self, base_samples=320, query_samples=500):
        super().__init__("CKA")
        self.base_samples = base_samples
        self.query_samples = query_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def align(self, image_embeddings, text_embeddings, **kwargs):
        return image_embeddings, text_embeddings

    def retrieve(
        self,
        queries: np.ndarray,        # test images (N_images, D)
        gt_ids=None,                # image_id -> list of caption indices
        documents: np.ndarray=None, # test captions (5N, D)
        support_embeddings=None,
        topk: int = 5,
        num_gt: int = 1,
        **kwargs
    ):

        # --------------------------------------------------
        # TRAIN SPLIT → BASE
        # --------------------------------------------------

        train_images = support_embeddings["train_image"]
        train_texts  = support_embeddings["train_text"]

        rng = np.random.default_rng(0)
        base_idx = rng.choice(
            len(train_images),
            size=min(self.base_samples, len(train_images)),
            replace=False
        )

        base_images = torch.tensor(train_images[base_idx], dtype=torch.float32).to(self.device)
        base_texts  = torch.tensor(train_texts[base_idx], dtype=torch.float32).to(self.device)

        # --------------------------------------------------
        # TEST SPLIT → QUERY (1-to-1 construction)
        # --------------------------------------------------

        images = torch.tensor(queries, dtype=torch.float32)
        captions = torch.tensor(documents, dtype=torch.float32)

        assert len(images) >= self.query_samples

        aligned_images = []
        aligned_captions = []

        for i in range(self.query_samples):
            aligned_images.append(images[i])

            # pick first valid caption for that image
            caption_idx = gt_ids[i][0]
            aligned_captions.append(captions[caption_idx])

        query_images = torch.stack(aligned_images).to(self.device)
        query_texts  = torch.stack(aligned_captions).to(self.device)

        # --------------------------------------------------
        # Square CKA Graph
        # --------------------------------------------------

        graph = linear_local_CKA(
            source_base=base_texts,
            target_base=base_images,
            source_query=query_texts,
            target_query=query_images,
            device=self.device
        )

        graph = graph.detach().cpu().numpy()

        # --------------------------------------------------
        # Diagonal Retrieval Evaluation
        # --------------------------------------------------

        results = []

        for i in range(self.query_samples):

            row = graph[i]
            sorted_idx = np.argsort(-row)[:topk]

            hit = np.zeros(topk)

            correct_index = i

            for k, idx in enumerate(sorted_idx):
                if idx == correct_index:
                    hit[k] = 1

            results.append(hit)

        return results