import numpy as np
from typing import Any, Dict, List
from base.base import AbsTask, AbsModel, AbsMethod
import torch 
from tqdm import tqdm


import numpy as np
from typing import Any, Dict
from base.base import AbsTask, AbsMethod
import torch
from tqdm import tqdm

class RetrievalTask(AbsTask):
    """Task for retrieval evaluation supporting i2t and t2i directions."""

    def __init__(
        self,
        name: str,
        queries: np.ndarray,
        documents: np.ndarray,
        gt_ids: np.ndarray,
        support_embeddings: Dict[str, np.ndarray] = None,
        topk: int = 20,
        num_gt: int = 10,
        direction: str = "i2t",       # "i2t" or "t2i"
        metric_mode: str = "completeness",  # "completeness" (num_gt=5) or "binary" (num_gt=1)
        n_clusters: int = 20,
        copying_exp = False,
        n_repeats = 5,
        translate=False,
        translation_std=0.01,
        translation_mean=0.0,
        experiment_name="cka_retrieval",
    ):
        super().__init__(name, "retrieval")
        self.queries        = np.array(queries)
        self.documents      = np.array(documents)
        self.gt_ids         = gt_ids
        self.support_embeddings = support_embeddings
        self.topk           = topk
        self.num_gt         = num_gt
        self.direction      = direction
        self.metric_mode    = metric_mode
        self.n_clusters     = n_clusters
        self.copying_exp    = copying_exp
        self.n_repeats      = n_repeats
        self.translate      = translate
        self.translation_std = translation_std
        self.translation_mean = translation_mean
        self.experiment_name = experiment_name

    def run(
        self,
        method: AbsMethod,
        support_embeddings: Dict[str, np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:

        if support_embeddings is None:
            support_embeddings = self.support_embeddings

        # ============================================================
        # Set up queries / documents / gt based on direction
        # ============================================================

        if self.direction == "i2t":
            queries_in   = self.queries
            documents_in = self.documents
            gt_ids_in    = self.gt_ids
            num_gt       = self.num_gt        # 5 (or whatever was passed)

        elif self.direction == "t2i":
            n_images     = len(self.queries)
            queries_in   = np.array([
                self.documents[self.gt_ids[i][0]] for i in range(n_images)
            ])
            documents_in = self.queries
            gt_ids_in    = [[i] for i in range(n_images)]
            num_gt       = 1

        else:
            raise ValueError(f"direction must be 'i2t' or 't2i', got: {self.direction}")

        # ============================================================
        # Retrieval phase
        # ============================================================

        if hasattr(method, 'retrieve'):
            all_hits, diagnostic_results = method.retrieve(
                queries_in,
                gt_ids_in,
                documents_in,
                support_embeddings,
                self.topk,
                num_gt,
                direction=self.direction,
                n_clusters=self.n_clusters,
                copying_exp=self.copying_exp,
                n_repeats=self.n_repeats,
                translate=self.translate,
                translation_std=self.translation_std,
                translation_mean=self.translation_mean,
                experiment_name=self.experiment_name
            )

        else:
            if self.direction == "i2t":
                aligned_queries, aligned_documents = method.align(
                    image_embeddings=queries_in,
                    text_embeddings=documents_in,
                    support_embeddings=support_embeddings
                )
            else:
                aligned_documents, aligned_queries = method.align(
                    image_embeddings=documents_in,
                    text_embeddings=queries_in,
                    support_embeddings=support_embeddings
                )

            if hasattr(method, 'similarity_function'):
                similarity_function = method.get_similarity_function()
            else:
                def similarity_function(x, y):
                    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
                    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-10)
                    return np.sum(x * y, axis=1)

            all_hits = []

            for idx in tqdm(range(aligned_queries.shape[0])):
                gt_query_ids = gt_ids_in[idx]

                q_emb      = aligned_queries[idx, :].reshape(1, -1)
                sim_scores = similarity_function(q_emb, aligned_documents)

                if isinstance(sim_scores, np.ndarray):
                    sim_scores = torch.from_numpy(sim_scores)

                sim_top_idx = torch.topk(
                    sim_scores,
                    self.topk,
                    largest=True,
                    sorted=True
                ).indices.cpu().numpy()

                hit = np.zeros(self.topk)
                for jj, top_idx in enumerate(sim_top_idx.reshape(-1)):
                    
                    hit[jj] = 1 if top_idx in gt_query_ids else 0

                all_hits.append(hit)

        # ============================================================
        # Metrics
        # ============================================================

        # t2i is always binary; i2t respects metric_mode
        use_binary = (self.direction == "t2i") or (self.metric_mode == "binary")

        if use_binary:
            # R@1, R@5, R@10, MRR  — binary single-hit success
            all_r1  = []
            all_r5  = []
            all_r10 = []
            all_mrr = []

            for hit in all_hits:
                hit = np.array(hit)

                all_r1.append(float(hit[:1].any()))
                all_r5.append(float(hit[:5].any()))
                all_r10.append(float(hit[:10].any()))

                relevant_positions = np.where(hit == 1)[0]
                mrr = 1.0 / (relevant_positions[0] + 1) if len(relevant_positions) > 0 else 0.0
                all_mrr.append(mrr)

            return {
                "R@1":  float(np.mean(all_r1)),
                "R@5":  float(np.mean(all_r5)),
                "R@10": float(np.mean(all_r10)),
                "MRR":  float(np.mean(all_mrr)),
            }

        else:
            # Completeness metrics — Recall@k, MAP, MRR, NDCG  (num_gt=5)
            all_recalls = []
            all_ap      = []
            all_mrr     = []
            all_ndcg    = []
            
            for hit in all_hits:
                hit = np.array(hit)

                recall_k = np.cumsum(hit) / num_gt
                all_recalls.append(recall_k)
                precision_k = np.cumsum(hit) / (np.arange(len(hit)) + 1)
                ap = np.sum(precision_k * hit) / num_gt
                all_ap.append(ap)

                relevant_positions = np.where(hit == 1)[0]
                mrr = 1.0 / (relevant_positions[0] + 1) if len(relevant_positions) > 0 else 0.0
                all_mrr.append(mrr)

                ranks      = np.arange(1, len(hit) + 1)
                dcg        = np.sum(hit / np.log2(ranks + 1))
                ideal_hits = np.ones(min(num_gt, len(hit)))
                idcg       = np.sum(ideal_hits / np.log2(np.arange(1, len(ideal_hits) + 1) + 1))
                ndcg       = dcg / idcg if idcg > 0 else 0.0
                all_ndcg.append(ndcg)

            avg_recall = np.mean(all_recalls, axis=0)

            return {
                "Recall@5":  None,#avg_recall[4] if self.topk >= 5  else None,
                "Recall@10": None,#avg_recall[9] if self.topk >= 10 else None,
                "MAP":       None,#float(np.mean(all_ap)),
                "MRR":       None,#float(np.mean(all_mrr)),
                "NDCG":      None#float(np.mean(all_ndcg)),
            }, diagnostic_results