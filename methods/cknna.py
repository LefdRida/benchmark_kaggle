import numpy as np
import torch
from base.base import AbsMethod
from .cka_core import linear_local_CKA , LinearCKA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Any, Optional
from .cka_core import linear_local_CKA , LinearCKA, kernel_local_CKA

def stretch_representations(repr):
    var = repr.var(axis=0)
    stretch = torch.diag(1 / torch.sqrt(var))
    repr = torch.matmul(repr, stretch)
    return repr


def select_base_by_clustering(embeddings, n_clusters):

    device = embeddings.device
    X     = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    centers = torch.tensor(
        kmeans.cluster_centers_,
        dtype=torch.float32,
        device=device
    )
    X_gpu = embeddings.to(device)
    
    centers_norm = F.normalize(centers, dim=1)  
    X_norm       = F.normalize(X_gpu,   dim=1)  

    cos_sims = centers_norm @ X_norm.T

    closest_indices = torch.argmax(cos_sims, dim=1).cpu().tolist()
    closest_indices = list(set(closest_indices))
    return closest_indices




    
## CKA with 5 captions optimized
class CKNNAAMethod(AbsMethod):

    def __init__(self, base_samples=320, query_samples=500, base_mode="clustering"): # specifiy 'full" in query_sample for fulltest 
        super().__init__("CKA")
        self.base_samples  = base_samples
        self.query_samples = query_samples
        self.base_mode     = base_mode
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"

    def align(self, image_embeddings, text_embeddings, **kwargs):
        return image_embeddings, text_embeddings

    

    def _vectorized_linear_cknna_graph(
        self,
        base_source: torch.Tensor,    # (B, D_s)
        base_target: torch.Tensor,    # (B, D_t)
        query_source: torch.Tensor,   # (M, D_s)
        query_target: torch.Tensor,   # (N, D_t)
        topk: int = 10,
        batch_size: int = 1,
    ) -> torch.Tensor:                # (N, M)

        B = base_source.shape[0]
        N = query_target.shape[0]
        M = query_source.shape[0]
        n = B + 1

        device = base_source.device

        # ── Precompute base/cross/self kernels (mirrors vectorized_linear_cka_graph) ──
        K_bb = base_source @ base_source.T           # (B, B)
        L_bb = base_target @ base_target.T           # (B, B)
        K_bq = base_source @ query_source.T          # (B, M)
        L_bq = base_target @ query_target.T          # (B, N)
        K_qq = (query_source * query_source).sum(1)  # (M,)
        L_qq = (query_target * query_target).sum(1)  # (N,)

        # ── Helpers ───────────────────────────────────────────────────────────────

        def build_kernel(base_kernel, cross_kernel, self_kernel):
            """Assemble full augmented n×n kernel matrices for a query batch.

            Args:
                base_kernel:  (B, B)   precomputed base-to-base gram matrix
                cross_kernel: (B, bs)  base-to-query cross terms
                self_kernel:  (bs,)    query self dot-products (diagonal)
            Returns: (bs, n, n)
            """
            bs = cross_kernel.shape[1]
            print(n)
            K = torch.empty(bs, n, n, device=device)
            K[:, :B, :B] = base_kernel            # shared B×B block
            K[:, :B,  B] = cross_kernel.T         # (bs, B)
            K[:,  B, :B] = cross_kernel.T         # (bs, B) — symmetric
            K[:,  B,  B] = self_kernel            # (bs,)
            return K

        def knn_mask(K_full):
            """Binary mask keeping only the top-k entries per row.

            Args:
                K_full: (bs, n, n)
            Returns: (bs, n, n)  — 1 at top-k positions along last dim, else 0
            """
            _, idx = torch.topk(K_full, topk, dim=-1)    # (bs, n, topk)
            return torch.zeros_like(K_full).scatter_(-1, idx, 1.0)

        def batch_hsic(K, L):
            """Biased HSIC  tr(K H L H)  for arbitrarily batched symmetric matrices.

            Uses the double-centering identity  tr(KHLH) = (K ⊙ HLH).sum()
            which avoids materialising the n×n H matrix.

            Args:
                K, L: (..., n, n)
            Returns: (...)
            """
            Lc = (L
                - L.mean(dim=-1, keepdim=True)       # subtract row means
                - L.mean(dim=-2, keepdim=True)       # subtract col means
                + L.mean(dim=(-1, -2), keepdim=True) # add back grand mean
                )
            return (K * Lc).sum(dim=(-1, -2))

        # ── Main computation ──────────────────────────────────────────────────────
        graph   = torch.zeros(N, M, device=device)

        # sim_kk depends only on j — compute and cache across outer loop iterations
        sim_kk   = torch.zeros(M, device=device)
        kk_ready = torch.zeros(M, dtype=torch.bool, device=device)

        for i_start in tqdm(range(0, N, batch_size), desc="[CKNNAMethod] CKNNA batches"):
            i_end = min(i_start + batch_size, N)

            # ── Target (L) kernel block for this i-batch ──────────────────────────
            l_bi   = L_bq[:, i_start:i_end]                       # (B, bs_i)
            l_ii   = L_qq[i_start:i_end]                          # (bs_i,)
            L_full = build_kernel(L_bb, l_bi, l_ii)               # (bs_i, n, n)
            mask_L = knn_mask(L_full)                              # (bs_i, n, n)
            ML     = mask_L * L_full                               # (bs_i, n, n)

            # sim_ll(i) = HSIC(mask_L·L, mask_L·L) — used as denominator
            sim_ll = batch_hsic(ML, ML)                            # (bs_i,)

            for j_start in range(0, M, batch_size):
                j_end = min(j_start + batch_size, M)

                # ── Source (K) kernel block for this j-batch ───────────────────────
                k_bj   = K_bq[:, j_start:j_end]                   # (B, bs_j)
                k_jj   = K_qq[j_start:j_end]                      # (bs_j,)
                K_full = build_kernel(K_bb, k_bj, k_jj)           # (bs_j, n, n)
                mask_K = knn_mask(K_full)                          # (bs_j, n, n)
                MK     = mask_K * K_full                           # (bs_j, n, n)

                # sim_kk(j) = HSIC(mask_K·K, mask_K·K) — cached after first i pass
                if not kk_ready[j_start:j_end].all():
                    sim_kk[j_start:j_end]   = batch_hsic(MK, MK)  # (bs_j,)
                    kk_ready[j_start:j_end] = True

                # ── Cross term over all (i, j) pairs ──────────────────────────────
                # Intersection of nearest-neighbour masks from both spaces:
                # mask_L: (bs_i, n, n) → (bs_i,  1,   n, n)
                # mask_K: (bs_j, n, n) → ( 1,   bs_j, n, n)
                M_ij  = mask_L.unsqueeze(1) * mask_K.unsqueeze(0)  # (bs_i, bs_j, n, n)

                K_hat = M_ij * K_full.unsqueeze(0)                  # (bs_i, bs_j, n, n)
                L_hat = M_ij * L_full.unsqueeze(1)                  # (bs_i, bs_j, n, n)

                sim_kl = batch_hsic(K_hat, L_hat)                   # (bs_i, bs_j)

                # ── CKNNA score ────────────────────────────────────────────────────
                # sim_kk: (bs_j,) → (1,    bs_j)
                # sim_ll: (bs_i,) → (bs_i, 1  )
                denom = torch.sqrt(
                    sim_kk[j_start:j_end].unsqueeze(0) * sim_ll.unsqueeze(1)
                ) + 1e-6                                             # (bs_i, bs_j)

                graph[i_start:i_end, j_start:j_end] = sim_kl / denom

        return graph
        
    def classify(self, 
        data: np.ndarray, 
        labels_emb: np.ndarray,
        support_embeddings: Dict[str, np.ndarray]
        ) -> np.ndarray:

        train_source = torch.tensor(
                support_embeddings["train_text"],
                dtype=torch.float32, device=self.device
            )
        train_target = torch.tensor(
            support_embeddings["train_image"],
            dtype=torch.float32, device=self.device
            )
        
        assert train_source.shape[0] == train_target.shape[0], "Number of training samples must match between source and target."
        n_classes = len(labels_emb)
        test_source = torch.tensor(labels_emb, dtype=torch.float32, device=self.device)
        test_target = torch.tensor(data, dtype=torch.float32, device=self.device)

        source_total = torch.cat([train_source, test_source], dim=0)
        target_total = torch.cat([train_target, test_target], dim=0)

        source_total = stretch_representations(source_total)
        target_total = stretch_representations(target_total)

        n_train = train_source.shape[0]
        train_source = source_total[:n_train]
        test_source  = source_total[n_train:]
        train_target = target_total[:n_train]
        test_target  = target_total[n_train:]
       
        train_images_for_clustering = train_target

        if self.base_mode == "full":
            base_target = train_target
            base_source = train_source
        
        elif self.base_mode == "random":
            n_base      = min(self.base_samples, n_train)
            idx         = torch.randperm(n_train, device=self.device)[:n_base]
            base_target = train_target[idx]
            base_source = train_source[idx]
        
        elif self.base_mode == "clustering":
            base_idx    = select_base_by_clustering(
                train_images_for_clustering,        
                n_clusters=min(n_classes, n_train)
            )
            base_target = train_target[base_idx]
            base_source = train_source[base_idx]
            print(f"[CKAMethod] Base mode: clustering ({len(base_idx)} from {n_train})")
        else:
            raise ValueError(f"base_mode must be 'clustering', 'random', or 'full'")
        if self.query_samples == "full":
            n_queries = test_target.shape[0]
        elif isinstance(self.query_samples, int):
            n_queries = self.query_samples
        else:
            raise ValueError(f"query_samples must be an int or 'full'")

        with torch.no_grad():
            # graph = kernel_local_CKA(
            #     base_source=base_source,
            #     base_target=base_target,
            #     query_source=test_source,
            #     query_target=test_target,
            #     device=self.device)
            graph = self._vectorized_linear_cknna_graph(
                base_source=base_source,
                base_target=base_target,
                query_source=test_source,
                query_target=test_target,
            )

        graph = graph.detach().cpu().numpy()
        predictions = []

        for i in tqdm(range(n_queries), desc="[CKAMethod] Evaluating hits"):
            
            row = graph[i]
            ind_row = sorted(list(range(len(row))), key = lambda x: -row[x])
            pred = ind_row[0]
            predictions.append(pred)

        print(f"[CKAMethod] Done.\n")

        return predictions
        

    def retrieve(
        self,
        queries: np.ndarray,
        gt_ids=None,
        documents: np.ndarray = None,
        support_embeddings=None,
        topk: int = 10,
        num_gt: int = 5,
        direction: str = "i2t",
        **kwargs
    ):

        if direction == "i2t":
            train_source = torch.tensor(
                support_embeddings["train_text"],
                dtype=torch.float32, device=self.device
            )
            train_target = torch.tensor(
                support_embeddings["train_image"],
                dtype=torch.float32, device=self.device
            )
        else:
            train_source = torch.tensor(
                support_embeddings["train_image"],
                dtype=torch.float32, device=self.device
            )
            train_target = torch.tensor(
                support_embeddings["train_text"],
                dtype=torch.float32, device=self.device
            )

        
        test_target = torch.tensor(queries,   dtype=torch.float32, device=self.device)
        test_source = torch.tensor(documents, dtype=torch.float32, device=self.device)

        source_total = torch.cat([train_source, test_source], dim=0)
        target_total = torch.cat([train_target, test_target], dim=0)

        source_total = stretch_representations(source_total)
        target_total = stretch_representations(target_total)

        n_train = train_source.shape[0]

        train_source = source_total[:n_train]
        test_source  = source_total[n_train:]
        train_target = target_total[:n_train]
        test_target  = target_total[n_train:]


        
        if direction == "i2t":
            train_images_for_clustering = train_target  
        else:
            train_images_for_clustering = train_source  
        
        n_train_samples = train_images_for_clustering.shape[0]
        
        if self.base_mode == "full":
            base_target = train_target
            base_source = train_source
            print(f"[CKAMethod] Base mode: full train ({n_train_samples} samples)")
        
        elif self.base_mode == "random":
            n_base      = min(self.base_samples, n_train_samples)
            idx         = torch.randperm(n_train_samples, device=self.device)[:n_base]
            base_target = train_target[idx]
            base_source = train_source[idx]
            print(f"[CKAMethod] Base mode: random ({n_base} from {n_train_samples})")
        
        elif self.base_mode == "clustering":
            base_idx    = select_base_by_clustering(
                train_images_for_clustering,        
                n_clusters=min(self.base_samples, n_train_samples)
            )
            base_target = train_target[base_idx]
            base_source = train_source[base_idx]
            print(f"[CKAMethod] Base mode: clustering ({len(base_idx)} from {n_train_samples})")
        else:
            raise ValueError(f"base_mode must be 'clustering', 'random', or 'full'")

 
        n_available = test_target.shape[0]

        if self.query_samples == "full":
            n_queries = n_available
        elif isinstance(self.query_samples, int):
            n_queries = self.query_samples
        else:
            raise ValueError(f"query_samples must be an int or 'full'")

        assert n_queries <= n_available, (
            f"query_samples={n_queries} exceeds available ({n_available})."
        )                                 

        query_doc_indices = []
        for i in range(n_queries):
            for cap_idx in gt_ids[i]:
                query_doc_indices.append(cap_idx)

        restricted_source = test_source[query_doc_indices]
        global_to_local   = {g: l for l, g in enumerate(query_doc_indices)}
        local_gt_ids      = [
            [global_to_local[cap_idx] for cap_idx in gt_ids[i]]
            for i in range(n_queries)
        ]

        n_documents = restricted_source.shape[0]

                                       

        print(f"\n[CKAMethod]")
        print(f"  direction     : {direction}")
        print(f"  device        : {self.device}")
        print(f"  base mode     : {self.base_mode}")
        print(f"  base size     : {base_target.shape[0]}")
        print(f"  query size    : {n_queries} / {n_available} available")
        print(f"  documents     : {n_documents} ({num_gt} x {n_queries})")
        print(f"  graph shape   : ({n_queries} x {n_documents})")
        print(f"  source dim    : {test_source.shape[1]}")
        print(f"  target dim    : {test_target.shape[1]}")
        print(f"  topk          : {topk}\n")

 

        print(f"[CKAMethod] Computing CKA graph ({n_queries} x {n_documents})...")

        with torch.no_grad():
            graph = self._vectorized_linear_cka_graph(
                base_source=base_source,
                base_target=base_target,
                query_source=restricted_source,
                query_target=test_target[:n_queries],
            )

        graph = graph.detach().cpu().numpy()


        all_hits = []

        for i in tqdm(range(n_queries), desc="[CKAMethod] Evaluating hits"):
            row        = graph[i]
            sorted_idx = np.argsort(-row)[:topk]
            hit        = np.zeros(topk)
            for k, idx in enumerate(sorted_idx):
                if idx in local_gt_ids[i]:
                    hit[k] = 1
            all_hits.append(hit)

        print(f"[CKAMethod] Done.\n")

        return all_hits

