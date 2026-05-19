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
from methods.csa_core import NormalizedCCA
from .visualize import run_diagnostics, run_visualization, run_visualization_kernels
def stretch_representations(repr):
    var = repr.var(axis=0)
    stretch = torch.diag(1 / torch.sqrt(var))
    repr = torch.matmul(repr, stretch)
    return repr



def select_base_by_knn(
    query_image:  torch.Tensor,
    train_images: torch.Tensor,
    train_texts:  torch.Tensor,
    k:            int,
) -> tuple:
    if query_image.dim() == 1:
        query_image = query_image.unsqueeze(0)

    q_norm = F.normalize(query_image, dim=-1)
    t_norm = F.normalize(train_images, dim=1)
    sim    = q_norm @ t_norm.T

    knn_idx = torch.topk(sim, k=k, dim=1).indices.squeeze(0)

    return (
        train_images[knn_idx],
        train_texts[knn_idx],
        knn_idx,
    )


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
class CKAMethod(AbsMethod):

    def __init__(self, base_samples=320, query_samples=500, base_mode="clustering"): # specifiy 'full" in query_sample for fulltest 
        super().__init__("CKA")
        self.base_samples  = base_samples
        self.query_samples = query_samples
        self.base_mode     = base_mode
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"

    def align(self, image_embeddings, text_embeddings, **kwargs):
        return image_embeddings, text_embeddings

    def _vectorized_rbf_cka_graph(
        self,
        base_source: torch.Tensor,    # (B, d1)
        base_target: torch.Tensor,    # (B, d2)
        query_source: torch.Tensor,   # (M, d1)
        query_target: torch.Tensor,   # (N, d2)
        sigma_source: float = None,
        sigma_target: float = None,
    ) -> torch.Tensor:

        B = base_source.shape[0]
        N = query_target.shape[0]
        M = query_source.shape[0]
        n = B + 1
        device = self.device

        # ── RBF kernels for SOURCE space ──────────────────────────
        base_src_sq  = (base_source ** 2).sum(1)                        # (B,)
        query_src_sq = (query_source ** 2).sum(1)                       # (M,)

        K_bb_dist = (base_src_sq.unsqueeze(1) + base_src_sq.unsqueeze(0)
                    - 2.0 * base_source @ base_source.T).clamp_min_(0.0)

        if sigma_source is None:
            upper = K_bb_dist.triu(diagonal=1).flatten()
            sigma_source = torch.sqrt(torch.median(upper[upper > 0]))

        inv2s_src = -1.0 / (2.0 * sigma_source ** 2)

        K_bb = torch.exp(K_bb_dist * inv2s_src)                         # (B, B)

        K_bq_dist = (base_src_sq.unsqueeze(1) + query_src_sq.unsqueeze(0)
                    - 2.0 * base_source @ query_source.T).clamp_min_(0.0)
        K_bq = torch.exp(K_bq_dist * inv2s_src)                        # (B, M)

        K_qq = torch.ones(M, device=device)                             # exp(0) = 1

        # ── RBF kernels for TARGET space ──────────────────────────
        base_tgt_sq  = (base_target ** 2).sum(1)                        # (B,)
        query_tgt_sq = (query_target ** 2).sum(1)                       # (N,)

        L_bb_dist = (base_tgt_sq.unsqueeze(1) + base_tgt_sq.unsqueeze(0)
                    - 2.0 * base_target @ base_target.T).clamp_min_(0.0)

        if sigma_target is None:
            upper = L_bb_dist.triu(diagonal=1).flatten()
            sigma_target = torch.sqrt(torch.median(upper[upper > 0]))

        inv2s_tgt = -1.0 / (2.0 * sigma_target ** 2)

        L_bb = torch.exp(L_bb_dist * inv2s_tgt)                        # (B, B)

        L_bq_dist = (base_tgt_sq.unsqueeze(1) + query_tgt_sq.unsqueeze(0)
                    - 2.0 * base_target @ query_target.T).clamp_min_(0.0)
        L_bq = torch.exp(L_bq_dist * inv2s_tgt)                        # (B, N)

        L_qq = torch.ones(N, device=device)                             # exp(0) = 1

        # ── Base-only statistics (computed once) ──────────────────
        K_bb_rowsum = K_bb.sum(1)
        K_bb_total  = K_bb_rowsum.sum()
        L_bb_rowsum = L_bb.sum(1)
        L_bb_total  = L_bb_rowsum.sum()

        KL_bb = (K_bb * L_bb).sum()
        KK_bb = (K_bb * K_bb).sum()
        LL_bb = (L_bb * L_bb).sum()

        # ── Batched graph fill (identical structure to linear) ────
        graph      = torch.zeros((N, M), device=device)
        batch_size = 50

        for i_start in tqdm(range(0, N, batch_size), desc="[CKAMethod] RBF-CKA batches"):
            i_end = min(i_start + batch_size, N)

            l_bi = L_bq[:, i_start:i_end]                              # (B, bs)
            l_ii = L_qq[i_start:i_end]                                  # (bs,)  ≡ 1

            L_rowsum_base = L_bb_rowsum.unsqueeze(1) + l_bi
            L_rowsum_last = l_bi.sum(0) + l_ii
            L_total       = L_bb_total + 2.0 * l_bi.sum(0) + l_ii

            LL_bq  = torch.einsum('bn,bn->n', l_bi, l_bi)
            LL_qq  = l_ii ** 2
            LL_sum = LL_bb + 2.0 * LL_bq + LL_qq
            LL_rs  = (L_rowsum_base ** 2).sum(0) + L_rowsum_last ** 2
            var_L  = LL_sum - (2.0/n)*LL_rs + (1.0/n**2)*(L_total**2)

            for j_start in range(0, M, batch_size):
                j_end = min(j_start + batch_size, M)

                k_bj = K_bq[:, j_start:j_end]                          # (B, ms)
                k_jj = K_qq[j_start:j_end]                              # (ms,)  ≡ 1

                K_rowsum_base = K_bb_rowsum.unsqueeze(1) + k_bj
                K_rowsum_last = k_bj.sum(0) + k_jj
                K_total       = K_bb_total + 2.0 * k_bj.sum(0) + k_jj

                KL_bq    = torch.einsum('bm,bn->mn', k_bj, l_bi)
                KL_qq    = k_jj.unsqueeze(1) * l_ii.unsqueeze(0)
                KL_sum   = KL_bb + 2.0 * KL_bq + KL_qq

                RS_base  = torch.einsum('bm,bn->mn', K_rowsum_base, L_rowsum_base)
                RS_last  = K_rowsum_last.unsqueeze(1) * L_rowsum_last.unsqueeze(0)
                RS       = RS_base + RS_last

                KL_total = K_total.unsqueeze(1) * L_total.unsqueeze(0)
                hsic     = KL_sum - (2.0/n)*RS + (1.0/n**2)*KL_total

                KK_bq  = torch.einsum('bm,bm->m', k_bj, k_bj)
                KK_qq  = k_jj ** 2
                KK_sum = KK_bb + 2.0 * KK_bq + KK_qq
                KK_rs  = (K_rowsum_base ** 2).sum(0) + K_rowsum_last ** 2
                var_K  = KK_sum - (2.0/n)*KK_rs + (1.0/n**2)*(K_total**2)

                denom  = torch.sqrt(var_K.unsqueeze(1) * var_L.unsqueeze(0)) + 1e-8
                graph[i_start:i_end, j_start:j_end] = (hsic / denom).T

        return graph
    
    def _vectorized_linear_cka_graph(
        self,
        base_source: torch.Tensor,   
        base_target: torch.Tensor,   
        query_source: torch.Tensor,   
        query_target: torch.Tensor,   
    ) -> torch.Tensor:                


        B  = base_source.shape[0]
        N  = query_target.shape[0]
        M  = query_source.shape[0]
        n  = B + 1

        device = self.device

        K_bb = base_source @ base_source.T
        L_bb = base_target @ base_target.T

        K_bq = base_source @ query_source.T   
        L_bq = base_target @ query_target.T   

        K_qq = (query_source * query_source).sum(dim=1)   
        L_qq = (query_target * query_target).sum(dim=1)   

        K_bb_rowsum = K_bb.sum(dim=1)
        K_bb_total  = K_bb_rowsum.sum()
        L_bb_rowsum = L_bb.sum(dim=1)
        L_bb_total  = L_bb_rowsum.sum()

        KL_bb = (K_bb * L_bb).sum()
        KK_bb = (K_bb * K_bb).sum()
        LL_bb = (L_bb * L_bb).sum()

        graph      = torch.zeros((N, M), device=device)
        batch_size = 50

        for i_start in tqdm(range(0, N, batch_size), desc="[CKAMethod] CKA batches"):

            i_end = min(i_start + batch_size, N)
            bs    = i_end - i_start

            l_bi = L_bq[:, i_start:i_end]
            l_ii = L_qq[i_start:i_end]

            L_rowsum_base = L_bb_rowsum.unsqueeze(1) + l_bi
            L_rowsum_last = l_bi.sum(0) + l_ii
            L_total       = L_bb_total + 2 * l_bi.sum(0) + l_ii

            LL_bq  = torch.einsum('bn,bn->n', l_bi, l_bi)
            LL_qq  = l_ii ** 2
            LL_sum = LL_bb + 2 * LL_bq + LL_qq
            LL_rs  = (L_rowsum_base ** 2).sum(0) + L_rowsum_last ** 2
            var_L  = LL_sum - (2.0/n)*LL_rs + (1.0/n**2)*(L_total**2)

            for j_start in range(0, M, batch_size):

                j_end = min(j_start + batch_size, M)
                ms    = j_end - j_start

                k_bj = K_bq[:, j_start:j_end]
                k_jj = K_qq[j_start:j_end]

                K_rowsum_base = K_bb_rowsum.unsqueeze(1) + k_bj
                K_rowsum_last = k_bj.sum(0) + k_jj
                K_total       = K_bb_total + 2 * k_bj.sum(0) + k_jj

                KL_bq    = torch.einsum('bm,bn->mn', k_bj, l_bi)
                KL_qq    = k_jj.unsqueeze(1) * l_ii.unsqueeze(0)
                KL_sum   = KL_bb + 2 * KL_bq + KL_qq

                RS_base  = torch.einsum('bm,bn->mn', K_rowsum_base, L_rowsum_base)
                RS_last  = K_rowsum_last.unsqueeze(1) * L_rowsum_last.unsqueeze(0)
                RS       = RS_base + RS_last

                KL_total = K_total.unsqueeze(1) * L_total.unsqueeze(0)
                hsic     = KL_sum - (2.0/n)*RS + (1.0/n**2)*KL_total

                KK_bq  = torch.einsum('bm,bm->m', k_bj, k_bj)
                KK_qq  = k_jj ** 2
                KK_sum = KK_bb + 2 * KK_bq + KK_qq
                KK_rs  = (K_rowsum_base ** 2).sum(0) + K_rowsum_last ** 2
                var_K  = KK_sum - (2.0/n)*KK_rs + (1.0/n**2)*(K_total**2)

                denom  = torch.sqrt(var_K.unsqueeze(1) * var_L.unsqueeze(0)) + 1e-8
                scores = (hsic / denom).T

                graph[i_start:i_end, j_start:j_end] = scores

        return graph
    
    def classify(self, 
        data: np.ndarray, 
        labels_emb: np.ndarray,
        support_embeddings: Dict[str, np.ndarray],
        use_cca = False,
        copying_exp = False,
        n_repeats = 5,
        translate=False,
        translation_std=0.01,
        translation_mean=0.0,
        experiment_name="cka_classification",
        **kwargs
        ) -> np.ndarray:

        train_source = support_embeddings["train_text"]
        train_target = support_embeddings["train_image"]
        
        test_source = labels_emb
        test_target = data

        assert train_source.shape[0] == train_target.shape[0], "Number of training samples must match between source and target."
        n_classes = len(labels_emb)
        
        if use_cca:
            cca = NormalizedCCA(sim_dim=700)

            train_source, train_target = cca.fit_transform_train_data(
                    train_source, 
                    train_target
                )
            dummy_data_target = np.zeros_like(test_target)
            dummy_data_source = np.zeros_like(test_source)
            
            test_source, _ = cca.transform_data(
                    test_source, 
                    dummy_data_target
                )
            
            _, test_target = cca.transform_data(
                    dummy_data_source, 
                    test_target
                )
        
        run_visualization(
            image_embeddings = train_target,
            text_embeddings  = train_source,
            n_samples        = train_target.shape[0],
            n_clusters       = 20,
            save_prefix       = f"{experiment_name}",
            perplexity       = 30,
        )
            
        train_source = torch.tensor(
            train_source,
            dtype=torch.float32, device=self.device
        )

        train_target = torch.tensor(
            train_target,
            dtype=torch.float32, device=self.device
        )


        test_source = torch.tensor(test_source, dtype=torch.float32, device=self.device)
        test_target = torch.tensor(test_target, dtype=torch.float32, device=self.device)


        source_total = torch.cat([train_source, test_source], dim=0)
        target_total = torch.cat([train_target, test_target], dim=0)

        source_total = stretch_representations(source_total)
        target_total = stretch_representations(target_total)

        n_train = train_source.shape[0]
        train_source = source_total[:n_train]
        train_target = target_total[:n_train]
        test_source  = source_total[n_train:]
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
            if copying_exp:
                random_idx = np.random.choice(base_idx)
                selected_base_target = train_target[random_idx].unsqueeze(0).repeat(n_repeats, 1)
                selected_base_source = train_source[random_idx].unsqueeze(0).repeat(n_repeats, 1)

                base_target = torch.cat((base_target[:random_idx], selected_base_target, base_target[random_idx+1:]), dim=0)
                base_source = torch.cat((base_source[:random_idx], selected_base_source, base_source[random_idx+1:]), dim=0)

                if translate:
                    base_source_noise = torch.randn(base_source.size(), device=self.device) * translation_std  + translation_mean
                    base_source[random_idx:random_idx+n_repeats] += base_source_noise

                    base_target_noise = torch.randn(base_target.size(), device=self.device) * translation_std  + translation_mean
                    base_target[random_idx:random_idx+n_repeats] += base_target_noise

            else:
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
            
            graph = self._vectorized_linear_cka_graph(
                base_source=base_source,
                base_target=base_target,
                query_source=test_source,
                query_target=test_target,
            )

        graph = graph.detach().cpu().numpy()
        print(f"\n[CKAMethod] base_source with shape: {base_source.shape}\n")
        print(f"\n[CKAMethod] base_target with shape: {base_target.shape}\n")
        print(f"\n[CKAMethod] query_source with shape: {test_source.shape}\n")
        print(f"\n[CKAMethod] query_target with shape: {test_target.shape}\n")
        print(f"\n[CKAMethod] CKA graph computed with shape: {graph.shape}\n")
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
        topk: int = 20,
        num_gt: int = 1,
        n_clusters: int = 20,
        direction: str = "i2t",
        copying_exp = False,
        n_repeats = 5,
        translate=False,
        translation_std=0.01,
        translation_mean=0.0,
        experiment_name="cka_retrieval",
        dynamic=False,
        **kwargs
    ):
        experiment_name = f"{experiment_name}_{direction}_base--{self.base_mode}_ncusters--{n_clusters}_copy--{copying_exp}_nrepeats--{n_repeats}_translate--{translate}"
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
        train_target = target_total[:n_train]
        test_source  = source_total[n_train:]
        test_target  = target_total[n_train:]

        # run_visualization(
        #     image_embeddings = train_target,
        #     text_embeddings  = train_source,
        #     n_samples        = train_target.shape[0],
        #     n_clusters       = 20,
        #     save_prefix       = f"{experiment_name}",
        #     perplexity       = 30,
        # )

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
                n_clusters=n_clusters,
                #min(self.base_samples, n_train_samples)
            )
            base_target = train_target[base_idx]
            base_source = train_source[base_idx]
            if copying_exp:
                
                
                random_idx = np.random.choice([i for i in range(base_target.shape[0])])
                selected_base_target = base_target[random_idx].unsqueeze(0).repeat(n_repeats, 1)
                selected_base_source = base_source[random_idx].unsqueeze(0).repeat(n_repeats, 1)

                base_target = torch.cat((base_target[:random_idx], selected_base_target, base_target[random_idx+1:]), dim=0)
                base_source = torch.cat((base_source[:random_idx], selected_base_source, base_source[random_idx+1:]), dim=0)
                print(f"[CKAMethod] Base shape: {base_target.shape[0]}, {base_source.shape[0]}")

                if translate:
                    base_source_noise = torch.randn(base_source.size(), device=self.device) * translation_std  + translation_mean
                    base_source[random_idx:random_idx+n_repeats] += base_source_noise

                    base_target_noise = torch.randn(base_target.size(), device=self.device) * translation_std  + translation_mean
                    base_target[random_idx:random_idx+n_repeats] += base_target_noise

                
        
            print(f"[CKAMethod] Base mode: clustering ({len(base_idx)} from {n_train_samples})")
        else:
            raise ValueError(f"base_mode must be 'clustering', 'random', or 'full'")

        if not dynamic:
            # run_visualization_kernels(
            #     image_embeddings = base_target,
            #     text_embeddings= base_source,
            #     n_samples        = base_target.shape[0],
            #     n_clusters       = 20,
            #     save_prefix       = f"{experiment_name}",
            #     seed             = 42,
            # )
            
            diagnostic_results = None#run_diagnostics(
            #     embeddings_text = base_source.cpu().numpy(),
            #     embeddings_image = base_target.cpu().numpy(),
            #     k = n_clusters
            # )

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
        if not dynamic:
            with torch.no_grad():
                graph = self._vectorized_linear_cka_graph(
                    base_source=base_source,
                    base_target=base_target,
                    query_source=restricted_source,
                    query_target=test_target[:n_queries],
                )
        else:
            diagnostic_results = {}
            graph              = torch.zeros((n_queries, n_documents), device=self.device)
        
            for i in tqdm(range(n_queries), desc="[CKA] per-query kNN"):

                b_img, b_txt, _ = select_base_by_knn(
                    query_image  = test_target[i].to(self.device),
                    train_images = train_target.to(self.device),
                    train_texts  = train_source.to(self.device),
                    k            = n_clusters,
                )
                b_txt = b_txt.to(self.device)
                b_img = b_img.to(self.device)
                # per-query base CKA
                graph[i] = self._vectorized_linear_cka_graph(
                    base_source  = b_txt,
                    base_target  = b_img,
                    query_source = restricted_source,
                    query_target = test_target[i:i+1],
                ).squeeze(0)

                if i == 100:
                    run_visualization_kernels(
                        image_embeddings = base_target,
                        text_embeddings= base_source,
                        n_samples        = base_target.shape[0],
                        n_clusters       = 20,
                        save_prefix       = f"{experiment_name}_query-{i}",
                        seed             = 42,
                    )
                if i in [0, 500, 1000]:  # run diagnostics on a few selected queries
                    r = run_diagnostics(
                        embeddings_text = base_source.cpu().numpy(),
                        embeddings_image = base_target.cpu().numpy(),
                        k = n_clusters
                    )
                    diagnostic_results[f"query_{i}"] = r

                    
            
        
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

        return all_hits#, diagnostic_results



