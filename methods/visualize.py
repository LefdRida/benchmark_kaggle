"""
8-panel visualization with t-SNE (better local structure) and 20 clusters:
  Row 1: Image kernel | Text kernel    (sorted by IMAGE clusters)
  Row 2: Image t-SNE  | Text t-SNE     (colored by IMAGE clusters)
  Row 3: Image kernel | Text kernel    (sorted by TEXT clusters)
  Row 4: Image t-SNE  | Text t-SNE     (colored by TEXT clusters)
"""

import numpy as np
import matplotlib.pyplot as plt
 
from .cka_core import linear_CKA
 


import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Core diagnostic functions
# ------------------------------------------------------------------
 
def greedy_erank_selection(embeddings, k, batch_size=50):
    """
    Greedily selects points that maximize effective rank.
    batch_size: evaluate candidates in batches for speed.
    """
    n, d = embeddings.shape
    candidates = set(range(n))
    selected = [np.random.randint(n)]
    candidates.discard(selected[0])

    for step in range(k - 1):
        best_idx, best_erank = -1, -1

        # subsample candidates for speed at large n
        if len(candidates) > batch_size * 10:
            eval_set = np.random.choice(
                list(candidates), batch_size * 10, replace=False
            )
        else:
            eval_set = list(candidates)

        current = embeddings[selected]

        for idx in eval_set:
            trial = np.vstack([current, embeddings[idx]])
            K = trial @ trial.T
            Kc = center_kernel(K)
            eigvals = np.linalg.eigvalsh(Kc)
            eigvals = eigvals[eigvals > 1e-10]
            p = eigvals / eigvals.sum()
            erank = np.exp(-np.sum(p * np.log(p)))

            if erank > best_erank:
                best_erank = erank
                best_idx = idx

        selected.append(best_idx)
        candidates.discard(best_idx)

        if step % 50 == 0:
            print(f"Step {step+1}/{k-1}, erank = {best_erank:.1f}")

    return np.array(selected)

def center_kernel(K: np.ndarray) -> np.ndarray:
    """Apply double centering: K_c = HKH where H = I - 11^T/n."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H
 
 
def kernel_eigenvalues(K_centered: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return positive eigenvalues of a centered kernel matrix, sorted descending."""
    eigvals = np.linalg.eigvalsh(K_centered)
    eigvals = eigvals[eigvals > tol]          # drop zero / negative numerical noise
    return np.sort(eigvals)[::-1]
 
 
def condition_number(eigvals: np.ndarray) -> float:
    """Ratio of largest to smallest positive eigenvalue."""
    return eigvals[0] / eigvals[-1]
 
 
def effective_rank(eigvals: np.ndarray) -> float:
    """exp(Shannon entropy of normalized eigenvalue spectrum)."""
    p = eigvals / eigvals.sum()
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

def run_diagnostics(embeddings_text: np.ndarray,
                    embeddings_image: np.ndarray,
                    k: int = 20
                    ) -> dict:
    """
    Parameters
    ----------
    embeddings_text  : (N_max, d_text) array — text embeddings for ALL anchor
                       candidates, ordered so the first k rows correspond to
                       the k cluster-selected anchors.
    embeddings_image : (N_max, d_image) array — same for image embeddings.
    cluster_counts   : list of k values to evaluate.
 
    Returns
    -------
    Dictionary with arrays of diagnostics per k.
    """
    results = {
        "k": [],
        "cond_text": [],
        "cond_image": [],
        "erank_text": [],
        "erank_image": [],
        "erank_ratio_text": [],
        "erank_ratio_image": [],
    }
 
    #idx = greedy_erank_selection(embeddings_image, k=350)
    cka_score = linear_CKA(
        torch.tensor(embeddings_text, dtype=torch.float32),
        torch.tensor(embeddings_image, dtype=torch.float32)
    ).item()
    results["cka_score"] = cka_score
    # --- build and center linear kernel matrices ---
    K_t = embeddings_text @ embeddings_text.T             # (k, k)
    K_i = embeddings_image @ embeddings_image.T
    diff_k_it = K_i - K_t
    diff_k_ti = K_t - K_i
    diff_k_it_norm = np.linalg.norm(K_i - K_t, 'fro')
    diff_k_ti_norm = np.linalg.norm(K_t - K_i, 'fro')
    K_t_norm = np.linalg.norm(K_t, 'fro')
    K_i_norm = np.linalg.norm(K_i, 'fro')

    Kc_t = K_t#center_kernel(K_t)
    Kc_i = K_i#center_kernel(K_i)

    Kc_diff_it = diff_k_it#center_kernel(diff_k_it)
    Kc_diff_ti = diff_k_ti#center_kernel(diff_k_ti)

    # --- eigenvalues ---
    eig_t = kernel_eigenvalues(Kc_t)
    eig_i = kernel_eigenvalues(Kc_i)
    eig_diff_it = kernel_eigenvalues(Kc_diff_it)
    eig_diff_ti = kernel_eigenvalues(Kc_diff_ti)

    # --- store diagnostics ---
    results["shape_text"] = embeddings_text.shape
    results["shape_image"] = embeddings_image.shape
    results["diff_k_it_norm"] = diff_k_it_norm
    results["diff_k_ti_norm"] = diff_k_ti_norm
    results["K_t_norm"] = K_t_norm
    results["K_i_norm"] = K_i_norm
    results["k"] = k
    results["text_min_eig"] = eig_t[-1]
    results["image_min_eig"] = eig_i[-1]
    results["text_max_eig"] = eig_t[0]
    results["image_max_eig"] = eig_i[0]
    try:
        results["log_text_min_eig"] = np.log(eig_t[-1])
    except:
        results["log_text_min_eig"] = 0
    try:
        results["log_image_min_eig"] = np.log(eig_i[-1])
    except:
        results["log_image_min_eig"] = 0
    try:
        results["log_text_max_eig"] = np.log(eig_t[0])
    except:
        results["log_text_max_eig"] = 0
    try:
        results["log_image_max_eig"] = np.log(eig_i[0])
    except:
        results["log_image_max_eig"] = 0
        
    results["cond_text"] = condition_number(eig_t)
    results["cond_image"] = condition_number(eig_i)
    try:
        results["log_cond_text"] = results["log_text_max_eig"]/results["log_text_min_eig"]
    except:
        results["log_cond_text"] = 0
    try:        
        results["log_cond_image"] = results["log_image_max_eig"]/results["log_image_min_eig"]
    except:
        results["log_cond_image"] = 0
    results["cond_diff_it"] = condition_number(eig_diff_it)
    results["cond_diff_ti"] = condition_number(eig_diff_ti)
    results["erank_text"] = effective_rank(eig_t)
    results["erank_image"] = effective_rank(eig_i)
    results["erank_diff_it"] = effective_rank(eig_diff_it)
    results["erank_diff_ti"] = effective_rank(eig_diff_ti)
    results["erank_ratio_text"] = effective_rank(eig_t) / k
    results["erank_ratio_image"] = effective_rank(eig_i) / k
    results["erank_ratio_diff_it"] = effective_rank(eig_diff_it) / k
    results["erank_ratio_diff_ti"] = effective_rank(eig_diff_ti) / k

 
    return results




def run_visualization_kernels(
    image_embeddings: np.ndarray,
    text_embeddings:  np.ndarray,
    n_samples:        int = 1000,
    n_clusters:       int = 20,
    save_prefix:      str = "base_analysis",
    seed:             int = 42,
):
    np.random.seed(seed)
    N = image_embeddings.shape[0]

    # ── Subsample ─────────────────────────────────────────────────────
    # if n_samples < N:
    #     idx = np.random.choice(N, size=n_samples, replace=False)
    #     idx.sort()
    # else:
    #     idx = np.arange(N)
    #     n_samples = N

    imgs = torch.tensor(image_embeddings, dtype=torch.float32)
    txts = torch.tensor(text_embeddings,  dtype=torch.float32)

    print(f"[Viz] Using {n_samples} samples, {n_clusters} clusters")

    # ── Cosine similarity matrices
    imgs_norm = F.normalize(imgs, dim=1)
    txts_norm = F.normalize(txts, dim=1)

    sim_img = (imgs_norm @ imgs_norm.T).cpu().numpy()
    sim_txt = (txts_norm @ txts_norm.T).cpu().numpy()

    # Cluster in BOTH spaces
    print("[Viz] Clustering...")
    km_img = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels_img = km_img.fit_predict(imgs.cpu().numpy())
    sort_by_img = np.argsort(labels_img)

    km_txt = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels_txt = km_txt.fit_predict(txts.cpu().numpy())
    sort_by_txt = np.argsort(labels_txt)

    print("[Viz] Plotting kernels...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    cmap_kernel = 'RdBu_r'
    n_show = min(5000, n_samples)

    # Row 0: sorted by image clusters
    ax = axes[0, 0]
    im = ax.imshow(sim_img[sort_by_img][:, sort_by_img][:n_show, :n_show],
                   cmap=cmap_kernel, vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_title('Image Kernel\n(sorted by image clusters)', fontsize=12)
    ax.set_xlabel('Sample index'); ax.set_ylabel('Sample index')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    im = ax.imshow(sim_txt[sort_by_img][:, sort_by_img][:n_show, :n_show],
                   cmap=cmap_kernel, vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_title('Text Kernel\n(same sort → image clusters hold?)', fontsize=12)
    ax.set_xlabel('Sample index'); ax.set_ylabel('Sample index')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 1: sorted by text clusters
    ax = axes[1, 0]
    im = ax.imshow(sim_img[sort_by_txt][:, sort_by_txt][:n_show, :n_show],
                   cmap=cmap_kernel, vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_title('Image Kernel\n(sorted by text clusters → blocks?)', fontsize=12)
    ax.set_xlabel('Sample index'); ax.set_ylabel('Sample index')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    im = ax.imshow(sim_txt[sort_by_txt][:, sort_by_txt][:n_show, :n_show],
                   cmap=cmap_kernel, vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_title('Text Kernel\n(sorted by text clusters → grouped ✓)', fontsize=12)
    ax.set_xlabel('Sample index'); ax.set_ylabel('Sample index')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    save_path = f"{save_prefix}_kernels.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Viz] Saved to: {save_path}")

def run_visualization(
    image_embeddings: np.ndarray,
    text_embeddings:  np.ndarray,
    n_samples:        int = 1000,
    n_clusters:       int = 20,
    save_prefix:      str = "base_analysis",
    seed:             int = 42,
    perplexity:       int = 30,
):
    np.random.seed(seed)
    N = image_embeddings.shape[0]

    # ── Subsample ─────────────────────────────────────────────────────
    if n_samples < N:
        idx = np.random.choice(N, size=n_samples, replace=False)
        idx.sort()
    else:
        idx = np.arange(N)
        n_samples = N

    imgs = torch.tensor(image_embeddings[idx], dtype=torch.float32)
    txts = torch.tensor(text_embeddings[idx],  dtype=torch.float32)

    print(f"[Viz] Using {n_samples} samples, {n_clusters} clusters")

    # ── Cosine similarity matrices
    imgs_norm = F.normalize(imgs, dim=1)
    txts_norm = F.normalize(txts, dim=1)

    sim_img = (imgs_norm @ imgs_norm.T).cpu().numpy()
    sim_txt = (txts_norm @ txts_norm.T).cpu().numpy()

    # Cluster in BOTH spaces 
    print("[Viz] Clustering...")
    km_img = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels_img = km_img.fit_predict(imgs.cpu().numpy())
    sort_by_img = np.argsort(labels_img)

    km_txt = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels_txt = km_txt.fit_predict(txts.cpu().numpy())
    sort_by_txt = np.argsort(labels_txt)

    # t-SNE 
    print(f"[Viz] Running t-SNE (perplexity={perplexity})...")
    tsne_img = TSNE(
        n_components=2, perplexity=perplexity,
        random_state=seed, n_iter=1000
    ).fit_transform(imgs.cpu().numpy())

    tsne_txt = TSNE(
        n_components=2, perplexity=perplexity,
        random_state=seed, n_iter=1000
    ).fit_transform(txts.cpu().numpy())

    print("[Viz] t-SNE done, plotting...")


    fig, axes = plt.subplots(4, 2, figsize=(16, 28))

    cmap_kernel   = 'RdBu_r'
    cmap_clusters = 'tab20'
    n_show = 5000 #min(2000, n_samples)
    dot_size = 12
    dot_alpha = 0.7


    
    # fig.text(0.5, 0.97, 'Clustered by IMAGE space',
    #          ha='center', fontsize=18, fontweight='bold',
    #          bbox=dict(boxstyle='round', facecolor='#D6EAF8', alpha=0.8))

    # Heatmaps sorted by image clusters
    ax = axes[0, 0]
    im = ax.imshow(sim_img[sort_by_img][:, sort_by_img][:n_show, :n_show],
                   cmap=cmap_kernel, vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_title('Image Kernel\n(sorted by image clusters)', fontsize=12)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Sample index')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    im = ax.imshow(sim_txt[sort_by_img][:, sort_by_img][:n_show, :n_show],
                   cmap=cmap_kernel, vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_title('Text Kernel\n(same sort → image clusters hold?)', fontsize=12)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Sample index')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 1: t-SNE colored by image clusters
    ax = axes[1, 0]
    sc = ax.scatter(tsne_img[:, 0], tsne_img[:, 1],
                    c=labels_img, cmap=cmap_clusters,
                    s=dot_size, alpha=dot_alpha, edgecolors='none')
    ax.set_title('Image t-SNE\n(colored by image clusters → grouped ✓)', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='Image cluster')

    ax = axes[1, 1]
    sc = ax.scatter(tsne_txt[:, 0], tsne_txt[:, 1],
                    c=labels_img, cmap=cmap_clusters,
                    s=dot_size, alpha=dot_alpha, edgecolors='none')
    ax.set_title('Text t-SNE\n(colored by image clusters → scattered?)', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='Image cluster')

    # fig.text(0.5, 0.49, 'Clustered by TEXT space',
    #          ha='center', fontsize=18, fontweight='bold',
    #          bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.8))

    #  Heatmaps sorted by text clusters
    ax = axes[2, 0]
    im = ax.imshow(sim_img[sort_by_txt][:, sort_by_txt][:n_show, :n_show],
                   cmap=cmap_kernel, vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_title('Image Kernel\n(sorted by text clusters → blocks?)', fontsize=12)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Sample index')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[2, 1]
    im = ax.imshow(sim_txt[sort_by_txt][:, sort_by_txt][:n_show, :n_show],
                   cmap=cmap_kernel, vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_title('Text Kernel\n(sorted by text clusters → grouped ✓)', fontsize=12)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Sample index')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 3: t-SNE colored by text clusters
    ax = axes[3, 0]
    sc = ax.scatter(tsne_img[:, 0], tsne_img[:, 1],
                    c=labels_txt, cmap=cmap_clusters,
                    s=dot_size, alpha=dot_alpha, edgecolors='none')
    ax.set_title('Image t-SNE\n(colored by text clusters → scattered?)', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='Text cluster')

    ax = axes[3, 1]
    sc = ax.scatter(tsne_txt[:, 0], tsne_txt[:, 1],
                    c=labels_txt, cmap=cmap_clusters,
                    s=dot_size, alpha=dot_alpha, edgecolors='none')
    ax.set_title('Text t-SNE\n(colored by text clusters → grouped ✓)', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='Text cluster')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = f"{save_prefix}_tsne_clusters.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Viz] Saved to: {save_path}")