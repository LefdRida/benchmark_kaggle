import numpy as np
from sklearn import metrics
from typing import Any, Dict, List
from base.base import AbsTask, AbsModel, AbsMethod
import torch
from IsoScore.IsoScore import *
# -*- coding: utf-8 -*-
"""
GPU-accelerated IsoScore implementation using CuPy.

Drop-in replacement for the CPU version. Falls back to NumPy
automatically if CuPy is not available.

Input convention (same as original):
    points : array of shape (num_samples, embedding_dim)
             i.e. each ROW is one data point.
"""
from __future__ import unicode_literals

try:
    import cupy as cp
    from cupy.linalg import norm as cp_norm
    from IsoScore.IsoScore import *

    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp          # transparent fallback
    from numpy.linalg import norm as cp_norm

    GPU_AVAILABLE = False

import numpy as np


# -----------------------------------------------------------------
#  Helper Functions (GPU)
# -----------------------------------------------------------------

def pca_normalization_gpu(points_gpu):
    """
    PCA via eigen-decomposition of the covariance matrix on the GPU.

    sklearn.decomposition.PCA is CPU-only, so we replace it with an
    explicit covariance → eigen-decomposition path that runs entirely
    on the device.

    Parameters
    ----------
    points_gpu : cupy.ndarray, shape (num_samples, embedding_dim)

    Returns
    -------
    projected : cupy.ndarray, shape (num_samples, embedding_dim)
        Data projected onto its principal components (centered).
    """
    # Center the data  (mean of each column)
    mean = cp.mean(points_gpu, axis=0, keepdims=True)
    centered = points_gpu - mean                        # (N, D)

    # Covariance matrix  (D, D)
    # Using  (X^T X) / (N-1)  which is equivalent to np.cov(X.T)
    n = centered.shape[0]
    cov = centered.T @ centered / (n - 1)               # (D, D)

    # Eigen-decomposition (symmetric → use eigh for speed / stability)
    eigenvalues, eigenvectors = cp.linalg.eigh(cov)     # ascending order

    # Sort descending (to match sklearn PCA convention)
    idx = cp.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project data onto principal components
    projected = centered @ eigenvectors                  # (N, D)
    return projected


def get_diag_of_cov_gpu(points_gpu):
    """
    Compute the diagonal of the covariance matrix.

    Parameters
    ----------
    points_gpu : cupy.ndarray, shape (num_samples, embedding_dim)
        Already PCA-transformed points.

    Returns
    -------
    cov_diag : cupy.ndarray, shape (embedding_dim,)
    """
    # Transpose so each ROW is a variable (dimension) → shape (D, N)
    X = points_gpu.T
    n = X.shape[1]
    mean = cp.mean(X, axis=1, keepdims=True)
    centered = X - mean
    # Diagonal of cov = variance of each row = mean of squared deviations
    cov_diag = cp.sum(centered * centered, axis=1) / (n - 1)
    return cov_diag


def normalize_diagonal_gpu(cov_diag):
    """Step 4 – normalize the diagonal so its L2 norm equals sqrt(n)."""
    n = cov_diag.shape[0]
    return (cov_diag * cp.sqrt(float(n))) / cp_norm(cov_diag)


def get_isotropy_defect_gpu(cov_diag_normalized):
    """Step 5 – isotropy defect: distance from the identity diagonal."""
    n = cov_diag_normalized.shape[0]
    iso_diag = cp.ones(n, dtype=cov_diag_normalized.dtype)
    l2 = cp_norm(cov_diag_normalized - iso_diag)
    normalization_constant = cp.sqrt(2.0 * (n - cp.sqrt(float(n))))
    return float(l2 / normalization_constant)


def get_IsoScore_gpu(isotropy_defect, n):
    """Steps 6 & 7 – final IsoScore from the defect and dimensionality."""
    score = ((n - (isotropy_defect ** 2) * (n - np.sqrt(n))) ** 2 - n) / (
        n * (n - 1)
    )
    return float(score)


# -----------------------------------------------------------------
#  Main entry point
# -----------------------------------------------------------------

def IsoScoreGPU(points):
    """
    Compute the IsoScore of a point cloud on the GPU.

    Parameters
    ----------
    points : array-like, shape (num_samples, embedding_dim)

    Returns
    -------
    score : float
    """
    # Move to GPU
    points_gpu = cp.asarray(points, dtype=cp.float64)

    # Step 2 – PCA
    points_pca = pca_normalization_gpu(points_gpu)

    # Step 3 – diagonal of covariance
    cov_diag = get_diag_of_cov_gpu(points_pca)

    # Step 4 – normalize diagonal
    cov_diag_normalized = normalize_diagonal_gpu(cov_diag)

    # Step 5 – isotropy defect
    isotropy_defect = get_isotropy_defect_gpu(cov_diag_normalized)

    # Steps 6 & 7 – final score
    n = points_gpu.shape[1]  # embedding dimensionality
    score = get_IsoScore_gpu(isotropy_defect, n)

    return score

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

        image_isoscore = IsoScoreGPU(train_image)
        text_isoscore = IsoScoreGPU(train_text)

        geo_preserve_metric = (1/train_image.shape[0]**2) * np.linalg.norm(train_image@train_image.T  - train_text@train_text.T)**2

        return {"geo_preserve_metric": geo_preserve_metric, "image_isoscore": image_isoscore, "text_isoscore": text_isoscore}
@


    
    
