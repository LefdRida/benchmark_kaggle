"""Canonical Correlation Analysis (CCA) related functions."""

import pickle
from pathlib import Path
import joblib
import numpy as np
from cca_zoo.linear import CCA
from typing import Tuple

def origin_centered(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-mean the data.
    
    Args:
        data: input data. shape: (num_samples, dim)
        
    Returns:
        centered_data: data after zero-meaning. shape: (num_samples, dim)
        mean: the mean of the data. shape: (dim,)
    """
    mean = data.mean(axis=0)
    return data - mean, mean

class NormalizedCCA:
    """Canonical Correlation Analysis (CCA) class which automatically zero-mean data."""

    def __init__(self, sim_dim: int | None = None, equal_weights:bool = False) -> None:
        """Initialize the CCA model."""
        self.traindata1_mean = None
        self.traindata2_mean = None
        self.sim_dim = sim_dim
        self.equal_weights = equal_weights

    def fit_transform_train_data(
        self, traindata1: np.ndarray, traindata2: np.ndarray
    ) -> None:
        """Fit the CCA model to the training data.

        Args:
            traindata1: the first training data. shape: (num_samples, dim)
            traindata2: the second training data. shape: (num_samples, dim)

        Returns:
            traindata1: the first training data after CCA. shape: (num_samples, dim)
            traindata2: the second training data after CCA. shape: (num_samples, dim)
            corr_coeff: the correlation coefficient. shape: (dim,)
        """
        # Check the shape of the training data
    
        # zero mean data
        traindata1, traindata1_mean = origin_centered(traindata1)
        traindata2, traindata2_mean = origin_centered(traindata2)
        print(traindata1.shape)
        print(traindata2.shape)
        # print(traindata1_mean)
        # print(traindata2_mean)
        self.traindata1_mean, self.traindata2_mean = traindata1_mean, traindata2_mean
        self.traindata1, self.traindata2 = traindata1, traindata2
        
        # check if training data is zero-mean
        # assert np.allclose(
        #     traindata1.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        # ), f"traindata1align not zero mean: {max(abs(traindata1.mean(axis=0)))}"
        # assert np.allclose(
        #     traindata2.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        # ), f"traindata2align not zero mean: {max(abs(traindata2.mean(axis=0)))}"
        self.sim_dim = min(self.sim_dim, traindata1.shape[1], traindata2.shape[1])
        # CCA dimensionality reduction
        if self.sim_dim is not None:
            self.cca = CCA(latent_dimensions=self.sim_dim)
        else:
            raise ValueError("Please set the sim_dim for CCA.")
        traindata1, traindata2 = self.cca.fit_transform((traindata1, traindata2))
        if self.equal_weights:
            corr_coeff = np.ones((traindata2.shape[1],))  # dim,
        else:
            corr_coeff = (
                np.diag(traindata1.T @ traindata2) / traindata1.shape[0]
            )  # dim,
        # assert (
        #     corr_coeff >= 0
        # ).all(), f"Correlation should be non-negative. {corr_coeff}"
        # assert (
        #     corr_coeff <= 1.05  # noqa: PLR2004
        # ).all(), f"Correlation should be less than 1. {corr_coeff}"

        self.corr_coeff = corr_coeff
        self.traindata1, self.traindata2 = traindata1, traindata2
        

    def transform_data(
        self, data1: np.ndarray, data2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform the data using the fitted CCA model.

        Args:
            data1: the first data. shape: (num_samples, dim)
            data2: the second data. shape: (num_samples, dim)

        Returns:
            data1: the first transformed data. shape: (num_samples, dim)
            data2: the second transformed data. shape: (num_samples, dim)
        """
        assert self.traindata1_mean is not None, "Please fit the cca model first."
        assert self.traindata2_mean is not None, "Please fit the cca model first."

        # zero mean data and transform
        data1 = data1 - self.traindata1_mean
        dummy_data2 = np.zeros((data1.shape[0], data2.shape[1]))
        data2 = data2 - self.traindata2_mean
        dummy_data1 = np.zeros((data2.shape[0], data1.shape[1]))
        
        data1, _ = self.cca.transform((data1, dummy_data2))
        _, data2 = self.cca.transform((dummy_data1, data2))
        return data1, data2



class ReNormalizedCCA:
    """
    Canonical Correlation Analysis (CCA) class with regularization.

    Automatically handles rank deficiency by:
    1. Adding regularization to both covariances
    2. Preventing numerical instability
    """

    def __init__(
        self, sim_dim: int | None = None, equal_weights:bool = False
    ) -> None:
        self.traindata1_mean = None
        self.traindata2_mean = None
        self.sim_dim = sim_dim
        self.equal_weights = equal_weights
        self.regularization = 0.01
        self.text_rank = None

    def fit_transform_train_data(
         self, traindata1: np.ndarray, traindata2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        traindata1 = traindata1.astype(np.float32)
        traindata2 = traindata2.astype(np.float32)

        traindata1, traindata1_mean = origin_centered(traindata1)
        traindata2, traindata2_mean = origin_centered(traindata2)

        self.traindata1_mean = traindata1_mean
        self.traindata2_mean = traindata2_mean

        assert np.allclose(
            traindata1.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        ), f"traindata1 not zero mean: {max(abs(traindata1.mean(axis=0)))}"
        assert np.allclose(
            traindata2.mean(axis=0), 0, atol=1e-3, rtol=1e-4
        ), f"traindata2 not zero mean: {max(abs(traindata2.mean(axis=0)))}"

        # STEP 1: Detect text rank
        C22 = (traindata2.T @ traindata2) / len(traindata2)
        eigenvalues = np.linalg.eigvalsh(C22)
        self.text_rank = np.sum(eigenvalues > 1e-6)

        # STEP 2: Set CCA dimension
        self.sim_dim = min(self.sim_dim, traindata1.shape[1], traindata2.shape[1])

        # STEP 3: Compute covariance inverses
        # if self.use_pseudo_inverse:
        #     sigma_z1_inv = np.linalg.pinv(traindata1.T @ traindata1)
        #     sigma_z2_inv = np.linalg.pinv(traindata2.T @ traindata2)

        #     eps = 1e-6
        #     sigma_z1_inv_sqrt = sqrtm(sigma_z1_inv + eps * np.eye(traindata1.shape[1]))
        #     sigma_z2_inv_sqrt = sqrtm(sigma_z2_inv + eps * np.eye(traindata2.shape[1]))
        # else:
        sigma_z1_inv = np.linalg.inv(
            traindata1.T @ traindata1 + self.regularization * np.eye(traindata1.shape[1])
        )
        sigma_z2_inv = np.linalg.inv(
            traindata2.T @ traindata2 + self.regularization * np.eye(traindata2.shape[1])
        )

        sigma_z1_inv_sqrt = sqrtm(sigma_z1_inv)
        sigma_z2_inv_sqrt = sqrtm(sigma_z2_inv)

        if np.iscomplexobj(sigma_z1_inv_sqrt) or np.iscomplexobj(sigma_z2_inv_sqrt):
            sigma_z1_inv_sqrt = np.real(sigma_z1_inv_sqrt)
            sigma_z2_inv_sqrt = np.real(sigma_z2_inv_sqrt)

        # STEP 4: SVD
        svd_mat = sigma_z1_inv_sqrt @ traindata1.T @ traindata2 @ sigma_z2_inv_sqrt
        u, s, vh = np.linalg.svd(svd_mat, full_matrices=False)

        self.A = sigma_z1_inv_sqrt @ u
        self.B = sigma_z2_inv_sqrt @ vh.T

        # STEP 5: Correlation coefficients
        if self.equal_weights:
            corr_coeff = np.ones((self.sim_dim,))
        else:
            corr_coeff = s

        if (corr_coeff < 0).any():
            corr_coeff = np.abs(corr_coeff)

        self.corr_coeff = corr_coeff

        # STEP 6: Transform training data
        self.traindata1 = (traindata1 @ self.A)[:, :self.sim_dim]
        self.traindata2 = (traindata2 @ self.B)[:, :self.sim_dim]

        if np.iscomplexobj(self.traindata1) or np.iscomplexobj(self.traindata2):
            self.traindata1 = np.real(self.traindata1)
            self.traindata2 = np.real(self.traindata2)

        return self.traindata1, self.traindata2, corr_coeff[:self.sim_dim]

    def transform_data(
        self, data1: np.ndarray, data2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        """Transform the data using the fitted CCA model.

        Args:
            data1: the first data. shape: (num_samples, dim)
            data2: the second data. shape: (num_samples, dim)

        Returns:
            data1: the first transformed data. shape: (num_samples, dim)
            data2: the second transformed data. shape: (num_samples, dim)
        """
        assert self.traindata1_mean is not None, "Please fit the cca model first."
        assert self.traindata2_mean is not None, "Please fit the cca model first."

        data1 = data1.astype(np.float32)
        data2 = data2.astype(np.float32)
        
        # zero mean data and transform
        data1 = data1 - self.traindata1_mean
        data2 = data2 - self.traindata2_mean
        data1 = (data1 @ self.A)
        data2 = (data2 @ self.B)

        if np.iscomplexobj(data1):
            data1 = np.real(data1)
        if np.iscomplexobj(data2):
            data2 = np.real(data2)

        return data1, data2

