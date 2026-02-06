import numpy as np

def origin_centered(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean the data.
    
    Args:
        data: input data. shape: (num_samples, dim)
        
    Returns:
        centered_data: data after zero-meaning. shape: (num_samples, dim)
        mean: the mean of the data. shape: (dim,)
    """
    mean = data.mean(axis=0)
    return data - mean, mean
