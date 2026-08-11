import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Returns: float scalar KL divergence averaged over the batch
    """
    # Your implementation here
    kl = -0.5 * (1+log_var-np.square(mu) -np.exp(log_var))
    per_sample_kl = np.sum(kl, axis=1)
    return np.mean(per_sample_kl, axis=0)
