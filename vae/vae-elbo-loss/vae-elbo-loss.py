import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    # Your implementation here
    
    recon = np.mean(np.sum(np.square(x-x_recon),axis = 1))
    kl_terms = -0.5 * (1 + log_var - np.square(mu) - np.exp(log_var)) 
    kl = np.mean(np.sum(kl_terms,axis=1))
    total = recon + kl
    return {
        "total": total,
        "recon": recon,
        "kl": kl
    }
