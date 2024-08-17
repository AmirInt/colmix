"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from src.common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = mixture.p.shape[0]

    i_gen_by_j = np.zeros((n, K))
    post = np.zeros((n, K))
    for j in range(K):
        s = X - mixture.mu[j]
        i_gen_by_j[:, j] = np.exp(-1 / (2 * mixture.var[j]) * (s * s).sum(axis=1)) / np.sqrt(2 * np.pi * mixture.var[j]) ** d
    
    post =  mixture.p * i_gen_by_j
    post /= post.sum(axis=1).reshape(-1, 1)

    log_likelihood = post * np.log(mixture.p * i_gen_by_j / post)
 
    return post, log_likelihood.sum()
    

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, K = post.shape
    d = X.shape[1]
    n_hat = post.sum(axis=0)
    p_hat = n_hat / n
    mu_hat = post.T @ X / n_hat.reshape(-1, 1)
    var_hat = np.zeros(K)
    for j in range(K):
        s = X - mu_hat[j]
        var_hat[j] = (post[:, j] * (s * s).sum(axis=1)).sum() / (d * n_hat[j])
    
    return GaussianMixture(mu=mu_hat, var=var_hat, p=p_hat)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_likelihood = None
    new_log_likelihood = None
    epsilon = 1e-6
    while old_log_likelihood is None or \
          new_log_likelihood is None or \
          abs(new_log_likelihood - old_log_likelihood) >= epsilon * abs(new_log_likelihood):
        
        old_log_likelihood = new_log_likelihood

        post, new_log_likelihood = estep(X, mixture)

        mixture = mstep(X, post)
    
    return mixture, post, new_log_likelihood
