"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from src.common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    K = mixture.p.shape[0]

    i_gen_by_j = np.zeros((n, K))
    post = np.zeros((n, K))
    for i in range(n):
        c = X[i] != 0.
        for j in range(K):
            s = X[i, c] - mixture.mu[j, c]
            i_gen_by_j[i, j] = (-1 / (2 * mixture.var[j]) * (s * s).sum()) - c.sum() / 2 * np.log(2 * np.pi * mixture.var[j])

    post =  np.log(mixture.p + 1e-16) + i_gen_by_j
    post_max = np.max(post, axis=1).reshape(-1, 1)
    post -= post_max + (logsumexp(post - post_max, axis=1).reshape(-1, 1))

    log_likelihood = np.exp(post) * (np.log(mixture.p + 1e-16) + i_gen_by_j - post)
 
    return np.exp(post), log_likelihood.sum()


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, K = post.shape
    d = X.shape[1]
    n_hat = post.sum(axis=0)
    p_hat = n_hat / n
    mu_hat = np.zeros(mixture.mu.shape)
    c = X != 0.

    for j in range(K):
        for l in range(d):
            if post[c[:, l], j].sum() >= 1.:
                mu_hat[j, l] = (post[:, j] * X[:, l]).sum() / post[c[:, l], j].sum()
            else:
                mu_hat[j, l] = mixture.mu[j, l]

    var_hat = np.zeros(K)
    for j in range(K):
        for i in range(n):
            s = X[i, c[i]] - mu_hat[j, c[i]]
            var_hat[j] += (post[i, j] * (s * s).sum()).sum()
        var_hat[j] /= np.array([c[u].sum() * post[u, j] for u in range(n)]).sum()
    
    var_hat[var_hat < min_variance] = min_variance
    
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
        mixture = mstep(X, post, mixture)
    
    return mixture, post, new_log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
