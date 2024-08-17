import sys
import numpy as np
import src.kmeans as kmeans
import src.common as common
import src.naive_em as naive_em
import src.em as em

def run_kmeans():
    X = np.loadtxt("datasets/toy_data.txt")
    for K in range(1, 5):
        best_mixture = None
        best_post = None
        best_cost = np.inf
        
        for seed in range(5):
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            common.plot(X, mixture, post, f"K={K}, seed={seed}, cost={cost}")
            if cost < best_cost:
                best_cost = cost
                best_post = post
                best_mixture = mixture
            
        common.plot(X, best_mixture, best_post, f"Best of K={K}, cost={cost}")


def run_naive_em():
    X = np.loadtxt("datasets/toy_data.txt")
    for K in range(1, 5):
        best_mixture = None
        best_post = None
        best_log_likelihood = -np.inf

        for seed in range(5):
            mixture, post = common.init(X, K, seed)
            mixture, post, log_likelihood = naive_em.run(X, mixture, post)
            common.plot(X, mixture, post, f"K={K}, seed={seed}, log_likelihood={log_likelihood}")
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_post = post
                best_mixture = mixture
            
        common.plot(X, best_mixture, best_post, f"Best of K={K}, log_likelihood={log_likelihood}")


def main():
    if sys.argv[1] == "kmeans":
        run_kmeans()
        
    if sys.argv[1] == "naive_em":
        run_naive_em()


if __name__ == "__main__":
    main()