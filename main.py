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
        best_cost = sys.float_info.max
        
        for seed in range(5):
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            common.plot(X, mixture, post, f"K={K}, seed={seed}, cost={cost}")
            if cost < best_cost:
                best_cost = cost
                best_post = post
                best_mixture = mixture
            
        common.plot(X, best_mixture, best_post, f"Best of K={K}, cost={cost}")
        

def main():
    if sys.argv[1] == "kmeans":
        run_kmeans()


if __name__ == "__main__":
    main()