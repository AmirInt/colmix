# colmix

Our task is to build a mixture model for collaborative filtering. We are given a data matrix containing movie ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled. The goal is to predict all the remaining entries of the matrix.

We will use mixtures of Gaussians to solve this problem. The model assumes that each user's rating profile is a sample from a mixture model. In other words, we have K possible types of users and, in the context of each user, we must sample a user type and then the rating profile from the Gaussian distribution associated with the type. We will use the Expectation Maximization (EM) algorithm to estimate such a mixture from a partially observed rating matrix. The EM algorithm proceeds by iteratively assigning (softly) users to types (E-step) and subsequently re-estimating the Gaussians associated with each type (M-step). Once we have the mixture, we can use it to predict values for all the missing entries in the data matrix.

In this project, we made our calculation in the log domain for numerical stability purposes and avoiding over/underflows. We've used tricks such as the LogSumExp.

## Run

Make sure you've installed the dependencies of the project:

```
pip3 install -r requirements.txt
```

Run main.py with option help to see a list of options to execute:

python3 main.py help
