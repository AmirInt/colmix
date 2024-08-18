import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import src.em as em
import src.common as common


X = np.array(
    [[0.85794562, 0.84725174],
    [0.6235637,  0.38438171],
    [0.29753461, 0.05671298],
    [0.,         0.47766512],
    [0.,         0.        ],
    [0.3927848,  0.        ],
    [0.,         0.64817187],
    [0.36824154, 0.        ],
    [0.,         0.87008726],
    [0.47360805, 0.        ],
    [0.,         0.        ],
    [0.,         0.        ],
    [0.53737323, 0.75861562],
    [0.10590761, 0.        ],
    [0.18633234, 0.        ]]
)

# Testing em methods

K = 6
n, d = X.shape
seed = 0

var = np.array([0.16865269, 0.14023295, 0.1637321,  0.3077471,  0.13718238, 0.14220473])
p = np.array([0.1680912,  0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722])
mu = np.array(
    [[0.6235637,  0.38438171],
    [0.3927848,  0.        ],
    [0.,         0.        ],
    [0.,         0.87008726],
    [0.36824154, 0.        ],
    [0.10590761, 0.        ]]
)
mixture = common.GaussianMixture(p=p, var=var, mu=mu)

X_pred = em.fill_matrix(X, mixture)

print(X_pred)

print("Success: em tests passed!")
