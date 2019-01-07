import numpy as np

def sample_lambda(k):
    x = np.random.uniform(0, 1, k + 1)
    x[0] = 0
    x[k] = 1
    x = np.sort(x)

    lambda_ = np.zeros(k)
    for i in range(1, k + 1):
        lambda_[i - 1] = x[i] - x[i - 1]
    assert(np.isclose(np.sum(lambda_), 1))
    return lambda_

def constrained_sample(k, c, lambda_):
    zis = np.ones((k, k)) * c
    np.fill_diagonal(zis, np.sqrt(1 - (k - 1) * c**2))
    x = np.zeros(k)
    for i in range(k):
        x += lambda_[i] * zis[i]
    return x

k = 3
c = .4

lambda_ = sample_lambda(k)
print(constrained_sample(k, c, lambda_))


