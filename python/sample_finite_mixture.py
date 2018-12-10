import numpy as np

def get_Fsj(p, s, j):
    if s == 1:
        if j == 1:
            return -np.sqrt(p)
        if j == 2:
            return np.sqrt(p)
        else:
            return 0
    else:
        if j <= s:
            return -np.sqrt(p*p) / np.sqrt(s * p)
        if j == s + 1:
            return np.sqrt(s * p)
        else:
            return 0

def get_Fs(p, s, k):
    Fs = list()
    for i in range(1, k + 1):
        vec_elem = get_Fsj(p, s, i)
        Fs.append(vec_elem)
    return np.array(Fs)

def get_orthonormal_basis(k):
    p = 1 / float(k)
    F = list()
    for i in range(1, k):
        orth_vec = get_Fs(p, i, k)
        F.append(orth_vec / np.linalg.norm(orth_vec))
    return np.array(F)

def get_varpi(k):
    varpi = np.hstack((np.random.uniform(0, np.pi, k-3), np.random.uniform(0, 2 * np.pi)))
    return varpi

def get_varphi2():
    return np.random.beta(2, 2)

def get_gamma(k, varphi2):
    varpi = get_varpi(k)
    on_basis = get_orthonormal_basis(k)
    gamma = np.zeros(k)
    for i in range(k - 1):
        mul = np.sqrt(varphi2)
        for j in range(min(i, k - 3)):
            mul *= np.sin(varpi[j])
        if (i < k - 2):
            mul *= np.cos(varpi[i])
        else:
            mul *= np.sin(varpi[i - 1])
        gamma += on_basis[i] * mul
    assert np.isclose(np.sum(gamma), 0)
    assert np.isclose(np.sum(gamma**2), varphi2)
    return gamma

def get_eta(k, varphi2):
    xi = np.random.uniform(0, np.pi / 2, k - 1)
    eta = np.ones(k) * np.sqrt(1 - varphi2)
    for i in range(k):
        prod = 1
        if i == 0:
            prod *= np.cos(xi[i])
        if i > 0 and i < k - 1:
            for j in range(0, i):
                prod *= np.sin(xi[j])
            prod *= np.cos(xi[i])
        if i == k - 1:
            for j in range(0, i):
                prod *= np.sin(xi[j])
        eta[i] *= prod 
    assert np.isclose(np.sum(eta**2), 1 - varphi2)
    return eta

def get_mixture(mean, sd, k):
    p = 1. / k
    varphi2 = get_varphi2()
    gamma = get_gamma(k, varphi2)
    eta = get_eta(k, varphi2)
    alpha = gamma / np.sqrt(p)
    tau = eta / np.sqrt(p)
    assert np.isclose(np.sum(alpha), 0)
    assert np.isclose(p * np.sum(tau**2 + alpha**2), 1)
    mu = alpha * sd + mean
    sigma = sd * tau
    assert np.isclose(np.sum(p * mu), mean)
    assert np.isclose(np.sum(p * (mu**2 + sigma**2)) - mean**2, sd**2)
    return (mu, sigma)

mean = 400
sd = 50
k = 10

child_means, child_sds = get_mixture(mean, sd, k)
print(child_means)
print(child_sds)
