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

def get_orthonormal_basis(p, k):
    F = list()
    for i in range(1, k):
        orth_vec = get_Fs(p, i, k)
        F.append(orth_vec / np.linalg.norm(orth_vec))
    return np.array(F)

basis = get_orthonormal_basis(1/4., 4)
print(basis)
basis = get_orthonormal_basis(1/7., 7)
print(basis)
