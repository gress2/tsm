import numpy as np

k = 10
varphi2 = .02
p = 1. / k
epsilon = .8

c = epsilon * np.sqrt(p) / np.sqrt(1 - varphi2)
print(c)

assert c <= 1 / np.sqrt(k)

z = np.ones((k, k)) * c
for i in range(k):
    z[i][i] = np.sqrt(1 - (k - 1) * c**2)
    assert np.isclose(np.linalg.norm(z[i]), 1)

x = np.random.uniform(0, 1, k + 1)
x[0] = 0
x[k] = 1
x = np.sort(x)

lambda_ = np.zeros(k)
for i in range(1, k + 1):
    lambda_[i - 1] = x[i] - x[i - 1]

q = list()
for j in range(k):
    s = 0
    for i in range(k):
        s += lambda_[i] * z[j][i]
    q.append(s)

x = np.array(q)
x = x / np.linalg.norm(x)
print(x)
assert len(x[np.where(x < c)]) == 0

