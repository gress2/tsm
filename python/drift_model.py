import matplotlib.pyplot as plt
import numpy as np

alpha = -1
ppos = .35
l = 1

def get_delta(p):
    is_positive = np.random.random() <= p
    delta = np.random.poisson(l)
    return delta if is_positive else -delta

td = list()
k = list()
d = list()

for j in range(10000):
    y = 50
    k.append(y)
    d.append(0)
    i = 1
    while True:
        y = alpha + y + get_delta(ppos)
        d.append(i)
        k.append(y)
        if (y <= 0):
            td.append(i)
            break
        i += 1

plt.hist(td, bins=50)
plt.xlabel('Terminal depth')
plt.show()

plt.scatter(k, d)
plt.xlabel('Num children')
plt.ylabel('Depth')
plt.show()
