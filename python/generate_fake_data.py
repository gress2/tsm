import numpy as np

N = 100000

mu = np.random.normal(-200, 500, N)
sigma = np.random.lognormal(3, 2, N)
k = np.random.poisson(25, N)
d = np.random.poisson(25, N)

a = 1/3. * np.sqrt(np.abs(mu + sigma + k + d))
b = 1/3. * np.sqrt(np.abs(mu - sigma + k * d))

y = list()
for i in range(N):
    y.append(np.random.beta(a[i], b[i]))
y = np.array(y)


with open('fake_data.csv', 'w') as data_f:
    for i in range(N):
        data_f.write('%f, %f, %f, %f, %f\n' % (mu[i], sigma[i], k[i], d[i], y[i])) 
