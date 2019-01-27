import numpy as np
import matplotlib.pyplot as plt

mu_gen = list()
sd_gen = list()
d_gen = list()
k_gen = list()
with open('rw_stats') as stats_f:
    lines = stats_f.readlines()
    splits = [line.split(',') for line in lines]
    for split in splits:
        mu_gen.append(float(split[0]))
        sd_gen.append(float(split[1]))
        d_gen.append(int(split[2]))
        k_gen.append(int(split[3]))

mu_gen = np.array(mu_gen)
sd_gen = np.array(sd_gen)
d_gen = np.array(d_gen)
k_gen = np.array(k_gen)

mu_sg = list()
sd_sg = list()
d_sg = list()
k_sg = list()
with open('train.csv') as stats_f:
    lines = stats_f.readlines()
    splits = [line.split(',') for line in lines]
    for split in splits:
        mu_sg.append(float(split[0]))
        sd_sg.append(float(split[1]))
        d_sg.append(int(split[2]))
        k_sg.append(int(split[3]))

mu_sg = np.array(mu_sg)
sd_sg = np.array(sd_sg)
d_sg = np.array(d_sg)
k_sg = np.array(k_sg)

plt.subplot(3, 2, 1)
plt.plot(mu_sg, d_sg, '.r')
plt.xlabel('mu')
plt.ylabel('d')
plt.title('mu vs d in samegame')

plt.subplot(3, 2, 2)
plt.plot(mu_gen, d_gen, '.r')
plt.xlabel('mu')
plt.ylabel('d')
plt.title('mu vs d in generic game')

plt.subplot(3, 2, 3)
plt.plot(sd_sg, d_sg, '.r')
plt.xlabel('sd')
plt.ylabel('d')
plt.title('stddev vs d in samegame')

plt.subplot(3, 2, 4)
plt.plot(sd_gen, d_gen, '.r')
plt.xlabel('sd')
plt.ylabel('d')
plt.title('stddev vs d in generic game')

plt.subplot(3, 2, 5)
plt.plot(k_sg, d_sg, '.r')
plt.xlabel('k')
plt.ylabel('d')
plt.title('k vs d in samegame')

plt.subplot(3, 2, 6)
plt.plot(k_gen, d_gen, '.r')
plt.xlabel('k')
plt.ylabel('d')
plt.title('k vs d in generic game')

plt.tight_layout()
plt.show()

plt.subplot(2, 2, 1)
plt.hist(k_sg, 100)
plt.title('histogram of k in samegame')

plt.subplot(2, 2, 2)
plt.hist(k_gen, 100)
plt.title('histogram of k in generic game')

plt.subplot(2, 2, 3)
plt.hist(d_sg, 200)
plt.title('histogram of d in samegame')

plt.subplot(2, 2, 4)
plt.hist(d_gen, 200)
plt.title('histogram of d in generic game')

plt.tight_layout()
plt.show()
