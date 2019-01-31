import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from scipy import stats
import matplotlib.pyplot as plt

def load_data(path_to_file):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()
        splits = [line.split(',') for line in lines]
        ds = list()
        ks = list()
        vs = list()
        for line in splits:
            d = int(line[0])
            k = int(line[1])
            v = float(line[2])
            if k == 1:
                continue
            ds.append(d)
            ks.append(k)
            vs.append(v)
        return ds, ks, vs

d, k, v = load_data('train_dkv.csv')
d = np.array(d)
k = np.array(k)
d = d.reshape(-1, 1)
k = k.reshape(-1, 1)
x = np.hstack((d, k))

x_tensor = torch.from_numpy(x).double()

model = (torch.jit.load('varphi_model.pt')).double()
y = model(x_tensor)

y_np = y.detach().numpy()

plt.scatter(k, y_np, s=1)
plt.show()
