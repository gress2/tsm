import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

def load_data(path_to_file):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()
        split_lines = [line.split(',') for line in lines]
        x = list()
        y = list()
        for line in split_lines:
            x.append([int(line[0]), int(line[1]), int(int(line[2]) > 0)])
            y.append((int(line[2])))
        return torch.from_numpy(np.array(x)).double(), torch.from_numpy(np.array(y)).double()

x_train, y_train = load_data('train_dkd.csv')
x_test, y_test = load_data('test_dkd.csv')

bs = 128
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

best_total_ll = -1e30
best_p = -1
best_lp = -1
best_ln = -1
best_w = -1

for p in np.linspace(.05, .95, 9):
    for lp in np.linspace(.2, 2.5, 8):
        for ln in np.linspace(.2, 2.5, 8):
            for w in range(-1, 0):
                total_ll = 0
                for i, data in enumerate(train_loader, 0):
                    _, y = data
                    delta = y

                    vareps = delta - w
                    vareps_p = vareps[vareps > 0]
                    vareps_n = vareps[vareps < 0]
                    vareps_z = vareps[vareps == 0]

                    lambda_pos = torch.ones(len(vareps_p)).double() * lp
                    lambda_neg = torch.ones(len(vareps_n)).double() * ln
                    p_p = torch.ones(len(vareps_p)).double() * p
                    p_n = torch.ones(len(vareps_n)).double() * (1 - p)

                    p_p_z = torch.ones(len(vareps_z)).double() * p
                    p_n_z = torch.ones(len(vareps_z)).double() * (1 - p)
                    l_p_z = torch.ones(len(vareps_z)).double() * lp
                    l_n_z = torch.ones(len(vareps_z)).double() * ln

                    total_ll += torch.sum(torch.log(p_p) + (vareps_p * torch.log(lambda_pos)) - lambda_pos - torch.lgamma(vareps_p + 1))
                    total_ll += torch.sum(torch.log(p_n) + (-vareps_n * torch.log(lambda_neg)) - lambda_neg - torch.lgamma(-vareps_n + 1))
                    total_ll += torch.sum(torch.log((p_p_z / torch.exp(l_p_z)) + (p_n_z / torch.exp(l_n_z))))

                print('LL: {}, pp: {}, lp: {}, ln: {}, w: {}'.format(total_ll.item(), p, lp, ln, w))

                if total_ll.item() > best_total_ll:
                    best_total_ll = total_ll.item()
                    best_p = p
                    best_lp = lp
                    best_ln = ln
                    best_w = w

print('Best LL: {}, pp: {}, lp: {}, ln: {}, w: {}'.format(best_total_ll, best_p, best_lp, best_ln, best_w))
