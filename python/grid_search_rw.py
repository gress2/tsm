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

bs = 256
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

best_total_ll = -1e30
best_p = -1
best_l = -1
best_a = -1

for p in np.linspace(.01, .99, 20):
    for l in np.linspace(.5, 2.5, 10):
        for a in range(-5, 0):
            total_ll = 0
            for i, data in enumerate(train_loader, 0):
                _, y = data
                delta = y
                delta_pos = delta[delta > 0] 
                delta_neg = torch.abs(delta[delta < 0])
                delta_zero = delta[delta == 0]

                pos_p = torch.ones(len(delta_pos)).double() * p
                pos_l = torch.ones(len(delta_pos)).double() * l
                neg_p = torch.ones(len(delta_neg)).double() * p
                neg_l = torch.ones(len(delta_neg)).double() * p
                zero_l = torch.ones(len(delta_zero)).double() * l

                total_ll += torch.sum(torch.log(pos_p) + (delta_pos - a) * torch.log(pos_l) - l - torch.lgamma(delta_pos + 1 - a)) 
                total_ll += torch.sum(torch.log(1 - neg_p) + (delta_neg - a) * torch.log(neg_l) - l - torch.lgamma(delta_neg + 1 - a)) 
                total_ll += torch.sum((delta_zero - a) * torch.log(zero_l) - l - torch.lgamma(delta_zero + 1 - a)) 

            print('LL: {}, p: {}, l: {}, a: {}'.format(total_ll.item(), p, l, a))

            if total_ll.item() > best_total_ll:
                best_total_ll = total_ll.item()
                best_p = p
                best_l = l
                best_a = a

print('Best LL: {}, p: {}, l: {}, a: {}'.format(best_total_ll, best_p, best_l, best_a))
