#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from scipy import stats

parser = argparse.ArgumentParser('Train varphi neural net with a given set of hyperparameters')
parser.add_argument('-l', type=float, default=1e-3, help='learning rate')
parser.add_argument('-w', type=float, default=.05, help='weight decay')
parser.add_argument('-b', type=int, default=32, help='batch size')
parser.add_argument('-e', type=int, default=50, help='number of epochs')

args = parser.parse_args()
learning_rate = args.l
weight_decay = args.w
batch_size = args.b
epochs = args.e
arch = '2-4-4-4-1'

def load_data(path_to_file):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()
        splits = [line.split(',') for line in lines]
        x = list()
        y = list()
        for line in splits:
            d = float(line[0])
            k = float(line[1])
            v = float(line[2])
            if k == 1:
                continue
            x.append([d,k])
            y.append(v)
        
        x_tensor = torch.from_numpy(np.array(x)).double()
        y_tensor = torch.from_numpy(np.array(y)).double()
        return x_tensor, y_tensor

x_train, y_train = load_data('train.dkv.csv')
x_test, y_test = load_data('test.dkv.csv')

train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

class VarphiModel(torch.jit.ScriptModule):
    def __init__(self):
        super(VarphiModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 2)
        self.fc3 = torch.nn.Linear(2, 1)
        self.dropout = torch.nn.Dropout(p=.2)

    @torch.jit.script_method
    def forward(self, x):
        d = x[:, 0] 
        d = (d / 81) - .5
        k = x[:, 1]
        k = ((k - 2) / 51) - .5
        x = torch.stack((d, k), dim=1)
        x = self.dropout(torch.clamp(self.fc1(x), min=0))
        x = self.dropout(torch.clamp(self.fc2(x), min=0))
        x = torch.clamp(torch.clamp(self.fc3(x), min=1e-2), max=1)
        return x

model = VarphiModel().double()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

reporting_freq = 200

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        y = y.view(len(y), 1)
        optimizer.zero_grad
        y_pred = model(x)

        diff = y_pred - y 
        loss = torch.mean(diff**2)
     
        loss.backward()
        optimizer.step()

        running_loss += loss

        if i % reporting_freq == reporting_freq - 1:
            print('[%d, %5d] avg_loss: %.3f' % (epoch + 1, i + 1, running_loss / reporting_freq))
            running_loss = 0.0

test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

test_loss = 0
ctr = 0
for i, data in enumerate(test_loader, 0):
    x, y = data
    y = y.view(len(y), 1)
    y_pred = model(x)

    if i == 0:
        print(y)
        print(y_pred)
    diff = y_pred - y 

    loss = torch.mean(diff**2)

    test_loss += loss
    ctr += 1

avg_test_loss = test_loss / float(ctr)
print(avg_test_loss)

model.save('varphi_model.pt')
model.save('pts/varphi_{}_l{}_w{}_b{}_e{}.pt'.format(arch, learning_rate, weight_decay, batch_size, epochs))
