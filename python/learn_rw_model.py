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
            x.append(int(line[0]))
            y.append(int(line[2]))
        print(np.corrcoef(x, y)[0, 1])
        exit()
        return torch.from_numpy(np.array(x)).double(), torch.from_numpy(np.array(y)).double()

x_train, y_train = load_data('train_dkd.csv')
x_test, y_test = load_data('test_dkd.csv')

bs = 128
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

class RandomWalkModel(torch.jit.ScriptModule):
    def __init__(self):
        super(RandomWalkModel, self).__init__()
        self.fc1 = torch.nn.Linear(1, 2)
        self.fc2 = torch.nn.Linear(2, 2)
        self.fc3 = torch.nn.Linear(2, 3)

    @torch.jit.script_method
    def forward(self, x):
        d = ((x - 0) / (81 - 0)) - .5
        d = d.view(len(d), 1)

        x = torch.clamp(self.fc1(d), min=0)
        x = torch.clamp(self.fc2(x), min=0)
        x = torch.clamp(self.fc3(x), min=1e-1)

        p = torch.clamp(x[:, 0], max=1-1e-1)
        lp = torch.clamp(x[:, 1], max=5)
        ln = torch.clamp(x[:, 2], max=5)

        return torch.stack((p, lp, ln), dim=1)

model = RandomWalkModel().double()
optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.1)
bs = 512
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

for epoch in range(int(10)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        optimizer.zero_grad()
        theta = model(x)

        pp = theta[:, 0]
        lp = theta[:, 1]
        ln = theta[:, 2]
        w = torch.ones(len(x)).double() * -1

        delta = y

        vareps = delta - w
        pos = vareps > 0
        neg = vareps < 0
        zero = vareps == 0

        loss = torch.ones(len(x)).double()

        loss[pos] = torch.log(pp[pos]) + (vareps[pos] * torch.log(lp[pos])) - lp[pos] - torch.lgamma(vareps[pos] + 1)
        loss[neg] = torch.log(1 - pp[neg]) + (-vareps[neg] * torch.log(lp[neg])) - lp[neg] - torch.lgamma(-vareps[neg] + 1)
        loss[zero] = torch.log((pp[zero] / torch.exp(lp[zero])) + ((1 - pp[zero]) / torch.exp(ln[zero])))

        loss = -torch.sum(loss)

        avg_loss = loss / len(x)
        avg_loss.backward()
        optimizer.step()

        running_loss += avg_loss.item()
        if i % 100 == 99:
            print('[%d, %5d] avg_loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=32)

test_loss = 0
ctr = 0
for i, data in enumerate(test_loader, 0):
    x, y = data
    optimizer.zero_grad()
    theta = model(x)
    pp = theta[:, 0]
    lp = theta[:, 1]
    ln = theta[:, 2]
    w = torch.ones(len(x)).double() * -1

    delta = y

    print(x)
    print(y)
    print(theta)

    vareps = delta - w
    pos = vareps > 0
    neg = vareps < 0
    zero = vareps == 0

    loss = torch.ones(len(x)).double()
    loss[pos] = torch.log(pp[pos]) + (vareps[pos] * torch.log(lp[pos])) - lp[pos] - torch.lgamma(vareps[pos] + 1)
    loss[neg] = torch.log(1 - pp[neg]) + (-vareps[neg] * torch.log(lp[neg])) - lp[neg] - torch.lgamma(-vareps[neg] + 1)
    loss[zero] = torch.log((pp[zero] / torch.exp(lp[zero])) + ((1 - pp[zero]) / torch.exp(ln[zero])))
    loss = -torch.sum(loss)
    avg_loss = loss / len(x)
    test_loss += avg_loss
    ctr += 1

avg_test_loss = test_loss / float(ctr)
print(avg_test_loss)

model.save("rw_model.pt")
