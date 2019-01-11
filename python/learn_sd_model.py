import numpy as np
import torch
import torch.utils.data
from scipy.special import gamma

def load_data(path_to_file):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()
        split_lines = [line.split(',') for line in lines]
        x = list()
        ys = list()
        for line in split_lines:
            x.append([float(elem) for elem in line[0:5]])
            ys.append(torch.DoubleTensor([float(elem) for elem in line[5:]]))
        return torch.DoubleTensor(x), torch.nn.utils.rnn.pad_sequence(ys, batch_first=True)

x_train, y_train = load_data('train.csv')
x_test, y_test = load_data('test.csv')

class SDModel(torch.jit.ScriptModule):
    def __init__(self):
        super(SDModel, self).__init__()
        self.fc1 = torch.nn.Linear(5, 5)
        self.fc2 = torch.nn.Linear(5, 5)
        self.fc3 = torch.nn.Linear(5, 2)
        self.dropout = torch.nn.Dropout(p=.5)

    @torch.jit.script_method
    def forward(self, x):
        c0 = ((x[:, 0] + 43034.) / (558.1960 + 603.3710)) - .5
        c1 = ((x[:, 1] - 0.) / (21625.7 - 0.)) - .5
        c2 = ((x[:, 2] - 0.) / (81. - 0.)) - .5
        c3 = ((x[:, 3] - 1.) / (53. - 1.)) - .5
        c4 = x[:,4]
        eps_ub = torch.sqrt(torch.clamp(1. - x[:,4], min=1e-5))
        x = torch.stack((c0, c1, c2, c3, c4), dim=1)

        x = self.dropout(torch.clamp(self.fc1(x), min=0))
        x = self.fc2(x)
        x[:, 0] = torch.min(torch.clamp(x[:, 0], min=1e-5), eps_ub - 1e-5)
        x[:, 1] = torch.clamp(x[:, 1], min=1e-5) 
        return x

model = SDModel().double()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=.9, weight_decay=2e-5)
bs = 32
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

for epoch in range(int(5)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        optimizer.zero_grad()
        fx = model(x)
        k = x[:,3]
        varphi2 = x[:,4]
        epsilon = fx[:,0]
        omega = fx[:,1]
        c = torch.min(1. / torch.sqrt(k - 1.), epsilon / torch.sqrt(k * (1. - varphi2 + 1e-5)))
        t = torch.sqrt(torch.clamp(1. - ((k - 1.) * (c**2.)), min=0.))
        tau = y
        eta = tau / (k.view(len(k), 1))
        v = eta / torch.sqrt(torch.clamp(1. - varphi2.view(len(varphi2), 1), min=0.))
        loss = 0.0
        for i in range(bs):
            ki = k[i]
            ki_long = ki.type(torch.LongTensor)
            Z = torch.ones((ki_long, ki_long), dtype=torch.double) * c[i] 
            Z[torch.eye(ki_long).byte()] = t[i]
            Zinv = Z.inverse()
            lambda_prime = torch.matmul(Zinv, v[i][0:ki_long])
            print('varphi2')
            print(varphi2[i])
            print('c')
            print(c[i])
            print('k')
            print(ki_long)
            print('t')
            print(t[i])
            print('epsilon')
            print(epsilon[i])
            print('omega')
            print(omega[i])
            print('Zinv')
            print(Zinv)
            lambda_ = lambda_prime / torch.sum(lambda_prime) 
            l1 = -torch.lgamma(ki_long * omega[i])
            l2 = ki_long * torch.lgamma(omega[i])
            l3 = -(omega[i] - 1.) * torch.sum(torch.log(lambda_))
            cur_loss = l1 + l2 + l3
            print(lambda_)
            print(cur_loss)
            exit()
            cur_loss = l1 + l2 + l3
            loss += cur_loss
        print(loss)
        exit()
