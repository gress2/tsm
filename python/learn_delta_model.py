import numpy as np
import torch
import torch.utils.data

def load_data(path_to_file):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()
        split_lines = [line.split(',') for line in lines]
        x = list()
        y = list()
        for line in split_lines:
            x.append([int(line[0]), int(line[1])]) 
            y.append(int(line[3]))
        return torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y))

x_train, y_train = load_data('train_dkd.csv')
x_test, y_test = load_data('test_dkd.csv')

class DeltaModel(torch.jit.ScriptModule):
    def __init__(self):
        super(DeltaModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 4)
        self.fc2 = torch.nn.Linear(4, 4)
        self.fc3 = torch.nn.Linear(4, 3)
        self.dropout = torch.nn.Dropout(p=.5)

    @torch.jit.script_method
    def forward(self, x):
        d = ((x[:, 0] - 1.) / (81. - 1.)) - .5
        k = ((x[:, 1] - 0.) / (52. - 0.)) - .5 
        x = torch.stack((d, k), dim=1)
        x = self.dropout(torch.clamp(self.fc1(x), min=0))
        x = torch.clamp(self.fc2(x), min=0)
        x = self.fc3(x)
        x[:, 0] = torch.round(x[:, 0])
        x[:, 1] = torch.clamp(torch.clamp(x[:, 1], min=0), max=1)
        x[:, 2] = torch.clamp(torch.clamp(x[:, 2], min=0), max=1)
        return x

f = DeltaModel().double()
optimizer = torch.optim.SGD(f.parameters(), lr=1e-3, momentum=.8, weight_decay=.5)
bs = 32
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        optimizer.zero_grad()
        g = f(x)
        n_binom = g[:,0]
        p_binom = g[:,1]
        p_bernoulli = g[:,2]

        delta = y
        


