import numpy as np
import torch
import torch.utils.data
from scipy.special import gamma

def load_data(path_to_data):
    with open(path_to_data, 'r') as data:
        lines = data.readlines()
        split_lines = [line.split(',') for line in lines]
        clean_lines = []
        for line in split_lines:
            varphi2 = float(line[4])
            if not np.isclose(varphi2, 1) and not np.isclose(varphi2, 0):
                clean_lines.append(line)
        x = np.array([[float(elem) for elem in line[0:4]] for line in clean_lines])
        y = np.array([float(line[4]) for line in clean_lines])
        return torch.from_numpy(x), torch.from_numpy(y)

x_train, y_train = load_data('train.csv')
x_test, y_test = load_data('test.csv')

class DispersionModel(torch.jit.ScriptModule):
    def __init__(self):
        super(DispersionModel, self).__init__()
        self.fc1 = torch.nn.Linear(4, 6)
        self.fc2 = torch.nn.Linear(6, 4)
        self.fc3 = torch.nn.Linear(4, 2)

    @torch.jit.script_method
    def forward(self, x):
        c0 = ((x[:,0] + 23118.9) / (666.2160 + 23118.9)) - .5
        c1 = ((x[:,1] - .2560) / (21625.7 - .2560)) - .5
        c2 = ((x[:,2] - 1.) / (77. - 1.)) - .5
        c3 = ((x[:,3] - 2.) / (52. - 2.)) - .5
        x = torch.stack((c0, c1, c2, c3), dim=1)
        x = torch.clamp(self.fc1(x), min=0) 
        x = torch.nn.functional.dropout(x, p=.5)
        x = torch.clamp(self.fc2(x), min=0)
        x = torch.clamp(self.fc3(x), min=.0001) 
        return x

model = DispersionModel().double()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.8, weight_decay=.5)

train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

for epoch in range(int(5)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        optimizer.zero_grad()
        g = model(x)
        a = g[:,0]
        b = g[:,1]

        l1 = torch.lgamma(a + b)
        l2 = -torch.lgamma(a)
        l3 = -torch.lgamma(b)
        l4 = (a - 1) * torch.log(y)
        l5 = (b - 1) * torch.log(1 - y)

        log_likelihood = l1 + l2 + l3 + l4 + l5 
        loss = -torch.sum(log_likelihood) / 32.
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # print every 10 mini-batches
            print('[%d, %5d] avg_loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=100)

test_loss = 0
ctr = 0
for i, data in enumerate(test_loader, 0):
    x, y = data
    g = model(x)
    if i == 0:
        print(g)
        print(y)
    l1 = torch.lgamma(torch.sum(g, dim=1))
    l2 = -1 * torch.lgamma(g[:,0])
    l3 = -1 * torch.lgamma(g[:,1])
    l4 = torch.log(y) * (g[:,0] - 1)  
    l5 = torch.log(1 - y) * (g[:,1] - 1)

    log_likelihood = l1 + l2 + l3 + l4 + l5 
    test_loss += -torch.sum(log_likelihood).item()
    ctr += 1

test_loss /= (100 * float(ctr)) 

print(test_loss)

model.save("dispersion_model.pt")
