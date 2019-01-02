import numpy as np
import torch
import torch.utils.data
from scipy.special import gamma
from torchviz import make_dot

def load_data(path_to_file):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()
        split_lines = [line.split(',') for line in lines]
        x = list()
        y = list()
        for line in split_lines:
            x.append([float(elem) for elem in line[0:3]])
            y.append(float(line[3]))
        return torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y))

x_train, y_train = load_data('train.csv')
x_test, y_test = load_data('test.csv')

class NumChildrenModel(torch.jit.ScriptModule):
    def __init__(self):
        super(NumChildrenModel, self).__init__()
        self.fc1 = torch.nn.Linear(3, 6)
        self.fc2 = torch.nn.Linear(6, 3)
        self.fc3 = torch.nn.Linear(3, 1)
        self.dropout = torch.nn.Dropout(p=.5)

    @torch.jit.script_method
    def forward(self, x):
        c0 = ((x[:, 0] + 43034.) / (666.2160 + 43034.)) - .5
        c1 = ((x[:, 1] - 0.) / (21625.7 - 0.)) - .5
        c2 = ((x[:, 2] - 1.) / (81. - 1.)) - .5
        x = torch.stack((c0, c1, c2), dim=1) 
        x = self.dropout(torch.clamp(self.fc1(x), min=0))
        x = torch.clamp(self.fc2(x), min=0)
        x = torch.clamp(self.fc3(x), min=0)
        return x

model = NumChildrenModel().double()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=.8, weight_decay=.5)
bs = 32
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

for epoch in range(int(5)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        optimizer.zero_grad()
        y_hat = model(x)[:,0]
        loss = torch.sum((y_hat - y)**2) / bs
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 1000 == 999:    # print every 10 mini-batches
            print('[%d, %5d] avg loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
    for param in model.parameters():
        print(param.data)

test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=32)

test_loss = 0
ctr = 0
for i, data in enumerate(test_loader, 0):
    x, y = data
    y_hat = model(x)[:,0]
    if i == 0:
        print(y_hat)
        print(y)
    test_loss += torch.sum((y_hat - y)**2) / 32
    ctr += 1
test_loss /= float(ctr)

print(test_loss)

model.save("nc_model.pt")
