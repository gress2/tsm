import numpy as np
import torch
import torch.utils.data

def load_data(file_path):
    with open(file_path, 'r') as data:
        lines = data.readlines()
        split_lines = [line.split(',') for line in lines]
        x = np.array([[float(elem) for elem in line[0:3]] 
            for line in split_lines])
        y = np.array([float(line[3]) for line in split_lines])
        return torch.from_numpy(x), torch.from_numpy(y)

x_train, y_train = load_data('train.csv')
x_test, y_test = load_data('test.csv')

class NumChildrenModel(torch.jit.ScriptModule):
    def __init__(self):
        super(NumChildrenModel, self).__init__()
        self.fc1 = torch.nn.Linear(3, 8)
        self.fc2 = torch.nn.Linear(8, 4)
        self.fc3 = torch.nn.Linear(4, 2)
        self.dropout = torch.nn.Dropout(p=.5)

    @torch.jit.script_method
    def forward(self, x):
        x = self.dropout(torch.clamp(self.fc1(x), min=0))
        x = self.dropout(torch.clamp(self.fc2(x), min=0))
        x = self.fc3(x)
        r = torch.clamp(torch.round(x[:,0]), min=1e-3)
        p = torch.clamp(torch.clamp(x[:,1], min=1e-3), max=1.0-1e-3)
        x = torch.stack((r,p), dim=1)
        return x

model = NumChildrenModel().double()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=.1)

train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

for epoch in range(int(5)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        optimizer.zero_grad()
        g = model(x)
        l1 = torch.lgamma(y + g[:,0])
        l2 = torch.lgamma(y + 1)
        l3 = torch.lgamma(g[:,0])
        l4 = g[:,0] * torch.log(1 - g[:,1])
        l5 = y * torch.log(g[:,1])

        log_likelihood = l1 - l2 - l3 + l4 + l5
        loss = -torch.sum(log_likelihood)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999: 
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=32)

test_loss = 0.
for _, data in enumerate(test_loader, 0):
    x, y = data
    g = model(x)
    l1 = torch.lgamma(y + g[:,0])
    l2 = torch.lgamma(y + 1)
    l3 = torch.lgamma(g[:,0])
    l4 = g[:,0] * torch.log(1 - g[:,1])
    l5 = y * torch.log(g[:,1])

    log_likelihood = l1 - l2 - l3 + l4 + l5
    loss = -torch.sum(log_likelihood)
    test_loss += loss / 32.

print('Test loss: ' + str(test_loss.item()))

model.save("nc_model.pt")

