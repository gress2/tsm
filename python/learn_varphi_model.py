import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from scipy import stats

def load_data(path_to_file):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()
        splits = [line.split(',') for line in lines]
        x = list()
        y = list()
        for line in splits:
            d = int(line[0])
            k = int(line[1])
            v = float(line[2])
            if k == 1:
                continue
            x.append([d, k])
            y.append(v)
        
        x_tensor = torch.from_numpy(np.array(x)).double()
        y_tensor = torch.from_numpy(np.array(y)).double()
        return x_tensor, y_tensor

x_train, y_train = load_data('train_dkv.csv')
x_test, y_test = load_data('test_dkv.csv')

bs = 128
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

class VarphiModel(torch.jit.ScriptModule):
    def __init__(self):
        super(VarphiModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=.5)

    @torch.jit.script_method
    def forward(self, x):
        d = x[:, 0]
        d = (d / 81) - .5
        k = x[:, 1]
        k = ((k - 2) / 51) - .5
        x = torch.stack((d, k), dim=1)
        x = self.dropout(torch.clamp(self.fc1(x), min=0))
        x = torch.clamp(torch.clamp(self.fc2(x), min=0.01), max=1)
        return x

model = VarphiModel().double()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=.01)

for epoch in range(int(100)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        y = y.view(len(y), 1)
        optimizer.zero_grad
        y_pred = model(x)

        loss = torch.mean((y_pred - y)**2)
        if (torch.isnan(loss)):
            print(x)
            print(y_pred)
            print(y)
            print((y_pred - y)**2)
            exit()

        loss.backward()
        optimizer.step()

        running_loss += loss
        if i % 1000 == 999:
            print('[%d, %5d] avg_loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=bs)

test_loss = 0
ctr = 0
for i, data in enumerate(test_loader, 0):
    x, y = data
    y = y.view(len(y), 1)
    y_pred = model(x)

    if i == 0:
        print(y)
        print(y_pred)

    loss = torch.mean((y_pred - y)**2)
    test_loss += loss
    ctr += 1

avg_test_loss = test_loss / float(ctr)
print(avg_test_loss)

model.save('varphi_model.pt')