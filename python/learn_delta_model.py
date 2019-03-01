import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

def load_data(path_to_file):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()
        splits = [line.split(',') for line in lines]
        x = list()
        y = list()
        for line in splits:
            d = float(line[0])
            k = float(line[1])
            delta = float(line[2])
            x.append([d, k])
            y.append(delta)
        x_tensor = torch.from_numpy(np.array(x)).double()
        y_tensor = torch.from_numpy(np.array(y)).double()
        return x_tensor, y_tensor

x_train, y_train = load_data('train.dkd.csv')
x_test, y_test = load_data('test.dkd.csv')

class DeltaModel(torch.jit.ScriptModule):
    def __init__(self):
        super(DeltaModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 8)
        self.fc2 = torch.nn.Linear(8, 8)
        self.fc3 = torch.nn.Linear(8, 3)
        self.dropout = torch.nn.Dropout(p=.2)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(8)

    @torch.jit.script_method
    def forward(self, x):
        d = x[:, 0] / 81.
        k = (x[:, 1] - 2) / 51.
        x = torch.stack((d, k), dim=1)
        x = self.bn1(torch.clamp(self.fc1(x), min=0))
        x = self.bn2(torch.clamp(self.fc2(x), min=0))
        x = torch.clamp(self.fc3(x), min=0)
        lambda_pos = torch.clamp(x[:,0], min=.01)
        lambda_neg = torch.clamp(x[:,1], min=.01)
        p = torch.clamp(torch.clamp(x[:,2], max=.99), min=.01)
        return torch.stack((lambda_pos, lambda_neg, p), dim=1)

bs = 128
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

model = DeltaModel().double()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=.5)
num_epoch = 5 

for epoch in range(num_epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad
        x, y = data
        theta = model(x)
        lambda_p = theta[:, 0]
        lambda_n = theta[:, 1]
        p = theta[:, 2]
        
        pos_log_likeli = torch.log(p[y > 0]) + (y[y > 0] * torch.log(lambda_p[y > 0])) - lambda_p[y > 0] - torch.lgamma(torch.abs(y[y > 0]) + 1)
        neg_log_likeli = torch.log(1 - p[y < 0]) - (y[y < 0] * torch.log(lambda_n[y < 0])) - lambda_n[y < 0] - torch.lgamma(torch.abs(y[y < 0]) + 1)
        zero_log_likeli = torch.log((p[y == 0] * torch.exp(-lambda_p[y == 0])) + ((1 - y[y == 0]) * torch.exp(-lambda_n[y == 0])))  
    
        combined = torch.cat((pos_log_likeli, neg_log_likeli, zero_log_likeli))
        loss = -1 * torch.sum(combined)

        loss.backward()
        optimizer.step()

        running_loss += loss
        if i % 100 == 99:
            print('[%d, %5d] avg_loss: %.3f' % (epoch + 1, i + 1, running_loss / (100 * bs)))
            running_loss = 0.0

test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=bs)

test_loss = 0
ctr = 0

for i, data in enumerate(test_loader, 0):
    x, y = data
    theta = model(x)
    lambda_p = theta[:, 0]
    lambda_n = theta[:, 1]
    p = theta[:, 2]

    pos_log_likeli = torch.log(p[y > 0]) + (y[y > 0] * torch.log(lambda_p[y > 0])) - lambda_p[y > 0] - torch.lgamma(torch.abs(y[y > 0]) + 1)
    neg_log_likeli = torch.log(1 - p[y < 0]) - (y[y < 0] * torch.log(lambda_n[y < 0])) - lambda_n[y < 0] - torch.lgamma(torch.abs(y[y < 0]) + 1)
    zero_log_likeli = torch.log((p[y == 0] * torch.exp(-lambda_p[y == 0])) + ((1 - y[y == 0]) * torch.exp(-lambda_n[y == 0])))  
    combined = torch.cat((pos_log_likeli, neg_log_likeli, zero_log_likeli))
    loss = -1 * torch.sum(combined)

    if i == 0:
        print(y)
        print(theta)
    
    test_loss += loss
    ctr += 1
avg_test_loss = test_loss / float(ctr * bs)
print(avg_test_loss)
model.save('delta_model.pt')
