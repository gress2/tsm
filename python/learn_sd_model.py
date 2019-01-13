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
            x_line = [float(elem) for elem in line[0:5]]
            y_line = [float(elem) for elem in line[5:]]
            if (x_line[4] == 1):
                continue
            x.append(x_line)
            ys.append(torch.DoubleTensor(y_line))
        return torch.DoubleTensor(x), torch.nn.utils.rnn.pad_sequence(ys, batch_first=True)

x_train, y_train = load_data('train.csv')
x_test, y_test = load_data('test.csv')

class SDModel(torch.jit.ScriptModule):
    def __init__(self):
        super(SDModel, self).__init__()
        self.fc1 = torch.nn.Linear(5, 10)
        self.fc2 = torch.nn.Linear(10, 5)
        self.fc3 = torch.nn.Linear(5, 1)
        self.dropout = torch.nn.Dropout(p=.5)

    @torch.jit.script_method
    def forward(self, x):
        c0 = ((x[:, 0] + 43034.) / (558.1960 + 603.3710)) - .5
        c1 = ((x[:, 1] - 0.) / (21625.7 - 0.)) - .5
        c2 = ((x[:, 2] - 0.) / (81. - 0.)) - .5
        c3 = ((x[:, 3] - 1.) / (53. - 1.)) - .5
        c4 = x[:,4]
        x = torch.stack((c0, c1, c2, c3, c4), dim=1)
        x = self.dropout(torch.clamp(self.fc1(x), min=0))
        x = torch.clamp(self.fc2(x), min=0) 
        x = torch.clamp(self.fc3(x), min=1e-10)
        return x

model = SDModel().double()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=.7, weight_decay=2e-5)
bs = 32
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

for epoch in range(int(1)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        optimizer.zero_grad()
        epsilon = model(x)[:, 0]
        k = x[:,3]
        varphi2 = x[:,4]
        tau = y
        eta = tau * torch.sqrt(1 / k.view(len(k), 1))
        v = eta / torch.sqrt(1 - varphi2.view(len(varphi2), 1))
        sample = v / torch.sum(v, dim=1).view(len(x), 1)
        batch_log_likelihood = 0.0
        for j in range(len(x)):
            sample_actual_sz = k[j].type(torch.LongTensor)
            sample_j = sample[j][0 : sample_actual_sz]
            ll1 = torch.lgamma(k[j] * epsilon[j])
            ll2 = -1 * k[j] * torch.lgamma(epsilon[j])
            ll3 = (epsilon[j] - 1) * torch.sum(torch.log(sample_j + 1e-10))
            log_likelihood = ll1 + ll2 + ll3 
            batch_log_likelihood += log_likelihood

        loss = -batch_log_likelihood
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
    epsilon = model(x)[:, 0]
    k = x[:,3]
    varphi2 = x[:,4]
    tau = y
    eta = tau * torch.sqrt(1 / k.view(len(k), 1))
    v = eta / torch.sqrt(1 - varphi2.view(len(varphi2), 1))
    sample = v / torch.sum(v, dim=1).view(len(x), 1)
    batch_log_likelihood = 0.0
    for j in range(len(x)):
        sample_j_actual_sz = k[j].type(torch.LongTensor)
        sample_j = sample[j][0 : sample_j_actual_sz]
        ll1 = torch.lgamma(k[j] * epsilon[j])
        ll2 = -1 * k[j] * torch.lgamma(epsilon[j])
        ll3 = (epsilon[j] - 1) * torch.sum(torch.log(sample_j + 1e-10))
        log_likelihood = ll1 + ll2 + ll3
        batch_log_likelihood += log_likelihood
    loss = -batch_log_likelihood
    avg_loss = loss / len(x)
    test_loss += avg_loss
    ctr += 1
avg_test_loss = test_loss / float(ctr)
print(avg_test_loss)

model.save("sd_model.pt")

