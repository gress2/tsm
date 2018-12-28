import numpy as np
import torch
import torch.utils.data
from scipy.special import gamma

with open("mixing_data.csv") as mix_data:
    lines = mix_data.readlines()
    split_lines = [line.split(',') for line in lines]
    clean_lines = []
    for line in split_lines:
        varphi2 = float(line[4])
        if not np.isclose(varphi2, 1) and not np.isclose(varphi2, 0):
            clean_lines.append(line)
    x = np.array([[float(elem) for elem in line[0:4]] for line in clean_lines])
    y = np.array([float(line[4]) for line in clean_lines])

class DispersionModel(torch.jit.ScriptModule):
    def __init__(self):
        super(DispersionModel, self).__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 4)
        self.fc3 = torch.nn.Linear(4, 2)

    @torch.jit.script_method
    def forward(self, x):
        x = torch.clamp(self.fc1(x), min=0) 
        x = torch.nn.functional.dropout(x, p=.5)
        x = torch.clamp(self.fc2(x), min=0)
        x = torch.clamp(self.fc3(x), min=.0001) 
        return x


x = torch.from_numpy(x)
y = torch.from_numpy(y)
model = DispersionModel().double()

#device = torch.device("cuda")
#x = torch.from_numpy(x)
#y = torch.from_numpy(y)
#model = DispersionModel().double().cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-10, momentum=0.7)

train = torch.utils.data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

for epoch in range(int(1e4)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        x, y = data
        optimizer.zero_grad()
        g = model(x)

        l1 = torch.lgamma(torch.sum(g, dim=1))
        l2 = -1 * torch.lgamma(g[:,0])
        l3 = -1 * torch.lgamma(g[:,1])
        l4 = torch.log(y) * (g[:,0] - 1)  
        l5 = torch.log(1 - y) * (g[:,1] - 1)

        log_likelihood = l1 + l2 + l3 + l4 + l5 
        loss = -torch.sum(log_likelihood)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

model.save("dispersion_model.pt")
