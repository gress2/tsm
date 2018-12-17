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
        self.fc1 = torch.nn.Linear(4, 100)
        self.fc2 = torch.nn.Linear(100, 2)

    @torch.jit.script_method
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x)) 
        x = torch.nn.functional.relu(self.fc2(x))
        return x

x = torch.from_numpy(x)
y = torch.from_numpy(y)

device = torch.device("cpu")

model = DispersionModel()
model = model.double()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)

train = torch.utils.data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, outputs = data
        optimizer.zero_grad()
        theta_pred = model(inputs) 
        t1 = torch.lgamma(torch.sum(theta_pred, dim=1))
        t2 = torch.lgamma(theta_pred[:,0])
        t3 = torch.lgamma(theta_pred[:,1])
        t4 = theta_pred[:,0] * torch.log(outputs)
        t5 = (theta_pred[:,1] - 1) * torch.log(1 - outputs) 
        comb = t1 - t2 - t3 + t4 + t5
        loss = -torch.sum(comb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

model.save("dispersion_model.pt")
