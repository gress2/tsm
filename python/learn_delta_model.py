import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

def load_data(path_to_file):
    with open(path_to_file, 'r') as data:
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
x_test, y_Test = load_data('test.dkd.csv')

bs = 32
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

class DeltaModel(torch.jit.ScriptModule):
    def __init__(self):
        super(DeltaModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 8)
        self.fc2 = torch.nn.Linear(8, 8)
        self.fc3 = torch.nn.Linear(8, 8)
        self.fc4 = torch.nn.Linear(8, 8)
        self.fc5 = torch.nn.Linear(8, 3)
        self.dropout = torch.nn.Dropout(p=.2)

    @torch.jit.script_method
    def forward(self, x):
        d = x[:, 0] / 81.
        k = (x[:, 1] - 2) / 51.
        x = torch.stack((d, k), dim=1)
        x = torch.clamp(self.fc1(x), min=0)
        x = torch.clamp(self.fc2(x), min=0)
        x = torch.clamp(self.fc3(x), min=0)
        x = torch.clamp(self.fc4(x), min=0)
        x = torch.clamp(self.fc5(x), min=0)
        lambda_pos = torch.clamp(x[:,0], min=1e-5)
        lambda_neg = torch.clamp(x[:,1], min=1e-5)
        p = torch.clamp(x[:,2], max=1-1e-5)
        return torch.stack((lambda_pos, lambda_neg, p), dim=1)

model = DeltaModel().double()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=.01)


