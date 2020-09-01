from Myown_dataset_Pytorch import Myown_Dataset
import torch
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# import data
dataset = Myown_Dataset(root='E:\Pytorch_samples\Myown_dataset', name='Cor')

# shuffle data
perm = torch.randperm(len(dataset))
dataset = dataset[perm]

# get the first 230 graphs as training (70%)
dataset_train = dataset[:230]
loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)

# get the remaining data for test
dataset_test = dataset[231:]
loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn


# Network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 3)
        self.conv2 = GCNConv(3, 1)
        self.outlayer = nn.Linear(5, 2)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # I don't get it why we have to use float here. In the introduction, everything was with torch.long....
        x = torch.tensor(x, dtype=torch.float, device=device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.reshape(x, (1,5))
        x = self.outlayer(x)

        return F.log_softmax(x, dim=1)


# training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
loss_fun_all = []
for epoch in range(100):
    train_running_loss = 0.0
    train_acc = 0.0
    i = 0
    labels = []

    for data in loader_train:

        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        labels.append(data.y)

        train_running_loss += loss.detach().item()
        train_acc += (torch.argmax(out, 1).flatten() == data.y).type(torch.float).mean().item()

        i += 1

    loss_fun_all.append(train_running_loss/ i)
    print('Epoch: %d | Loss: %.8f | Train Accuracy: %.8f' \
          % (epoch, train_running_loss / i, train_acc / i))


plt.plot(loss_fun_all)
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()

# testing
model.eval()
test_acc = 0.0
i = 0
labels_GT = []
Predict_model = []

for data in loader_test:

    data = data.to(device)
    out = model(data)
    test_acc += (torch.argmax(out, 1).flatten() == data.y).type(torch.float).mean().item()
    preds = torch.argmax(out, 1).flatten().cpu().detach().numpy()

    labels_GT.append(data.y)
    Predict_model.append(preds)

    i += 1

print('Test Accuracy: %.2f'%(test_acc/i))


