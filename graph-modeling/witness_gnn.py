import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, AGNNConv, SAGEConv, XConv
from torch_geometric.data import Data, DataLoader, Dataset
import pickle
from torch.nn import Linear, ReLU, Flatten
import random
import numpy as np
import matplotlib.pyplot as plt


print('Loading dataset...')
with open('subgraph_datasets/subgraph_data_2021_08_26_19_45_36.pkl', 'rb') as f:
    data_list = pickle.load(f)


for data in data_list:
    data.num_classes = 2


def create_dataloaders(data_list, train_ratio, val_ratio, batch_size):
    random.shuffle(data_list)
    data_train, data_val, data_test = data_list[0:int(train_ratio*len(data_list))], data_list[int(train_ratio*len(
        data_list)):int((train_ratio + val_ratio)*len(data_list))], data_list[int((train_ratio + val_ratio)*len(data_list)):len(data_list)+1]
    train_loader = DataLoader(data_train, batch_size=batch_size)
    val_loader = DataLoader(data_val, batch_size=batch_size)
    test_loader = DataLoader(data_test, batch_size=batch_size)
    return train_loader, val_loader, test_loader

batch_size = 8
print('Splitting dataset and creating DataLoaders...')
train_loader, val_loader, test_loader = create_dataloaders(data_list, 0.7, 0.2, batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(train_loader.dataset[0].num_features, 16)

        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)

        self.conv1 = GCNConv(train_loader.dataset[0].num_features, 16, normalize=False)
        self.conv2 = GCNConv(16, 16, normalize=False)
        self.conv3 = GCNConv(16, 16, normalize=False)

        self.xconv = XConv(16, 16, dim=2, kernel_size=3)
        self.lin2 = torch.nn.Linear(16, train_loader.dataset[0].num_classes)

    def forward(self, data):
        x, edge_index, edge_weight, pos = data.x, data.edge_index, data.edge_attr, data.pos

        # x = F.relu(self.lin1(x))
        # # x = self.prop1(x, edge_index)
        # # x = self.prop2(x, edge_index)
        # x = self.xconv(x, pos)
        #
        # # x = F.dropout(x, training=self.training)
        # x = self.lin2(x)

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        # x = self.conv2(x, edge_index, edge_weight)
        x = self.lin2(x)

        return F.softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)


train_loss_epoch = []
val_loss_epoch = []
for epoch in range(200):
    train_losses, val_losses = [], []
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        # if batch.y.shape != (batch_size,):
        #     continue
        # train_loss = F.mse_loss(out, batch.y)
        y_one_hot = F.one_hot(batch.y, num_classes=2).type(torch.float32)
        train_loss = F.binary_cross_entropy(out, y_one_hot)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

    with torch.no_grad():
        for val_batch in val_loader:
            val_batch.to(device)

            model.eval()

            pred = model(val_batch)

            val_loss = F.binary_cross_entropy(pred, F.one_hot(val_batch.y, num_classes=2).type(torch.float32))
            val_losses.append(val_loss.item())


    print(f"Epoch: {str(epoch)}\t Train Loss: {str(np.mean(train_losses))}\t Validation Loss:"
          f" {str(np.mean(val_losses))}")


print('Training complete. Evaluating on test data...')
predictions, actual = [], []
for test_batch in test_loader:
    test_batch.to(device)
    model.eval()
    pred = model(test_batch)
    pred_classes = np.argmax(pred.cpu().detach().numpy(), axis=1)
    predictions += list(pred_classes)

    actual += list(test_batch.y.cpu().detach().numpy())

num_correct = len([predictions.index(y) for x, y in zip(predictions, actual) if y == x])
acc = num_correct / len(actual)

# how accurate would it be if we guessed all witnesses?
all_yes = [1] * len(actual)
num_correct_all_yes = len([all_yes.index(y) for x, y in zip(all_yes, actual) if y == x])
acc_all_yes = num_correct_all_yes / len(actual)

print("Model Accuracy on Test Set:        {:2.2%}\n"
      "Accuracy if guessed all witnesses: {:2.2%}".format(acc, acc_all_yes))