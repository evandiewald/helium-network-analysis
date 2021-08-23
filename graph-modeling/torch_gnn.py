import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx, train_test_split_edges
from torch_geometric.data import Dataset, ClusterData, ClusterLoader, NeighborSampler, Data
from torch_geometric.transforms import AddTrainValTestMask
from networkx import read_gpickle

g = read_gpickle('pittsburgh_graph_cleaned.pkl')
for node in g.nodes():
    del g.nodes()[node]['location_hex']
    del g.nodes()[node]['name']
data_full = from_networkx(g)

data_train = Data(x=data_full['elevation'][:350],
                  y=data_full['rssi'][:350],
                  edge_index=data_full.edge_index,
                  pos=data_full.pos[:350],
                  edge_attr=data_full['weight'][:350])

data_test = Data(x=data_full['elevation'][350:],
                 y=data_full['rssi'][350:],
                 edge_index=data_full.edge_index,
                 pos=data_full.pos[350:],
                 edge_attr=data_full['weight'][350:])


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(data_train.num_features, 16)
        self.conv1 = GCNConv(16, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin2 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.relu(self.lin1(x))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.lin2(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data_train = data_train.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data_train)
    loss = F.nll_loss(out[data_train], data_train.y)
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data_test).max(dim=1)
# correct = int(pred[data_test.x].eq(data_test.y).sum().item())
# # acc = correct / int(data.test_mask.sum())
# print('Accuracy: {:.4f}'.format(acc))

