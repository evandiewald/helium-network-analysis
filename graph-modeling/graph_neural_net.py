import torch
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv, GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
from networkx import read_gpickle
from get_data import *

g = read_gpickle('pittsburgh_graph.pkl')
g = clean_graph_mapping_data(g)
# for node in g.nodes():
#     del g.nodes()[node]['location_hex']
#     del g.nodes()[node]['name']
data_full = from_networkx(g)

features = torch.hstack((data_full.elev.reshape((data_full.num_nodes,1)), data_full.num_hotspots.reshape((data_full.num_nodes,1))))#, data_full.pos))
# features = data_full.pos

mask = np.zeros((data_full.num_nodes))
train_mask = mask.astype(bool)
train_mask[:int(data_full.num_nodes*0.7)] = True

val_mask = mask.astype(bool)
val_mask[int(data_full.num_nodes*0.7)+1:int(data_full.num_nodes*0.9)] = True

test_mask = mask.astype(bool)
test_mask[int(data_full.num_nodes*0.9)+1:] = True

data = Data(x=features.float(),
            y=data_full.snr.float(),
            edge_index=data_full.edge_index,
            # edge_attr=data_full.weight,
            train_mask=torch.Tensor(train_mask).type(torch.bool),
            val_mask=torch.Tensor(val_mask).type(torch.bool),
            test_mask=torch.Tensor(test_mask).type(torch.bool),
            pos=data_full.pos,
            )
# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(data.num_features, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.conv1 = GCNConv(data.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, 1, cached=True)
        self.lin2 = torch.nn.Linear(16, 16)
        self.lin3 = torch.nn.Linear(16, 1)

    def forward(self):
        # x = F.dropout(data.x, training=self.training)
        # x = F.relu(self.lin1(x))
        # x = self.prop1(x, data.edge_index)
        # x = self.prop2(x, data.edge_index)

        x, edge_index, edge_weight, pos = data.x, data.edge_index, data.edge_attr, data.pos
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        # x = F.relu(self.conv2(x, pos))
        x = F.dropout(x, training=self.training)
        # x = self.lin2(x)
        # x = self.lin3(x)
        return x  # F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    F.mse_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    # F.l1_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    pred, losses = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):

        loss = F.mse_loss(data.y[mask], pred[mask])
        # loss = F.l1_loss(data.y[mask], pred[mask])

        losses.append(loss)
    return losses


best_val_loss = test_loss = 0
for epoch in range(1, 201):
    train()
    train_loss, val_loss, test_loss = test()
    # if val_loss > best_val_loss:
    #     best_val_loss = val_loss
    #     test_loss = tmp_test_loss
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, val_loss, test_loss))


pred = model().detach().numpy()
real = data.y.detach().numpy()
