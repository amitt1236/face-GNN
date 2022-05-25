import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader

import random
import data_to_graph
from datetime import datetime

from tensorboardX import SummaryWriter


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNStack, self).__init__()
        self.convs = nn.ModuleList()  # creating a list of gnn convs layers
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()  # creating a list of Layer Normalization
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(3):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#dense-convolutional-layers
        return pyg_nn.GATConv(input_dim, hidden_dim, heads=1, dropout=0.2, edge_dim=1)

    def forward(self, data):
        x, edge_index, edge_atrr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_atrr)  # applying conv
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = pyg_nn.global_mean_pool(x, batch)  # global mean pooling of all the nodes features
        x = self.post_mp(x)  # additional two linear layers

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)  # The negative log likelihood loss


def train(dataset, writer):
    # loading data into batches for training 
    data_size = len(dataset)
    loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)

    # build model
    model = GNNStack(2, 64, 2)
    opt = optim.Adam(model.parameters(), lr=0.01)
    sec = optim.lr_scheduler.StepLR(opt, 50, 0.1)
    MAX_ACC = 0
    # train
    for epoch in range(200):
        total_loss = 0
        model.train()

        for batch in loader:
            opt.zero_grad()

            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)

            loss.backward()
            opt.step()

            total_loss += loss.item() * batch.num_graphs

        sec.step()  # step of learning rate scheduler

        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)  # writing loss into tensorboard

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            # saving the model with max acc on the test set
            if test_acc > MAX_ACC:
                MAX_ACC = test_acc
                torch.save(model, './models/' + str(MAX_ACC) + '.pth')

            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)

    return model


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():  # testing, not calculating gradiants
            pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        correct += pred.eq(label).sum().item()

    total = len(loader.dataset)
    return correct / total


if __name__ == "__main__":
    writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # load data
    data = data_to_graph.data()
    random.shuffle(data)

    # start training 
    model = train(data, writer)
