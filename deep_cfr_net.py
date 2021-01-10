import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AdvantageNet(nn.Module):

    def __init__(self, embed_dim=9):

        super(AdvantageNet, self).__init__()

        self.fc1 = nn.Linear(embed_dim, 128)
        #self.fc1.weight.data.fill_(0.0)
        #self.fc1.bias.data.fill_(0.0)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 64)
        #self.fc2.weight.data.fill_(0.0)
        #self.fc2.bias.data.fill_(0.0)

        self.fc4 = nn.Linear(64, 32)
        #self.fc3.weight.data.fill_(0.0)
        #self.fc3.bias.data.fill_(0.0)

        self.norm_layer = nn.LayerNorm(32, 32)

        self.fc5 = nn.Linear(32, 2)
        #self.fc4.weight.data.fill_(0.0)
        #self.fc4.bias.data.fill_(0.0)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.norm_layer(x)
        x = self.fc5(x)

        return x


class PolicyNet(nn.Module):

    def __init__(self, embed_dim=9):

        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x))

        return x


def loss_fn(y_pred, t, y):

    loss = t.view(-1, 1) * (y_pred - y)**2

    return torch.sum(loss) / len(t)
