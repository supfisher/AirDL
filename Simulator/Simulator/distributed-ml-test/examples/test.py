import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
"""
    Build Neural network
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    return model.state_dict()


def evaluate(model, train_loader):
    model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = F.nll_loss(output, target)
        if batch_idx % 100 == 0:
            print('EVAL: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            0, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))


def main_work():
    train_kwargs = {'batch_size': 128}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    return model, train_loader, optimizer, scheduler



def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


if __name__ == "__main__":

    net_global = Net()
    models = []
    for i in range(3):
        model, train_loader, optimizer, scheduler = main_work()
    w_locals = []
    for epoch in range(10):
        for i in range(3):

            model.load_state_dict(net_global.state_dict())
            w = train(model, train_loader, optimizer, epoch)
            w_locals.append(copy.deepcopy(w))

        w_glob = FedAvg(w_locals)
        net_global.load_state_dict(w_glob)
        evaluate(net_global, train_loader)


