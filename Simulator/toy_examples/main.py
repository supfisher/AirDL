from network import Topo
import yaml
from models import *
from utils import *
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from wireless import Gaussian
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = data.split([64, 64], dim=0), target.split([64, 64], dim=0)
        optimizer.zero_grad()
        output = model(data)
        if output is not None:
            loss = criterion(output, target)
            for l in loss:
                l.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)))
                print("loss: ", list([l.item() for l in loss]))



def test(args, model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    print("test")
    topo = Topo()
    with open('./data/simple_graph.yaml', 'r') as f:
        dict = yaml.load(f)
    topo.load_from_dict(dict)
    print(topo)
    print('rank: ', topo.rank)
    print("server: ", topo.servers)
    print("client: ", topo.clients)



    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    train_loader = DataParallel(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), topo=topo,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataParallel(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), topo=topo,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model = Net().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # model_p = ModelParallel(model, topo, Qos=Gaussian())
    model_p = ModelParallel(model, topo=topo, Qos=Gaussian())

    optimizer = OptimizerParallel(optim.Adadelta, model_p.parameters(), lr=args.lr)
    criterion = CriterionParallel(F.nll_loss)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model_p, criterion, device, train_loader, optimizer, epoch)
        test(args, model_p, criterion, device, test_loader)
        # scheduler.step()

    if args.save_model:
        torch.save(model_p.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()

    # class Buffer:
    #     def __init__(self, _data):
    #         self._data = _data
    #
    #     @property
    #     def data(self):
    #         return self._data+1
    #
    #     @data.setter
    #     def data(self, value):
    #         self._data = value
    #
    #     def set_data(self, value):
    #         self.update(value)
    #         return self._data
    #
    #     def update(self, data):
    #         for _d, d in zip(self._data, data):
    #             _d += d+10
    #
    #
    # import threading
    # class Send(threading.Thread):
    #     def __init__(self, data, dst):
    #         self.data = data
    #         self.dst = dst
    #         super(Send, self).__init__()
    #         # self.run()
    #
    #     def run(self):
    #         w = dist.isend(self.data, dst=self.dst)
    #         w.wait()
    #
    # class Receive(threading.Thread):
    #     def __init__(self, data, dst):
    #         self.data = data
    #         self.src = dst
    #         super(Receive, self).__init__()
    #         # self.run()
    #
    #     def run(self):
    #         w = dist.irecv(self.data, src=self.src)
    #         w.wait()
    #
    # import torch.distributed as dist
    #
    # dist.init_process_group(backend='mpi')
    # a = torch.ones([30,30])*10
    # b = torch.ones([30,30])*-100
    # c, d = torch.rand([30,30]), torch.rand([30,30])
    # buff = Buffer(c)
    # sender = [Send(a, 0), Send(b, 0)]
    #
    # recevicer = [Receive(buff.set_data(c), 0), Receive(buff.set_data(d), 0)]
    #
    # for s in sender:
    #     s.run()
    # for r in recevicer:
    #     r.run()
    # # print(c)
    # # print(d)
    # print(buff.data)
    #


    # import copy
    # class A:
    #     def __init__(self, a=None):
    #         self.a = a
    #
    #     def test(self, a=10):
    #         self.a = a
    #         print("A: ", self.a)
    #
    # class B:
    #     class ListCall:
    #         def __init__(self, list_fn):
    #             self.list_fn = list_fn
    #
    #         def __call__(self, *args, **kwargs):
    #             for fn in self.list_fn:
    #                 fn(*args, **kwargs)
    #
    #     def __init__(self, A=None):
    #         self.A = [copy.deepcopy(A()) for _ in range(3)]
    #         for key in dir(A):
    #             if '__' not in key:
    #                 value = self.ListCall([getattr(a, key) for a in self.A])
    #                 setattr(self, key, value)
    #
    #
    #     def run(self, list_fn):
    #         for fn in list_fn:
    #             fn()
    #         return self.empty
    #
    #     def empty(self, *args, **kwargs):
    #         pass
    #
    # test = A().test
    # print(A.__dict__)
    # print(test.__code__)
    # print(dir(test.__code__))
    # print(test.__code__.co_argcount)
    #
    #
    # b = B(A)
    # print("after initial")
    # b.test(11)
    # getattr(b, 'test')(12)
