from network import *
import yaml
from models import *
from utils import *
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
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
                    help='how many batches to wait before .logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

parser.add_argument('--backend', default=None, type=str,
                    help="The backend used for the distributed system.")
parser.add_argument('--dist_url', default='tcp://127.0.0.1:8001', type=str,
                    help='For Saving the current Model')
parser.add_argument('--rank', default=1, type=int,
                    help="Currently, we only support the backend of mpi and gloo,You don't need "
                         "to care about it if using mpi. However, you have to assert it to be 0"
                         "on your master process.")
parser.add_argument('--world_size', default=2, type=int,
                    help="The total number of processes.")
parser.add_argument('--clients', default=2, type=int,
                    help="The total number of processes.")


parser.add_argument('--epsilon', default=0.2, type=float,
                    help="Chanel sensitive paramters: delay criterion.")


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if output is not None:
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data[0]), len(train_loader.datasets[0]),
                    100. * batch_idx / len(train_loader)))

                args.topo.report('rank', args.topo.rank, 'reset')
                args.topo.report('loss', list(loss.item()), 'reset')
                print("report: ", args.topo.report)
                args.topo.report.write('results_'+str(args.clients)+'_epsilon_'+str(args.epsilon)+'.json')


def test(args, model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    args = parser.parse_args()
    args.batch_size = int(640/args.clients)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net()

    # topo = Topo(model)
    # with open('./data/simple_graph.yaml', 'r') as f:
    #     dict = yaml.load(f)
    # topo.load_from_dict(dict)

    """
        You need to specify a backend: 'mpi', 'gloo' or None
        For using 'mpi', you have to re-complie Pytorch.
        For using 'gloo', you need to follow the guideline in run.sh
        For using None, it is running on a single CPU. 
    """
    topo = RandTopo(model, backend=args.backend, rank=args.rank, size=args.world_size+1, dist_url=args.dist_url,
                    rand_method=('static', args.clients))
    if topo.rank == 0:
        print(topo)

    args.topo = topo

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

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

    qos = QoSDemo(topo, epsilon=args.epsilon)
    model_p = ModelParallel(qos=qos)
    optimizer = OptimizerParallel(optim.Adadelta, model_p.parameters(), lr=args.lr)
    criterion = CriterionParallel(F.nll_loss, topo=topo)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model_p, criterion, device, train_loader, optimizer, epoch)
        # test(args, model_p, criterion, device, test_loader)
        # scheduler.step()

    if args.save_model:
        torch.save(model_p.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()

