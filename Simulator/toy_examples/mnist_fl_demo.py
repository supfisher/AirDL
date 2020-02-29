# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: mnist_fl_demo.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-02-28 (YYYY-MM-DD)
-----------------------------------------------
"""
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
from time import time
import numpy as np
import pandas as pd
from collections import deque

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--num_channels', type=int, default=1, help='input channels')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before .logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

parser.add_argument('--dist_url', default='tcp://127.0.0.1:8001', type=str,
                    help='For Saving the current Model')
parser.add_argument('--rank', default=0, type=int,
                    help="Currently, we only support the backend of mpi and gloo,You don't need "
                         "to care about it if using mpi. However, you have to assert it to be 0"
                         "on your master process.")
parser.add_argument('--world_size', default=0, type=int,
                    help="The total number of processes.")
parser.add_argument('--clients', type=int, default=10, help='number of clients')
parser.add_argument('--epsilon', type=float, default=0.1, help='time window')
parser.add_argument('--mode', type=str, default='wireless', help='channel mode')
parser.add_argument('--stop', type=bool, default=False, help='set a stop condition or not')
parser.add_argument('--condition', type=float, default=95, help='stop condition value')


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    removed_links = model.qos.removed_edges
    down_links = removed_links[:args.clients].sum().item()
    up_links = removed_links[args.clients:].sum().item()

    batch_loss = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if output is not None:
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            # print(loss.item())
            # print("report: ", args.topo.report)

    avg_loss = np.array(batch_loss).mean(axis=0)
    consume_energy = args.topo.report.energy_cost
    used_time = args.topo.report.time_cost
    goodput = args.topo.report.throughput
    packet_loss = args.topo.report.packet_loss
    print('Train Epoch: {} [{}/{} ({:.2f})%], unstable up-links: {}, down-links: {}, '
          'decay threshold: {:.4f}, Avg loss: {:.4f}, '
          'Consumed energy: {:.4f}, Used time: {:.4f}, '
          'Goodput: {:.4f}, Packet loss: {:.4f}'.format(epoch, epoch,
                                                        args.epochs,
                                                        100. * epoch / args.epochs,
                                                        up_links,
                                                        down_links,
                                                        args.epsilon,
                                                        avg_loss.mean(axis=0),
                                                        consume_energy,
                                                        used_time, goodput,
                                                        packet_loss))
    return avg_loss, down_links, up_links, consume_energy, used_time, goodput, packet_loss


def test(args, model, criterion, device, test_loader, epoch):
    model.eval()
    test_loss = []
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            test_loss.append(sum(loss) / len(loss))  # sum up batch loss
            out = torch.cat(output, dim=0)
            pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss = sum(test_loss) / len(test_loss)
    val_acc = 100. * correct / len(test_loader.dataset)

    print('Test Epoch: {}, epsilon threshold: {:.4f}, Avg loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epoch, args.epsilon, val_loss, val_acc
    ))
    return val_loss, val_acc


def main():
    # Training settings
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(use_cuda)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")

    # model = Net().to(device)
    model = CNNMnist(args).to(device)

    print(args.batch_size)

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
    topo = RandTopo(model, backend='none', rank=args.rank, size=args.world_size + 1, dist_url=args.dist_url,
                    rand_method=('static', args.clients))
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

    qos = QoSDemo(topo, args)
    model_p = ModelParallel(qos=qos)
    print(model_p.qos.channel.epsilon)
    optimizer = OptimizerParallel(optim.SGD, model_p.parameters(), lr=args.lr)
    criterion = CriterionParallel(F.nll_loss, topo=topo)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    train_loss_history, d_link_history, u_link_history = [], [], []
    test_loss_history, test_acc, test_auc = [], [], []
    exp_results = []
    queue_len = 5
    test_acc_deque = deque([90.0]*queue_len, maxlen=queue_len)
    for epoch in range(1, args.epochs + 1):
        # Training the model for one epoch and get the results
        train_loss, d_link, u_link, consumed_energy, used_time, goodput, packe_tloss = train(args, model_p,
                                                                                             criterion, device,
                                                                                             train_loader,
                                                                                             optimizer, epoch)

        train_loss_history.append(train_loss)
        # d_link_history.append(d_link)
        # u_link_history.append(u_link)

        # Test model performance
        avg_model = AvgParallel(model_p.aggregate(), args)
        test_loss, acc = test(args, avg_model, criterion, device, test_loader, epoch)
        test_acc_deque.append(acc)
        if args.stop & (np.mean(test_acc_deque) > args.condition):
            break
        test_loss_history.append(test_loss)
        # test_acc.append(acc)
        exp_results.append((args.epsilon, d_link, u_link, acc, consumed_energy, used_time, goodput, packe_tloss))

    df_exp = pd.DataFrame(exp_results, columns=['epsilon', 'down_link',
                                                'up_link', 'test_acc', 'energy', 'time', 'goodput', 'packet_loss'])
    file_name = './data/mnist_exp_epsilon={}_clients={}_channel={}'.format(args.epsilon, args.clients, args.mode)
    df_exp.to_csv(file_name + '_acc.csv', index=False, float_format='%.4f')

    df_train_loss = pd.DataFrame(train_loss_history)
    df_test_loss = pd.DataFrame(test_loss_history)

    df_train_loss.to_csv(file_name + '_train_loss.csv', index=False, float_format='%.4f')
    df_test_loss.to_csv(file_name + '_test_loss.csv', index=False, float_format='%.4f')

    if args.save_model:
        torch.save(model_p.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
