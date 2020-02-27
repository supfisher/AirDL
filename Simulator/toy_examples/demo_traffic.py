# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: demo_traffic.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-02-26 (YYYY-MM-DD)
-----------------------------------------------
"""
from network import *
import yaml
from models import *
from utils.dataloader_v2 import DataParallel
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from time import time
import numpy as np
import pandas as pd
import h5py
import random
from sklearn import metrics

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
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
parser.add_argument('--rank', default=1, type=int,
                    help="Currently, we only support the backend of mpi and gloo,You don't need "
                         "to care about it if using mpi. However, you have to assert it to be 0"
                         "on your master process.")
parser.add_argument('--world_size', default=2, type=int,
                    help="The total number of processes.")
parser.add_argument('--clients', type=int, default=20, help='number of clients')
parser.add_argument('--epsilon', type=float, default=0.6, help='decay threshold')

parser.add_argument('--close_size', type=int, default=3,
                    help='how many time slots before target are used to model closeness')
parser.add_argument('--test_days', type=int, default=7,
                    help='how many days data are used to test model performance')
parser.add_argument('--val_days', type=int, default=0,
                    help='how many days data are used to valid model performance')

# FL Neural Network parameters
parser.add_argument('--input_dim', type=int, default=1,
                    help='input feature dimension of LSTM')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='hidden neurons of LSTM layer')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers of LSTM')
parser.add_argument('--out_dim', type=int, default=1,
                    help='how many steps we would like to predict for the future')


def get_data(args):
    f = h5py.File('./data/milano.h5', 'r')

    idx = f['idx'][()]
    cell = f['cell'][()]
    data = f['net'][()][:, cell - 1]

    df = pd.DataFrame(data, index=pd.to_datetime(idx.ravel(), unit='s'), columns=cell)
    df.fillna(0, inplace=True)

    random.seed(args.seed)
    selected_cells = sorted(random.sample(list(cell), args.clients))
    # print('Selected cells:', selected_cells)

    df_cells = df[selected_cells]
    # print(df_cells.head())

    train_data = df_cells.iloc[:-7 * 24]
    mean = train_data.mean()
    std = train_data.std()

    normalized_df = (df_cells - mean) / std
    # print(normalized_df.head())

    return normalized_df, df_cells, selected_cells, mean, std


def process_isolated(args, dataset):
    train, val, test = dict(), dict(), dict()
    column_names = dataset.columns

    for col in column_names:
        close_arr, period_arr, label_arr = [], [], []

        cell_traffic = dataset[col]
        start_idx = args.close_size
        for idx in range(start_idx, len(dataset) - args.out_dim + 1):
            y_ = [cell_traffic.iloc[idx + i] for i in range(args.out_dim)]
            label_arr.append(y_)

            if args.close_size > 0:
                x_close = [cell_traffic.iloc[idx - c] for c in range(1, args.close_size + 1)]
                close_arr.append(x_close)

        cell_arr_close = np.array(close_arr)
        cell_arr_close = cell_arr_close[:, :, np.newaxis]
        cell_label = np.array(label_arr)

        test_len = args.test_days * 24
        val_len = args.val_days * 24
        train_len = len(cell_arr_close) - test_len - val_len

        train_x_close = cell_arr_close[:train_len]
        val_x_close = cell_arr_close[train_len:train_len + val_len]
        test_x_close = cell_arr_close[-test_len:]

        train_label = cell_label[:train_len]
        val_label = cell_label[train_len:train_len + val_len]
        test_label = cell_label[-test_len:]

        train[col] = (train_x_close, train_label)
        val[col] = (val_x_close, val_label)
        test[col] = (test_x_close, test_label)

    return train, val, test


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    removed_links = model.qos.removed_edges
    down_links = removed_links[:args.clients].sum().item()
    up_links = removed_links[args.clients:].sum().item()

    batch_loss = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        if output is not None:
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

    avg_loss = np.array(batch_loss).mean(axis=0)
    consume_energy = args.topo.report.energy_cost
    used_time = args.topo.report.time_cost
    print('Train Epoch: {} [{}/{} ({:.2f})%], unstable up-links: {}, down-links: {}, '
          'decay threshold: {:.4f}, Avg loss: {:.4f}, '
          'Consumed energy: {:.4f}, Used time: {:.4f}'.format(epoch, epoch,
                                                              args.epochs,
                                                              100. * epoch / args.epochs,
                                                              up_links,
                                                              down_links,
                                                              args.epsilon,
                                                              avg_loss.mean(axis=0),
                                                              consume_energy,
                                                              used_time))
    return avg_loss, down_links, up_links, consume_energy, used_time


def test(args, model, criterion, device, test_loader, epoch):
    model.eval()
    test_loss = []
    correct = 0
    batch_loss = []

    pred_list, truth_list = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device), target.float().to(device)
            output = model(data)
            loss = criterion(output, target).item()
            batch_loss.append(loss)
            pred_list.append([o.numpy().ravel() for o in output])
            truth_list.append([t.numpy().ravel() for t in target])

    final_pred = np.concatenate(pred_list).ravel()
    final_truth = np.concatenate(truth_list).ravel()
    avg_loss = np.array(batch_loss).mean(axis=0)
    mse = metrics.mean_squared_error(final_truth, final_pred)
    mae = metrics.mean_absolute_error(final_truth, final_pred)
    print('Test Epoch: {}, epsilon threshold: {:.4f}, Avg loss: {:.4f}, MSE: {:.4f}, MAE: {:.4f}'.format(
        epoch, args.epsilon, avg_loss.mean(axis=0), mse, mae
    ))
    return avg_loss, mse, mae


def main():
    # Training settings
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(use_cuda)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = LSTM(args).to(device)
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

    data, df_ori, selected_cells, mean, std = get_data(args)
    selected_cells = data.columns
    train_data, val_data, test_data = process_isolated(args, data)

    for cell_id in selected_cells:
        print(cell_id, train_data[cell_id][0].shape, train_data[cell_id][1].shape)

    train_loader = DataParallel(train_data, topo=topo,
                                batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataParallel(test_data, topo=topo,
                               batch_size=args.test_batch_size, shuffle=True, **kwargs)

    qos = QoSDemo(topo)
    model_p = ModelParallel(topo=topo, QoS=qos)
    print(model_p.qos.channel.epsilon)
    optimizer = OptimizerParallel(optim.SGD, model_p.parameters(), lr=args.lr)
    criterion = CriterionParallel(F.mse_loss, topo=topo)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    train_loss_history, d_link_history, u_link_history = [], [], []
    test_loss_history, test_acc, test_auc = [], [], []
    exp_results = []
    for epoch in range(1, args.epochs + 1):
        # Training the model for one epoch and get the results
        train_loss, d_link, u_link, consumed_energy, used_time = train(args, model_p,
                                                                       criterion, device,
                                                                       train_loader,
                                                                       optimizer, epoch)

        train_loss_history.append(train_loss)
        # d_link_history.append(d_link)
        # u_link_history.append(u_link)

        # Test model performance
        test_loss, mse, mae = test(args, model_p, criterion, device, test_loader, epoch)
        test_loss_history.append(test_loss)
        # # test_acc.append(acc)
        exp_results.append((args.epsilon, d_link, u_link, mse, mae, consumed_energy, used_time))

    df_exp = pd.DataFrame(exp_results, columns=['epsilon', 'down_link', 'up_link', 'test_mse',
                                                'test_mae', 'energy', 'time'])
    file_name = './data/exp_epsilon={}'.format(args.epsilon)
    df_exp.to_csv(file_name + '_acc.csv', index=False, float_format='%.4f')

    df_train_loss = pd.DataFrame(train_loss_history)
    df_test_loss = pd.DataFrame(test_loss_history)

    df_train_loss.to_csv(file_name + '_train_loss.csv', index=False, float_format='%.4f')
    df_test_loss.to_csv(file_name + '_test_loss.csv', index=False, float_format='%.4f')

    if args.save_model:
        torch.save(model_p.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
