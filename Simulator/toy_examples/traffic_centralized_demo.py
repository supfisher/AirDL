# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: traffic_centralized_demo.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-02-29 (YYYY-MM-DD)
-----------------------------------------------
"""
from network import *
import yaml
from models import *
from utils.dataloader_tmp import DataParallel
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import h5py
import random
from sklearn import metrics

parser = argparse.ArgumentParser(description='Wireless traffic prediction example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 1.0)')
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
parser.add_argument('--clients', type=int, default=10, help='number of clients')
parser.add_argument('--epsilon', type=float, default=0.1, help='decay threshold')

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
    f = h5py.File('D:\Projects\FedMMoE\scripts\dataset/milano.h5', 'r')

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


def process_centralized(args, dataset):
    train_x_close, val_x_close, test_x_close = [], [], []
    train_x_period, val_x_period, test_x_period = [], [], []
    train_label, val_label, test_label = [], [], []

    column_names = dataset.columns

    for col in column_names:
        close_arr = []
        period_arr = []
        label_arr = []

        cell_traffic = dataset[col]
        start_idx = args.close_size
        for idx in range(start_idx, len(dataset)-args.out_dim+1):
            y_ = [cell_traffic.iloc[idx+i] for i in range(args.out_dim)]
            label_arr.append(y_)
            if args.close_size > 0:
                x_close = [cell_traffic.iloc[idx - c] for c in range(1, args.close_size + 1)]
                close_arr.append(x_close)

        cell_arr_close = np.array(close_arr)

        cell_label = np.array(label_arr)

        test_len = args.test_days * 24
        val_len = args.val_days * 24
        train_len = len(cell_arr_close) - test_len - val_len

        train_x_close.append(cell_arr_close[:train_len])
        val_x_close.append(cell_arr_close[train_len:train_len+val_len])
        test_x_close.append(cell_arr_close[-test_len:])

        train_label.append(cell_label[:train_len])
        val_label.append(cell_label[train_len:train_len+val_len])
        test_label.append(cell_label[-test_len:])

    train_xc = np.concatenate(train_x_close)[:, :, np.newaxis]
    if len(val_x_close) > 0:
        val_xc = np.concatenate(val_x_close)[:, :, np.newaxis]
    test_xc = np.concatenate(test_x_close)[:, :, np.newaxis]

    train_y = np.concatenate(train_label)
    val_y = np.concatenate(val_label)
    test_y = np.concatenate(test_label)

    return (train_xc, train_y), (val_xc, val_y), (test_xc, test_y)

if __name__ == '__main__':
    # Training settings
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(use_cuda)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = LSTM(args).to(device)

    data, df_ori, selected_cells, mean, std = get_data(args)
    selected_cells = data.columns
    train, val, test = process_centralized(args, data)
    train_data = list(zip(*train))  # I am not quit sure the * operation
    test_data = list(zip(*test))

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    train_loss = []
    results = []
    train_truth, train_y_hat = [], []
    for epoch in range(args.epochs):
        model.train()
        batch_loss = []
        for idx, (xc, y) in enumerate(train_loader):
            xc = xc.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            pred = model(xc)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            if idx % 30 == 0:
                print('Epoch {:} [{:}/{:} ({:.2f}%)]\t Loss: {:.4f}'.format(epoch, idx, len(train_loader),
                                                                            idx/len(train_loader)*100, loss.item()))
            batch_loss.append(loss.item())

        avg_batch = sum(batch_loss)/len(batch_loss)
        print('Epoch {:}, Avg loss {:.4f}'.format(epoch, avg_batch))
        train_loss.append(avg_batch)

        # validation
        model.eval()
        val_loss = []
        truth, pred = [], []
        for idx, (xc, y) in enumerate(test_loader):
            xc = xc.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            y_hat = model(xc)

            loss = criterion(y_hat, y)
            val_loss.append(loss.item())
            truth.append(y.detach().cpu())
            pred.append(y_hat.detach().cpu())

        truth_arr = np.concatenate(truth).reshape((-1, args.test_days * 24)).T
        pred_arr = np.concatenate(pred).reshape((-1, args.test_days * 24)).T
        mse = metrics.mean_squared_error(truth_arr.ravel(), pred_arr.ravel())
        mae = metrics.mean_absolute_error(truth_arr.ravel(), pred_arr.ravel())

        results.append((avg_batch, mse, mae))

        print('Test MSE: {:.4f}'.format(mse))
        print('Test MAE: {:.4f}'.format(mae))

    file_name = './data/traffic_exp_centralized.csv'

    df_train_loss = pd.DataFrame(results, columns=['train_loss', 'test_mse', 'test_mae'])
    df_train_loss.to_csv(file_name, index=False)
