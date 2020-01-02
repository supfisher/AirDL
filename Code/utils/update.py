# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: update.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-01-02 (YYYY-MM-DD)
-----------------------------------------------
"""
import torch
from torch import nn
from utils.utils import process_fed
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import copy

torch.manual_seed(2019)


class LocalUpdate(object):
    def __init__(self, args, dataset):
        self.args = args
        self.train_loader, self.test_loader = self.train_test(dataset)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.MSELoss().to(self.device)

    def train_test(self, dataset):
        train_x, train_y, test_x, test_y = process_fed(self.args, dataset)
        X_train = torch.from_numpy(train_x).type(torch.Tensor)
        X_test = torch.from_numpy(test_x).type(torch.Tensor)
        y_train = torch.from_numpy(train_y).type(torch.Tensor)
        y_test = torch.from_numpy(test_y).type(torch.Tensor)

        train_data = list(zip(X_train, y_train))
        val_data = list(zip(X_test, y_test))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.args.local_bs)
        test_loader = DataLoader(val_data, shuffle=False, batch_size=self.args.local_bs)
        return train_loader, test_loader

    def update_weights(self, model, global_round, cell):
        model.train()
        epoch_loss = []

        g_model = copy.deepcopy(model)

        optimizer = torch.optim.Adam(g_model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                g_model.zero_grad()
                pred = g_model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 20 == 0):
                    print('| Global Round : {} | Cell: {}  Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, cell, iter, batch_idx*len(x), len(self.train_loader.dataset),
                        100.*batch_idx/len(self.train_loader), loss.item()
                    ))

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return g_model.state_dict(), sum(epoch_loss)/len(epoch_loss)

    def inference(self, model):
        """
        Returns the predictions
        :param model: model
        :return: prediction
        """
        model.eval()
        loss = 0.0
        pred_list, truth_list = [], []
        mse_list = []

        for batch_idx, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)

            # inference
            pred = model(x)
            batch_loss = self.criterion(pred, y)
            loss += batch_loss.item()

            batch_mse = torch.mean((y - pred) ** 2)
            mse_list.append(batch_mse)

            pred_list.append(pred.cpu())
            truth_list.append(y.cpu())

        mean_square_error = torch.stack(mse_list).mean()

        return loss, mean_square_error


def test_inference(args, model, dataset):
    """
    perform testing
    :param args: model configurations
    :param model: trained global model
    :param dataset: test dataset
    :return: prediction and loss
    """
    model.eval()
    loss, mse = 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.MSELoss().to(device)
    train_shifts = [dataset.shift(i) for i in range(1 - args.out_dim, args.window_size + 1, 1)]

    df_test = pd.concat(train_shifts, axis=1, ignore_index=True)
    df_test.dropna(inplace=True)

    data_x, data_y = df_test.iloc[:, args.out_dim:], df_test.iloc[:, :args.out_dim]
    data_x = torch.from_numpy(data_x.values[:, :, np.newaxis]).type(torch.Tensor)
    data_y = torch.from_numpy(data_y.values).type(torch.Tensor)

    data = list(zip(data_x, data_y))
    data_loader = DataLoader(data, batch_size=args.local_bs, shuffle=False)
    pred_list = []
    truth_list = []
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        batch_loss = criterion(pred, y)
        loss += batch_loss.item()

        batch_mse = torch.mean((pred - y) ** 2)
        mse += batch_mse.item()

        pred_list.append(pred.cpu().detach().numpy())
        truth_list.append(y.cpu().detach().numpy())

    final_prediction = np.concatenate(pred_list).ravel()
    final_truth = np.concatenate(truth_list).ravel()

    return loss, mse, final_prediction, final_truth