# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: fed_tp_demo.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-01-02 (YYYY-MM-DD)
-----------------------------------------------
"""
import copy
import h5py
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
sys.path.append('../')
from utils.options import args_parser
from utils.utils import get_dataset, process_fed, average_weights, weighted_average_weights
from utils.update import LocalUpdate, test_inference
from models.lstm import LSTM

torch.manual_seed(2019)

if __name__ == '__main__':
    args = args_parser()
    device = 'cuda' if args.gpu else 'cpu'

    train, test, cell_idx = get_dataset(args)
    print(cell_idx)

    global_model = LSTM(args).to(device)
    print(global_model)

    global_weights = global_model.state_dict()

    # training
    train_loss = []
    print_interval = 2

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        for cell in cell_idx:
            cell_train = train.loc[:, cell]
            w_idx = cell_idx.tolist().index(cell)

            local_model = LocalUpdate(args, cell_train)

            global_model.load_state_dict(global_weights)

            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch, cell=w_idx)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)
        # global_weights = weighted_average_weights(args, local_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # calculate avg training MSE over all users at every epoch
        loss_list = []
        mse_list = []
        for cell in cell_idx:
            cell_train = train.loc[:, cell]
            w_idx = cell_idx.tolist().index(cell)
            global_model.load_state_dict(global_weights)
            local_model = LocalUpdate(args, cell_train)
            loss, mse = local_model.inference(model=global_model)
            loss_list.append(loss)
            mse_list.append(mse.item())

        if (epoch+1) % print_interval == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss: {np.mean(np.array(train_loss))}')
            print(f'Val Loss: {np.mean(np.array(loss_list))}')
            print(f'Val MSE: {np.mean(np.array(mse_list))}')

    # test inference after completing training
    pred = dict()
    truth = dict()
    test_loss_list, test_mse_list = [], []
    for cell in cell_idx:
        cell_test = test.loc[:, cell]
        w_idx = cell_idx.tolist().index(cell)
        global_model.load_state_dict(global_weights)
        test_loss, test_mse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
        print(test_mse)
        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    print('MSE: {:.4f}'.format(np.mean(np.array(test_mse_list))))
    print('Loss: {:.4f}'.format(np.mean(np.array(test_loss_list))))

    f = h5py.File('pred_fed_'+str(args.alpha)+'.h5', 'w')
    f.create_dataset(name='train_loss', data=train_loss)
    f.create_dataset(name='test_loss', data=test_loss_list)
    f.create_dataset(name='pred', data=df_pred)
    f.create_dataset(name='truth', data=df_truth)
    f.close()

    pred = np.reshape(df_pred.values, (-1, args.num_users))
    truth = np.reshape(df_truth.values, (-1, args.num_users))

    print('Federated MSE: {:.4f}'.format(metrics.mean_squared_error(
        truth.ravel(), pred.ravel()
    )))

    print('Federated MAE: {:.4f}'.format(metrics.mean_absolute_error(
        truth.ravel(), pred.ravel()
    )))

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=300)
    axes[0, 0].plot(pred[:, 0], 'r-', label='pred')
    axes[0, 0].plot(truth[:, 0], 'k-', label='truth')

    axes[0, 1].plot(pred[:, 4], 'r-', label='pred')
    axes[0, 1].plot(truth[:, 4], 'k-', label='truth')

    axes[1, 0].plot(pred[:, 5], 'r-', label='pred')
    axes[1, 0].plot(truth[:, 5], 'k-', label='pred')

    axes[1, 1].plot(train_loss, 'ro-', label='training loss')
    # axes[1, 1].plot(test_loss_list, 'ko-', label='test loss')
    plt.suptitle('Federated results, training loss')
    plt.savefig('Federated_results.png', dpi=300)
    plt.legend()
    plt.show()