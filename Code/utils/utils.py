# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: utils.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-01-02 (YYYY-MM-DD)
-----------------------------------------------
"""
import h5py
import numpy as np
import pandas as pd
import copy
import torch

# for reproducibility
np.random.seed(201910)
torch.manual_seed(2019)


def get_dataset(args):
    df = pd.read_csv(args.file_path, header=0, index_col=0)
    df.fillna(0.0, inplace=True)
    selected_cells = np.random.choice(df.columns, args.num_users, replace=False)
    print(selected_cells)

    # selected_cells = [37]

    # df_cells = np.log1p(df[selected_cells])
    df_cells = df[selected_cells]
    print(df_cells.shape)

    train_data = df_cells.iloc[:-args.test_days * 24]
    test_data = df_cells.iloc[-args.test_days * 24:]

    # normalize the data to zero mean and unit deviation using only train data
    mean_train = train_data.mean(axis=0)
    std_train = train_data.std(axis=0)
    train_data = (train_data - mean_train) / std_train
    test_data = (test_data - mean_train) / std_train

    return train_data, test_data, selected_cells


def process_baseline(args, train_data, test_data):
    train_shifts = [train_data.shift(i) for i in range(1 - args.out_dim, args.window_size + 1, 1)]
    test_shifts = [test_data.shift(j) for j in range(1 - args.out_dim, args.window_size + 1, 1)]

    # next we need put all the BS's data together. there are two ways to achieve this, i.e., concat them along
    # the first dimension (axis=0) or along the second dimension (axis=1). We adopt the first one as BS can only
    # know its own dataset.
    n = len(train_shifts[0])
    train_x_flatten = [tr_shift.values[args.window_size:n-args.out_dim+1].flatten('F') for tr_shift in
                       train_shifts[args.out_dim:]]
    test_x_flatten = [te_shift.values[args.window_size:n-args.out_dim+1].flatten('F') for te_shift in
                      test_shifts[args.out_dim:]]

    df_train_x = pd.DataFrame(train_x_flatten).transpose()
    # df_train_x.dropna(inplace=True)
    df_test_x = pd.DataFrame(test_x_flatten).transpose()
    # df_test_x.dropna(inplace=True)

    df_train_x = df_train_x.values[:, :, np.newaxis]
    df_test_x = df_test_x.values[:, :, np.newaxis]

    train_y_flatten = [tr_shift.values[args.window_size:n-args.out_dim+1].flatten('F') for tr_shift in
                       train_shifts[:args.out_dim]]
    test_y_flatten = [te_shift.values[args.window_size:n-args.out_dim+1].flatten('F') for te_shift in
                      test_shifts[:args.out_dim]]

    df_train_y = pd.DataFrame(train_y_flatten).transpose()
    # df_train_y.dropna(inplace=True)
    df_test_y = pd.DataFrame(test_y_flatten).transpose()
    # df_test_y.dropna(inplace=True)

    df_train_y = df_train_y.values
    df_test_y = df_test_y.values
    print(df_train_x.shape, df_train_y.shape, df_test_x.shape, df_test_y.shape)

    return df_train_x, df_train_y, df_test_x, df_test_y


def process_fed(args, dataset):
    train_shifts = [dataset.shift(i) for i in range(1 - args.out_dim, args.window_size + 1, 1)]
    # test_shifts = [test.shift(j) for j in range(1 - args.out_dim, args.window_size + 1, 1)]

    df_train = pd.concat(train_shifts, axis=1, ignore_index=True)
    # df_test = pd.concat(test_shifts, axis=1, ignore_index=True)
    df_train.dropna(inplace=True)

    train_x, train_y = df_train.iloc[:, args.out_dim:], df_train.iloc[:, :args.out_dim]

    val_days = 1
    train_x, test_x = train_x[:-val_days*24], train_x[-val_days*24:]
    train_y, test_y = train_y[:-val_days*24], train_y[-val_days*24:]

    train_x = train_x.values[:, :, np.newaxis]
    test_x = test_x.values[:, :, np.newaxis]
    train_y = train_y.values
    test_y = test_y.values

    return train_x, train_y, test_x, test_y


def average_weights(w):
    """
    return the average of the weights
    :param w: weight
    :return: averaged weight
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg


def weighted_average_weights(args, w):
    w_total = copy.deepcopy(w[0])
    w_global = copy.deepcopy(w)
    k = len(w)
    # args.alpha = 1.0 / k
    beta = 0.0 if k == 1 else (1 - args.alpha) / (k - 1)

    for key in w_total.keys():
        for i in range(1, args.num_users):
            w_total[key] += w[i][key]

        for j in range(0, args.num_users):
            w_global[j][key] = (args.alpha-beta)*w_global[j][key] + beta*w_total[key]

    return w_global
