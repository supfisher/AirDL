# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: options.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-01-02 (YYYY-MM-DD)
-----------------------------------------------
"""

import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Federated LSTM')

    parser.add_argument('--file_path', type=str, default='../data/demo_data.h5',
                        help='file path')

    # Federated parameters
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")

    # model parameters
    parser.add_argument('--input-dim', type=int, default=1, metavar='N',
                        help='input dimension (default: 1)')
    parser.add_argument('--hidden-dim', type=int, default=32, metavar='N',
                        help='hidden dimension of LSTM (default: 32)')
    parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                        help='number of layers of LSTM (default: 2)')
    parser.add_argument('--out-dim', type=int, default=1, metavar='N',
                        help='how many steps we predict for the future (default: 1)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for test (default: 8)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='# of epochs for training (default: 300)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Use CUDA for training (default: True)')
    parser.add_argument('--seed', type=int, default=20191015, metavar='S',
                        help='Random seed (default: 20191015)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='Log status every N batches (default: 10)')

    parser.add_argument('--verbose', type=bool, default=True,
                        help='show the status (default: True)')

    # other paramteres
    parser.add_argument('--test-days', type=int, default=7, metavar='N',
                        help='How many days to be test (default: 7)')
    parser.add_argument('--window-size', type=int, default=5, metavar='N',
                        help='Sliding window size (default: 3)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='The proportion of local model to global model (default: 10%)')

    args = parser.parse_args()

    return args