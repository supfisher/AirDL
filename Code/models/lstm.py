# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: lstm.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-01-02 (YYYY-MM-DD)
-----------------------------------------------
"""
import torch
from torch import nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.num_layers = args.num_layers
        self.device = 'cuda' if args.gpu else 'cpu'

        self.lstm_layer = nn.LSTM(input_size=self.input_dim,
                                  hidden_size=self.hidden_dim,
                                  num_layers=self.num_layers, batch_first=True)

        self.linear_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        bz = x.size(0)
        h0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)

        self.lstm_layer.flatten_parameters()
        lstm_out, hn = self.lstm_layer(x, (h0, c0))
        y_pred = self.linear_layer(lstm_out[:, -1, :])
        return y_pred