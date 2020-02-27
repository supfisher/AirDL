# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: dataloader_v2.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-02-27 (YYYY-MM-DD)
-----------------------------------------------
"""
from torch.utils.data import DataLoader
from random import Random
from torch.utils.data import Dataset
from models import *
from network import *
import torch
import numpy as np


class Partition(Dataset):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.index = index

        xs, ys = [], []
        for i in self.index:
            x, y = data[i]
            xs.append(x)
            ys.append(y)

        self.xs, self.ys = np.vstack(xs), np.vstack(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return (self.xs[index], self.ys[index])


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, partition_clients, seed=1234):
        assert type(data) is dict
        indexes = list(data.keys())

        self.data = data
        self.partitions = []

        partition_sizes = [len(clients) for clients in partition_clients]

        part_len = int(len(indexes)/sum(partition_sizes))
        assert part_len > 0
        for frac in partition_sizes:
            self.partitions.append([indexes[i*part_len:(i+1)*part_len] for i in range(frac)])
            indexes = indexes[part_len*frac:]

    def on(self, rank):
        return [Partition(self.data, partition)
                for partition in self.partitions[rank]]

class DataParallel:
    """
        Partition dataset
    """

    def __init__(self, dataset, topo, *args, **kwargs):
        self.dataset = dataset

        partition = DataPartitioner(self.dataset, topo.clients_partitioned, seed=topo.seed)
        self.datasets = partition.on(topo.rank)

        self.dataloaders = [DataLoader(dataset, *args, **kwargs) for dataset in self.datasets]


    def __iter__(self):
        self.itera_time = 0
        return self

    def __next__(self):
        if self.itera_time < self.__len__():
            self.itera_time+=1
            xs = []
            ys = []
            for d in self.dataloaders:
                x,y = iter(d).__next__()
                xs.append(x)
                ys.append(y)

            return torch.stack(xs), torch.stack(ys)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.dataloaders[0])