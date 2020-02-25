from torch.utils.data import DataLoader
from random import Random
from torch.utils.data import Dataset
from models import *
from network import *
import torch



class Partition(Dataset):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, partition_clients, seed=1234):

        self.data = data
        self.partitions = []

        data_len = len(data)
        indexes = [x for x in range(0, data_len)]

        rng = Random()
        rng.seed(seed)
        rng.shuffle(indexes)

        partition_sizes = [len(clients) for clients in partition_clients]
        part_len = int(data_len/sum(partition_sizes))
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
        partition = DataPartitioner(dataset, topo.clients_partitioned, seed=topo.seed)
        self.datasets = partition.on(topo.rank)
        self.dataloaders = [DataLoader(dataset, *args, **kwargs) for dataset in self.datasets]

    def __iter__(self):
        return self

    def __next__(self):
        xs = []
        ys = []
        for d in self.dataloaders:
            x,y = iter(d).__next__()
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)

    def __len__(self):
        return len(self.dataset)