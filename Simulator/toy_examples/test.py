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


class ListFn:
    """
        It will serially process the given list function
    """

    def __init__(self, list_fn):
        self.list_fn = list_fn

    def __call__(self, *args, **kwargs):
        return list(fn(*args, **kwargs) for fn in self.list_fn)


class ObjectParallel:
    def __init__(self, objects=None):
        self.objects = objects
        for key in dir(objects[0]):
            if '__' not in key:
                setattr(self, key, ListFn([getattr(obj, key)
                                           for obj in self.objects]))


class DataParallel:
    """
        Partition dataset
    """

    def __init__(self, dataset, topo, *args, **kwargs):
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
        return sum(len(d) for d in self.dataloaders)

# class DataParallel(DataLoader):
#     """
#         Partition dataset
#     """
#     def __init__(self, dataset, topo, *args, **kwargs):
#         partition_ratio = list(map(lambda i: len(i)/len(topo.clients), topo.clients_partitioned))
#         partition = DataPartitioner(dataset, partition_ratio, seed=topo.seed)
#         self.dataset = partition.use(topo.rank)
#
#         assert 'batch_size' in kwargs.keys()
#         kwargs['batch_size'] *= len(topo.clients_on_device)
#
#         super(DataParallel, self).__init__(self.dataset, *args, **kwargs)


if __name__=='__main__':
    from torchvision import datasets, transforms
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    partition = DataPartitioner(dataset, [[1,2],[3,4],[5,6]])
    partition0 = partition.on(0)
    print(len(partition0))
    partition1 = partition.on(1)
    print(len(partition1))
    partition2 = partition.on(2)
    print(len(partition2))

    model = Net()
    topo = RandTopo(model, backend='None', rank=0, size=1, dist_url=None,
                    rand_method=('static', 5))
    print(topo)
    train_loader = DataParallel(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), topo=topo,
        batch_size=128, shuffle=True)
    print(len(train_loader))
    for i,j in train_loader:
        print(j)

