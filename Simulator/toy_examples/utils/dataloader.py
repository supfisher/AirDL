import torch.distributed as dist
from torch.utils.data import DataLoader

class DataParallel(DataLoader):
    """
        It should be an independent class
    """
    def __init__(self, dataset, topo, *args, **kwargs):
        self.dataset = dataset
        self.topo = topo
        if 'batch_size' in kwargs.keys():
            kwargs['batch_size'] *= len(self.topo.clients_on_device)

        super(DataParallel, self).__init__(dataset, *args, **kwargs)

