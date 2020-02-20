import copy
import torch.distributed as dist
from .channel import *


class QoS:
    def __init__(self, topo, *args, **kwargs):
        self.topo_origin = topo
        self.topo = copy.deepcopy(self.topo_origin)
        self.removed_nodes = None
        self.removed_edges = None
        # self.update()

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    @property
    def nodes(self):
        return self.topo_origin.nodes

    @property
    def edges(self):
        return self.topo_origin.edges

    @staticmethod
    def hot_vector_to_index(hot_vector):
        return filter(lambda i: hot_vector[i] == 1, range(len(hot_vector)))

    def nodes_from(self, hot_vector):
        index = self.hot_vector_to_index(hot_vector)
        return map(lambda i: list(self.nodes)[i], index)

    def edges_from(self, hot_vector):
        index = self.hot_vector_to_index(hot_vector)
        return map(lambda i: list(self.edges)[i], index)

    def remove_nodes(self):
        """
            Users should implement it according to the constraints,
            and generate an iterable object of nodes which needs to be deleted from topo
        """
        raise NotImplementedError

    def remove_edges(self):
        """
            Users should implement it according to the constraints,
            and generate an iterable object of directed edges which needs to be deleted from topo
        """
        raise NotImplementedError

    def broadcast_removed_nodes(self):
        self.removed_nodes = torch.zeros(len(self.topo.nodes))
        if self.topo.rank == self.topo.monitor_rank:
            self.removed_nodes = self.remove_nodes()
        dist.broadcast(self.removed_nodes, src=self.topo.monitor_rank)
        return self.nodes_from(self.removed_nodes)

    def broadcast_removed_edges(self):
        self.removed_edges = torch.zeros(len(self.topo.edges))
        if self.topo.rank == self.topo.monitor_rank:
            self.removed_edges = self.remove_edges()
        dist.broadcast(self.removed_edges, src=self.topo.monitor_rank)
        return self.edges_from(self.removed_edges)

    def update(self, *args, **kwargs):
        self.topo = copy.deepcopy(self.topo_origin)
        self.topo.remove(nodes=self.broadcast_removed_nodes(), edges=self.broadcast_removed_edges())


class QoSDemo(QoS):
    def __init__(self, topo):
        self.channel = ChannelDemo(topo, N=len(topo.edges))
        super(QoSDemo, self).__init__(topo)

    def remove_nodes(self):
        return self.channel.remove_nodes()

    def remove_edges(self):
        return self.channel.remove_edges()
