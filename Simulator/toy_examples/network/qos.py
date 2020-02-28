import copy
import torch.distributed as dist
from .channel import *
from .logging import logger


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

    @classmethod
    def broadcast(cls, tensor, src=0):
        if dist.is_initialized():
            dist.broadcast(tensor, src=src)

    def broadcast_removed_nodes(self):
        self.removed_nodes = torch.zeros(len(self.topo.nodes))
        if self.topo.rank == self.topo.monitor_rank:
            self.removed_nodes = self.remove_nodes()
        if self.topo_origin.is_multiprocess:
            self.broadcast(self.removed_nodes, src=self.topo.monitor_rank)
        removed_nodes = list(self.nodes_from(self.removed_nodes))

        logger.info("My rank is %d, removed_nodes: %s" % (self.topo.rank, str(removed_nodes)))

        self.topo_origin.report('removed_nodes', removed_nodes, 'reset')

        return removed_nodes

    def broadcast_removed_edges(self):
        self.removed_edges = torch.zeros(len(self.topo.edges))
        if self.topo.rank == self.topo.monitor_rank:
            self.removed_edges = self.remove_edges()
        if self.topo_origin.is_multiprocess:
            self.broadcast(self.removed_edges, src=self.topo.monitor_rank)
        removed_edges = list(self.edges_from(self.removed_edges))

        logger.info("My rank is %d, removed_edges: %s" % (self.topo.rank, str(removed_edges)))

        self.topo_origin.report('removed_edges', removed_edges, 'reset')

        return removed_edges

    def update(self, *args, **kwargs):
        self.topo = copy.deepcopy(self.topo_origin)
        self.topo.remove(nodes=self.broadcast_removed_nodes(), edges=self.broadcast_removed_edges())


class QoSDemo(QoS):
    def __init__(self, topo):
        self.channel = ChannelDemo_Perfect(topo)
        super(QoSDemo, self).__init__(topo)

    def remove_nodes(self):
        return self.channel.remove_nodes()

    def remove_edges(self):
        return self.channel.remove_edges()

