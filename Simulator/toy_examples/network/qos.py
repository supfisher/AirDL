import copy
import torch
import torch.distributed as dist
from .mathwork import *

class Channel:
    """
        % Setting wireless environment parameters *********************************
        N0=1; % Average noise power
        m=1; % Fading figure
        sigma=1; % Shadowing figure
        alpha=2; % Path loss exponent
        % *************************************************************************

        % Setting system configuration parameters *********************************
        N=100; % Number of total clients
        epsilon=1; % Preset time window
        d0=1; % Reference distance
        delta=100; % Cell service radius (spatial distribution range of clients)
        B=1; % Uniform bandwidth
        S=1; % Packet size
        PT=100;% Transmit power in watt: 100W=20dBW; 1000W=30dBW.
        phi=1; % Combined antenna gain

        % *************************************************************************
    """

    def __init__(self, N0=1, m=1, sigma=1, alpha=2, N=100, epsilon=1, d0=1, delta=100, B=1, S=1, PT=100, phi=1, *args, **kwargs):
        self.N0 = N0
        self.m = m
        self.sigma = sigma
        self.alpha = alpha
        self.N = N
        self.epsilon = epsilon
        self.d0 = d0
        self.delta = delta
        self.B = B
        self.S = S
        self.PT = PT
        self.phi = phi

    def __call__(self, *args, **kwargs):
        removed_edges = torch.zeros(self.N)
        for eta in range(self.N):
            d = (self.delta + self.d0) / 2
            PR_bar = self.PT * self.phi * (self.d0 / d) ** self.alpha
            omega = gamrnd(1 / (exp(self.sigma ** 2) - 1),
                           PR_bar * (exp(self.sigma ** 2) - 1) * exp(self.sigma ** 2 / 2))
            G = gamrnd(self.m, omega / self.m)
            SNR = self.PT * G / self.N0
            Rate = self.B * log2(1 + SNR)
            Latency = self.S / Rate
            if Latency > self.epsilon:
                removed_edges[eta] = 1
        return removed_edges


class ChannelDemo(Channel):
    def __init__(self, N0=1, m=1, sigma=1, alpha=2, N=100, epsilon=1, d0=1, delta=100, B=1, S=1, PT=100, phi=1, *args, **kwargs):
        super(ChannelDemo, self).__init__(N0, m, sigma, alpha, N, epsilon, d0, delta, B, S, PT, phi, *args, **kwargs)


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
        self.channel = ChannelDemo(N=len(topo.edges))
        super(QoSDemo, self).__init__(topo)

    def remove_nodes(self):
        removed_nodes = torch.tensor([0,1,0])
        return removed_nodes

    def remove_edges(self):
        return self.channel()
