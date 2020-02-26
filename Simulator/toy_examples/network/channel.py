from .mathwork import *
import torch
import numpy as np


class ChannelParams:
    """
        % Setting wireless environment parameters *********************************
        N0 % Average noise power
        m % Fading figure
        sigma % Shadowing figure
        alpha % Path loss exponent
        % *************************************************************************

        % Setting system configuration parameters *********************************
        N % Number of total clients
        epsilon % Preset time window
        d0 % Reference distance
        delta % Cell service radius (spatial distribution range of clients)
        B % Uniform bandwidth
        PT % Transmit power in watt: 100W=20dBW; 1000W=30dBW.
        phi % Combined antenna gain

        % *************************************************************************

    """

    def __init__(self, **kwargs):
        self.kb = 1.380649e-23  ## Boltzmann constant
        self.T = 300  ## Temperatur in Kelvin
        self.B = 312.5e3
        self.N0 = self.kb * self.B * self.T
        self.m = 1
        self.sigma = 3.5
        self.alpha = 3
        self.epsilon = 0.1
        self.d0 = 3.5
        self.delta = 100
        self.phi = 1
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def gaussian_params(self):
        param_dict = {
            'N0': self.kb * self.B * self.T, 'm': 1, 'sigma': 3.5,
            'alpha': 3, 'epsilon': 0.1, 'd0': 3.5, 'delta': 100, 'phi': 1
        }
        return param_dict.items()


class ChannelBase:
    """
        This is base class for the nodes qos properties.
        Here we only consider the properties involving in
        the node itself.
    """
    def __init__(self, topo):
        self.topo = topo
        self.nodes = self.topo.nodes
        self.edges = self.topo.edges
        self.model_size = self.topo.model_size

    def _cal_E(self, node, key):
        """
            It calculates the energy consumed given the node and the reason
        """
        energy = self.nodes[node]['energy']
        energy_cost = self.nodes[node][key]*self.model_size
        energy -= energy_cost
        energy_cost += self.nodes[node]['energy_cost']
        return {'energy': energy, 'energy_cost': energy_cost}

    def check_energy(self, node):
        for key in ['com_P', 'cal_P']:
            E = self._cal_E(node, key)
        self.topo.set_node(node=node, attr=E)
        return E

    def remove_nodes(self):
        removed_nodes = torch.zeros(len(self.topo.nodes))
        for i, node in enumerate(self.topo.nodes):
            checked_energy = self.check_energy(node)
            if checked_energy['energy'] <= 0:
                removed_nodes[i] = 1

            self.topo.report('energy_cost', checked_energy['energy_cost'], 'plus')

        return removed_nodes


class Channel(ChannelBase):

    def __init__(self, topo, **kwargs):
        params = ChannelParams().gaussian_params
        self.__dict__.update(params)
        self.__dict__.update(kwargs)

        super(Channel, self).__init__(topo)

    def remove_edges(self):
        removed_edges = torch.zeros(len(self.edges))
        time_cost = []
        for i, edge in enumerate(self.edges):
            PT = self.nodes[edge[0]]['com_P']
            d = (self.delta + self.d0) / 2
            PR_bar = PT * self.phi * (self.d0 / d) ** self.alpha
            omega = gamrnd(1 / (exp(self.sigma ** 2) - 1),
                           PR_bar * (exp(self.sigma ** 2) - 1) * exp(self.sigma ** 2 / 2))
            G = gamrnd(self.m, omega / self.m)
            SNR = PT * G / self.N0
            Rate = self.B * log2(1 + SNR)
            Latency = self.model_size / Rate
            time_cost.append(Latency)
            if Latency > self.epsilon:
                removed_edges[i] = 1

        self.topo.report('time_cost', max(time_cost), 'plus')

        return removed_edges


class ChannelDemo(Channel):
    def __init__(self, topo, **kwargs):
        """
        :param topo:
        :param kwargs: It left an input interface for the self-defined channel params
                    The channel params should wirte like as following:
                    {'B': 312.5e3, 'T': 300, 'm': 1, 'sigma': 3.5, 'alpha': 3,
                    'epsilon': 0.1, 'd0': 3.5, 'delta': 100, 'phi': 1}
                    You can give an dict.items() as input.
        """
        super(ChannelDemo, self).__init__(topo)

class ChannelDemo_Perfect(Channel):
    """
        This is a Channel class, which doesn't care about the wireless loss
        and thus return 0 removed nodes and 0 removed edges.
        You can compare the results with that of ChannelDemo.
    """
    def __init__(self, topo):
        super(ChannelDemo_Perfect, self).__init__(topo)

    def remove_edges(self):
        removed_edges = torch.zeros(len(self.edges))
        return removed_edges

    def remove_nodes(self):
        removed_nodes = torch.zeros(len(self.topo.nodes))
        return removed_nodes