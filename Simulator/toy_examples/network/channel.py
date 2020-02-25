from .mathwork import *
import torch
import numpy as np

class ChannelBase:
    """
        This is base class for the nodes qos properties.
        Here we only consider the properties involving in
        the node itself.
    """
    def __init__(self, topo):
        self.topo = topo

    def _cal_E(self, node, key):
        """
            It calculates the energy consumed given the node and the reason
        """
        energy = self.topo.nodes[node]['energy']
        energy_cost = self.topo.nodes[node][key]*self.topo.model_size
        energy -= energy_cost
        energy_cost += self.topo.nodes[node]['energy_cost']
        return {'energy': energy, 'energy_cost': energy_cost}

    def check_energy(self, node):
        for key in ['send_P', 'recv_P', 'cal_P']:
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

    def __init__(self, topo, N0=1, m=1, sigma=1, alpha=2, N=100,
                 epsilon=1, d0=1, delta=100, B=1, S=1, PT=100, phi=1, *args, **kwargs):
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
        super(Channel, self).__init__(topo)

    def remove_edges(self, *args, **kwargs):
        removed_edges = torch.zeros(self.N)
        time_cost = []
        for eta in range(self.N):
            d = (self.delta + self.d0) / 2
            PR_bar = self.PT * self.phi * (self.d0 / d) ** self.alpha
            omega = gamrnd(1 / (exp(self.sigma ** 2) - 1),
                           PR_bar * (exp(self.sigma ** 2) - 1) * exp(self.sigma ** 2 / 2))
            G = gamrnd(self.m, omega / self.m)
            SNR = self.PT * G / self.N0
            Rate = self.B * log2(1 + SNR)
            Latency = self.S / Rate
            time_cost.append(Latency)
            if Latency > self.epsilon:
                removed_edges[eta] = 1

        self.topo.report('time_cost', max(time_cost), 'plus')

        return removed_edges


class ChannelDemo(Channel):
    def __init__(self, topo, N0=1, m=1, sigma=1, alpha=2, N=100, epsilon=1, d0=1, delta=100, B=1, S=1, PT=100, phi=1, *args, **kwargs):
        super(ChannelDemo, self).__init__(topo, N0, m, sigma, alpha, N, epsilon, d0, delta, B, S, PT, phi, *args, **kwargs)
