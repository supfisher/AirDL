import networkx as nx
import torch.distributed as dist
##TODO: Should I define a class called Node with node attributes?

class Topo(nx.Graph):
    """
        This is a base class for build network topology and it inherients from
        n.graph base class, thus has all the features of nx.graph
    """
    def __init__(self, **kwargs):
        super(Topo, self).__init__(**kwargs)

        dist.init_process_group(backend='mpi')

    def __str__(self):
        msg = 'node: \n'
        for node in self.nodes:
            msg += "  --name: %s, --attrs: %s \n" % (str(node), self.nodes[node])
            msg += "  --adjcency: %s \n" % (self.adj[node])
        return msg

    def load_from_dict(self, dict):
        """
            load a graph from a dict of {'c1': {node_attrs: {}, adj: {'c2': {edge_attrs: }}}}
            The keys() of dict is the nodes,
            the values() of each key is the node attributes and adjacency of the node
        """
        self.add_nodes_from(zip(dict.keys(), dict.values()))

        for node in self.nodes:
            if 'adj' in dict[node].keys():
                adj = dict[node]['adj']
                self.add_edges_from(zip([node]*len(adj), adj.keys(), adj.values()))

        self.partition()
        self.defaults()

    def defaults(self):
        """
            If some attributes are not registered, make it defaults.
        """
        # default type
        for k, v in dict(self.nodes(data='type', default='client')).items():
            self.nodes[k]['type'] = v

        # default edge


    @property
    def servers(self):
        """
            It checks which node is a FL server and return the FL server nodes
        """

        return [k for k, v in dict(self.nodes(data='type', default='client')).items() if v is not 'client']

    @property
    def clients(self):
        """
            It checks which node is a computing device and return the computing device nodes
        """
        return [k for k, v in dict(self.nodes(data='type', default='client')).items() if v is 'client']

    @property
    def rank(self):
        return dist.get_rank()

    @property
    def partitioned(self):
        """
        :return: A list with the indices being the rank,
                    and the values being the nodes on that rank
        """
        return self.__dict__['partitioned']

    def partition(self, world_size=None):
        """
            Partition the topo clients and servers into a distributed version
            default group is the group.world
        """
        if world_size is None:
            world_size = dist.get_world_size()
        partitioned = [[] for _ in range(world_size)]
        for i, node in enumerate(self.nodes):
            if 'rank' not in self.nodes[node].keys():
                self.nodes[node]['rank'] = i % world_size
                partitioned[i % world_size] += [node]
            else:
                partitioned[self.nodes[node]['rank']] += [node]
        self.__dict__['partitioned'] = partitioned

    @property
    def nodes_on_device(self):
        return self.partitioned[self.rank]

    @property
    def clients_on_device(self):
        return [node for node in self.nodes_on_device if self.nodes[node]['type'] == 'client']

    @property
    def servers_on_device(self):
        return [node for node in self.nodes_on_device if self.nodes[node]['type'] == 'server']
