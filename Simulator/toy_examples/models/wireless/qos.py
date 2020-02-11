import copy

class Channel:
    """
        data is a tensor
        constraints is a class
    """

    def __init__(self, constraints):
        self.constraints = constraints

class Gaussian(Channel):
    def __init__(self, constraints=None):
        super(Gaussian, self).__init__(constraints)

    def __call__(self, data, *args, **kwargs):
        return data


class QoS:
    def __init__(self, topo, *args, **kwargs):
        self.topo_origin = topo
        self.topo = copy.deepcopy(self.topo_origin)
        # self.update()

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    @property
    def nodes(self):
        return self.topo_origin.nodes


    def removed_nodes(self):
        """
            Users should implement it according to the constraints,
            and generate an iterable object of nodes which needs to be deleted from topo
        """
        raise NotImplementedError

    def removed_edges(self):
        """
            Users should implement it according to the constraints,
            and generate an iterable object of directed edges which needs to be deleted from topo
        """
        raise NotImplementedError

    def update(self, *args, **kwargs):
        self.topo = copy.deepcopy(self.topo_origin)
        self.topo.remove(nodes=self.removed_nodes(), edges=self.removed_edges())


class QoSDemo(QoS):
    def __init__(self, topo):
        super(QoSDemo, self).__init__(topo)

    def removed_nodes(self):
        import random
        ## TODO: generate random removed nodes according qos
        removed_nodes = (node for node in self.nodes if random.uniform(0, 1) > 0.5)
        return removed_nodes

    def removed_edges(self):
        import random
        removed_edges = []
        for node in self.nodes:
            for adj in self.topo_origin.nodes[node]['adj'].keys():
                if random.uniform(0, 1) > 0.5:
                    removed_edges.append((node, adj))
        return removed_edges