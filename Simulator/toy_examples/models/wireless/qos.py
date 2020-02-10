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
    def __init__(self, topo):
        self.topo = topo
        self.topo_temp = copy.deepcopy(self.topo)
        self.rank = self.topo.rank
        self.update()

    def update(self):
        import random
        removed_nodes = []
        nodes = list(self.topo.nodes)
        ## TODO: generate random removed nodes according qos
        for node in nodes:
            if random.uniform(0, 1) > 0.5:
                removed_nodes.append(node)

        self.topo_temp = copy.deepcopy(self.topo)
        self.topo_temp.remove_nodes_from(removed_nodes)
