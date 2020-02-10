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
        self.update()

    def update(self, *args, **kwargs):
        raise NotImplementedError

class QoSDemo(QoS):
    def __init__(self, topo):
        super(QoSDemo, self).__init__(topo)

    def update(self):
        import random
        removed_nodes = []
        nodes = list(self.topo_origin.nodes)
        ## TODO: generate random removed nodes according qos
        for node in nodes:
            if random.uniform(0, 1) > 0.5:
                removed_nodes.append(node)

        self.topo = copy.deepcopy(self.topo_origin)

        # TODO:这里不仅要把节点删去，还要删去其他点与找个点相邻的点
        self.topo.remove_nodes_from(removed_nodes)

