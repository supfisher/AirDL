import copy

class ListFn:
    """
        It will serially process the given list function
    """

    def __init__(self, list_fn):
        self.list_fn = list_fn

    def __call__(self, *args, **kwargs):
        return list(fn(*args, **kwargs) for fn in self.list_fn)


class ObjectParallel:
    def __init__(self, objects=None):
        self.objects = objects
        for key in dir(objects[0]):
            if '__' not in key:
                setattr(self, key, ListFn([getattr(obj, key)
                                           for obj in self.objects]))


class OptimizerParallel(ObjectParallel):
    def __init__(self, optimizer, params, **kwargs):
        self.optimizer = [optimizer(param, **kwargs) for param in params]
        super(OptimizerParallel, self).__init__(self.optimizer)


class CriterionParallel(ObjectParallel):
    def __init__(self, criterion, topo):
        self.len_client = len(topo.clients_on_device)
        self.criterion = [copy.deepcopy(criterion) for _ in range(self.len_client)]
        super(CriterionParallel, self).__init__(self.criterion)

    def __call__(self, output, target, *args, **kwargs):
        target = target.split([int(len(target)/self.len_client) for _ in range(self.len_client)], dim=0)
        loss_list = []
        for d, t, criterion in zip(output, target, self.criterion):
            loss_list.append(criterion(d, t, *args, **kwargs))
        return ObjectParallel(loss_list)


from .distributed import Distributed, Buffer


class ModelParallel(ObjectParallel):
    """
        This is a DataParallel override based on torch.nn.Moule,
        the gather and scatter function is re-writen with QoS constraints
    """
    def __init__(self, topo, QoS, debug=True):
        self.debug = debug
        self.len_client = len(topo.clients_on_device)

        self.module = {node: copy.deepcopy(topo.model) for _, node in enumerate(topo.nodes_on_device)}

        data_dict = {node: list(param.data for param in m.parameters())
                     for node, m in self.module.items()}

        self.qos = QoS(topo=topo)
        self.buff = Buffer(data_dict, self.qos)

        self.distributed = Distributed(self.buff)

        super(ModelParallel, self).__init__(list(self.module.values()))

    def __call__(self, data, *args, **kwargs):
        data = data.split([int(len(data) / self.len_client) for _ in range(self.len_client)], dim=0)

        self.distributed.gather_scatter(async_flag=True)

        return [model(d) for d, model in zip(data, iter(self.module.values()))]

    def stop_condition(self):
        """
            This function checks whether the stop conditions satisfy.
            And it should return the user-defined QoS values
        """
        ##TODO: As the illustration

        return None