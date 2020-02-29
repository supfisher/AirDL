import copy
import torch

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
        loss_list = list(map(lambda d, t, criterion: criterion(d, t, *args, **kwargs), output, target, self.criterion))
        # loss_list = []
        # for d, t, criterion in zip(output, target, self.criterion):
        #     loss_list.append(criterion(d, t, *args, **kwargs))
        return ObjectParallel(loss_list)


class AvgParallel(ObjectParallel):
    def __init__(self, avg_model, args):
        self.len_client = args.clients
        self.model = [copy.deepcopy(avg_model) for _ in range(self.len_client)]
        super(AvgParallel, self).__init__(self.model)

    def __call__(self, data, *args, **kwargs):
        return [model(d) for d, model in zip(data, self.model)]


from .distributed import Distributed, Buffer


class ModelParallel(ObjectParallel):
    """
        This is a DataParallel override based on torch.nn.Moule,
        the gather and scatter function is re-writen with QoS constraints
    """
    def __init__(self, qos, async_flag=False, debug=True):
        self.async_flag = async_flag
        self.debug = debug

        self.qos = qos
        topo = self.qos.topo

        self.buff = Buffer(self.qos)
        self.distributed = Distributed(self.buff)

        # self.module = {key: self.buff.models[key] for key in topo.clients_on_device}
        self.module = {key: self.buff.models[key] for key in topo.nodes_on_device}

        super(ModelParallel, self).__init__(list(self.module.values()))

    def __call__(self, data, *args, **kwargs):

        # for c in self.qos.topo_origin.clients_on_device:
        #     check1 = [(b - param.data).sum() for b, param in zip(self.buff.data[c], self.module[c].parameters())]
        #     print('rank: ', self.qos.topo_origin.rank, c, 'check: ', sum(check1))
        #
        # c0 = self.qos.topo_origin.clients_on_device[0]
        # c1 = self.qos.topo_origin.clients_on_device[1]
        #
        # check01 = [(a - b).sum() for a, b in zip(self.buff.data[c0], self.buff.data[c1])]
        # print('rank: ', self.buff.topo.rank, "before check01: ", sum(check01))

        self.distributed.gather_scatter(async_flag=self.async_flag)

        # c0 = self.qos.topo_origin.clients_on_device[0]
        # c1 = self.qos.topo_origin.clients_on_device[1]
        #
        # check01 = [(a-b).sum() for a, b in zip(self.buff.data[c0], self.buff.data[c1])]
        # print('rank: ', self.qos.topo_origin.rank, "check01: ", sum(check01))
        #
        # check01 = [(a.data - b.data).sum() for a, b in zip(self.module[c0].parameters(), self.module[c1].parameters())]
        # print('rank: ', self.qos.topo_origin.rank, "check02: ", sum(check01))

        return [model(d) for d, model in zip(data, iter(self.module.values()))]

    def aggregate(self):
        if self.qos.topo.rank == self.qos.topo.monitor_rank:
            return self.module[self.qos.topo.servers_on_device[0]]
        else:
            return None

    def stop_condition(self):
        """
            This function checks whether the stop conditions satisfy.
            And it should return the user-defined QoS values
        """
        ##TODO: As the illustration

        return None