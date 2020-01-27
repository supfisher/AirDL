import copy
import torch
import torch.distributed as dist


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
    def __init__(self, criterion):
        self.criterion = [copy.deepcopy(criterion) for _ in range(2)]
        super(CriterionParallel, self).__init__(self.criterion)

    def __call__(self, data, target, *args, **kwargs):
        loss_list = []
        for d, t, criterion in zip(data, target, self.criterion):
            loss_list.append(criterion(d, t, *args, **kwargs))
        return loss_list


class ModelP(ObjectParallel):
    def __init__(self, model):
        self.model = [copy.deepcopy(model) for _ in range(2)]
        super(ModelP, self).__init__(self.model)

    def __call__(self, data):
        return [model(d) for d, model in zip(data, self.model)]


class Buffer:
    def __init__(self, module=None):
        self._data = {node: [param.data for param in m] for node, m in module.items()}
        self.register()

    def register(self):
        self.count = 0

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @data.setter
    def data(self, value):
        (data, id) = value
        for b, d in zip(self._data[id], data):# TODO should maintain the localtion while changeing the vale
            b -= b
            b += d

    def gather_fn(self, data, id):
        self.data = (self.update_avg(self.data[id], data), id)
        self.count += 1
        return data

    def update_avg(self, avg_list, new_list):
        return [torch.div(avg*self.count+new, self.count+1) for avg, new in zip(avg_list, new_list)]

    def scatter_fn(self, data, id):
        self.data = (data, id)
        return data


class ModelParallel(ObjectParallel):
    """
        This is a DataParallel override based on torch.nn.Moule,
        the gather and scatter function is re-writen with QoS constraints
    """
    def __init__(self, module, topo, Qos=None, debug=True):
        self.debug = debug
        self.topo = topo
        print("rank: ", dist.get_rank(), "number of module: ", len(topo.rank[dist.get_rank()]))
        self.module = {node: copy.deepcopy(module) for _, node in enumerate(self.topo.rank[dist.get_rank()])}
        self.parameter = {node: list(m.parameters()) for node, m in self.module.items()}
        print("rank: ", dist.get_rank(), "module: ", self.module)
        self.QoS = Qos

        self.buff = Buffer(self.parameter)
        print("data distributed initialized")
        super(ModelParallel, self).__init__(list(self.module.values()))

    def __call__(self, data, *input, **kwargs):
        self.buff.register()
        self.gather(self.buff.data, self.topo, self.QoS)

        self.scatter(self.buff.data, self.topo, self.QoS)

        if self.debug:
            self.__check__()

        return [model(d) for d, model in zip(data, iter(self.module.values()))]

    def __check__(self):
        for i, node in enumerate(self.topo.rank[dist.get_rank()]):
            if self.topo.nodes[node]['type'] == 'client':
                data, module = self.buff.data[node], self.module[node]
                for d, params in zip(data, module.parameters()):
                    if (d-params.data).sum().abs() != 0:
                        print((d-params.data).sum().abs())
                        print("Client: Something Wrong!!")
            elif self.topo.nodes[node]['type'] == 'server':
                data, module = self.buff.data[node], self.module[node]
                for d, params in zip(data, module.parameters()):
                    if (d-params.data).sum().abs()/d.sum().abs() != 0:
                        print((d-params.data).sum().abs()/d.sum().abs())
                        print("Server: Something Wrong!!")

    @staticmethod
    def send(QoS, data_dict, dst_addr, send_fn=None, **kwargs):
        """if receiver and sender are not in the same process,
                do the real send
            else: do a simulate send
        """
        (self_rank, self_node, dst_rank, dst_node) = dst_addr
        if not self_rank == dst_rank:
            worker = [dist.isend(QoS(data), dst=dst_rank, **kwargs) for data in data_dict[self_node]]
            for w in worker:
                w.wait()


    @staticmethod
    def recv(QoS, data_dict, src_addr, recv_fn=None, **kwargs):
        """if receiver and sender are not in the same process,
                do the real recv
            else: do a simulate recv
        """
        (self_rank, self_node, src_rank, src_node) = src_addr
        if not self_rank == src_rank:
            worker = [dist.irecv(data, src=src_rank, **kwargs) for data in data_dict[self_node]]
            for w in worker:
                w.wait()
        else:
            data_dict[self_node] = copy.deepcopy(data_dict[src_node])
        if recv_fn:
            recv_fn(data_dict[self_node], self_node)

    def communicate(self, data_dict, topo, QoS, sender, recv_fn):
        for node_from in topo.rank[dist.get_rank()]:
            for node_to in topo.nodes[node_from]['adj'].keys():

                addr = (topo.nodes[node_from]['rank'], node_from,
                        topo.nodes[node_to]['rank'], node_to)

                self.send(QoS, data_dict, dst_addr=addr) \
                    if topo.nodes[node_from]['type'] == sender \
                    else self.recv(QoS, data_dict, src_addr=addr, recv_fn=recv_fn)

    def gather(self, data_dict, topo, QoS):
        self.communicate(data_dict, topo, QoS, sender='client', recv_fn=self.buff.gather_fn)

    def scatter(self, data_dict, topo, QoS):
        self.communicate(data_dict, topo, QoS, sender='server', recv_fn=self.buff.scatter_fn)
