import torch.nn as nn
import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
import copy
import sys


class Test:
    def __init__(self, i):
        self.v = list([torch.ones([3])*i, torch.ones([3])])


class Buffer:
    def __init__(self, module=None):
        # self._data = {node: [param.data for param in m.parameters()] for node, m in module.items()}
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


class ModelParallel(nn.Module):
    """
        This is a DataParallel override based on torch.nn.Moule,
        the gather and scatter function is re-writen with QoS constraints
    """
    def __init__(self, module, topo, Qos=None, debug=True):
        super(ModelParallel, self).__init__()
        self.debug = debug
        self.topo = topo
        print("rank: ", dist.get_rank(), "number of module: ", len(topo.rank[dist.get_rank()]))
        self.module = {node: copy.deepcopy(module) for _, node in enumerate(self.topo.rank[dist.get_rank()])}
        self.parameter = {node: list(m.parameters()) for node, m in self.module.items()}
        print("rank: ", dist.get_rank(), "module: ", self.module)
        self.QoS = Qos

        # buff = {node: [param.data for param in m.parameters()] for node, m in self.module.items()}
        self.m = {node: Test(i+1) for i, node in enumerate(self.topo.rank[dist.get_rank()])}
        self.buff = Buffer(self.parameter)
        print("data distributed initialized")

    def __call__(self, *input, **kwargs):
        self.buff.register()
        self.gather(self.buff.data, self.topo, self.QoS)

        self.scatter(self.buff.data, self.topo, self.QoS)

        if self.debug:
            self.__check__()

        for i, module in enumerate(self.module.values()):
            return self.parallel_apply(module, self.topo, *input, **kwargs)

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

    @staticmethod
    def parallel_apply(module, topo, *input, **kwargs):
        ##TODO
        for node in topo.rank[dist.get_rank()]:
            if topo.nodes[node]['type'] == 'client':
                return module(*input, **kwargs)


    def parameters(self, recurse=True):
        # params = []
        # for model in self.module.values():
        #     params.append(list(model.parameters()))

        return self.parameter.values()
        # return list(copy.deepcopy(list(module.parameters())) for module in self.module.values())

# class Optim(Optimizer):
#     def __init__(self, *inputs, **kwargs):
#         super(Optim, self).__init__(*inputs, **kwargs)
