import copy
import torch.distributed as dist

class PackData:
    """
        A data structure for packing communicated data on current computing node (real machine).
        On each computing node, we simulate multiple computing devices (virtual machine). To avoid
        dead lock, we need to packing these data on each simulated computing device, and
        send the packed data to corresponding computing node. The computing node will unpack the
        data for each simulated computing device.
    """
    def __init__(self, data):
        """
        :param data: should be a dict with keys being the virtual computing device ids
                    and values being the corresponding data
        """
        self._data = data

    def pack_data(self):
        return self._data

    def unpack_data(self, packed_data):
        return None


class Buffer:
    def __init__(self, data=None):
        self._data = data
        self._mem = copy.deepcopy(self._data)
        self.count = 1

    def register(self):
        self.count = 1

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        (data, id) = value
        for b, d in zip(self._data[id], data):# TODO should maintain the localtion while changeing the vale
            b -= b
            b += d

    @property
    def mem(self):
        return self._mem

    @mem.setter
    def mem(self, _mem):
        self._mem = _mem

    def update_avg(self, avg_list, new_list):
        return [torch.div(avg * self.count + new, self.count + 1) for avg, new in zip(avg_list, new_list)]

    def recursive_avg(self, id):
        self.data = (self.update_avg(self.mem[id], self.data[id]), id)
        self.count += 1
        self.mem = copy.deepcopy(self.data)


class Distributed:
    def __init__(self, buff, group):
        self.buff = buff
        self.group = group
        self.nodes = self.group.nodes_on_device

    @property
    def rank(self):
        return self.group.topo.rank

    def isend(self, data_dict, socket, **kwargs):
        (self_rank, self_node, dst_rank, dst_node) = socket
        worker = []
        if not self_rank == dst_rank:
            worker += [dist.isend(data, dst=dst_rank, **kwargs) for data in data_dict[self_node]]
        else:
            data_dict[dst_node] = copy.deepcopy(data_dict[self_node])
        return worker

    def irecv(self, data_dict, socket, **kwargs):
        (self_rank, self_node, src_rank, src_node) = socket
        worker = []
        if not self_rank == src_rank:
            worker += [dist.irecv(data, src=src_rank, **kwargs) for data in data_dict[self_node]]
        else:
            data_dict[self_node] = copy.deepcopy(data_dict[src_node])
        return worker

    def send(self, data_dict, socket, **kwargs):
        worker = self.isend(data_dict, socket, **kwargs)
        for w in worker:
            w.wait()

    def recv(self, data_dict, socket, **kwargs):
        worker = self.irecv(data_dict, socket, **kwargs)
        for w in worker:
            w.wait()

    def communicate(self, data_dict, sender_type):
        for node_self in self.nodes:
            for node_adj in self.group.nodes[node_self]['adj'].keys():
                socket = (self.group.nodes[node_self]['rank'], node_self,
                          self.group.nodes[node_adj]['rank'], node_adj)
                if self.group.nodes[node_self]['type'] == sender_type:
                    self.send(data_dict, socket=socket)
                else:
                    self.recv(data_dict, socket=socket)
                    if sender_type == 'client':
                        self.buff.recursive_avg(node_self)

    def gather(self):
        self.communicate(self.buff.data, sender_type='client')

    def scatter(self):
        self.communicate(self.buff.data, sender_type='server')

    def async_gather_scatter(self):
        client_worker = []
        for client in self.group.client_on_device:
            for adj in self.group.nodes[client]['adj'].keys():
                socket = (self.group.nodes[client]['rank'], client,
                          self.group.nodes[adj]['rank'], adj)
                client_worker += self.isend(self.buff.data, socket=socket)

                client_worker += self.irecv(self.buff.data, socket=socket)

        for server in self.group.server_on_device:
            for adj in self.group.nodes[server]['adj'].keys():
                socket = (self.group.nodes[server]['rank'], server,
                          self.group.nodes[adj]['rank'], adj)
                self.recv(self.buff.data, socket=socket)
                ##TODO: Process received data
                self.buff.recursive_avg(server)
                self.send(self.buff.data, socket=socket)

        for w in client_worker:
            w.wait()


class Group:
    def __init__(self, topo, qos):
        self.topo = topo
        self.qos = qos
        self.remove_nodes = []

    def __call__(self, *args, **kwargs):
        import random

        nodes = list(self.topo.nodes)
        ## TODO: generate random removed nodes
        for node in nodes:
            if random.uniform(0, 1) > 0.5:
                self.remove_nodes.append(node)
        print(self.remove_nodes)

    @property
    def nodes(self):
        ## TODO: need to be more efficient
        topo = copy.deepcopy(self.topo)
        topo.remove_nodes_from(self.remove_nodes)
        self.remove_nodes = []
        return topo.nodes

    @property
    def nodes_on_device(self):
        return list(set(self.topo.partitioned[dist.get_rank()]).intersection(set(self.nodes)))

    @property
    def client_on_device(self):
        return [node for node in self.nodes_on_device if self.topo.nodes[node]['type'] == 'client']

    @property
    def server_on_device(self):
        return [node for node in self.nodes_on_device if self.topo.nodes[node]['type'] == 'server']


if __name__ == "__main__":
    import torch
    from network import Topo
    import yaml

    topo = Topo()
    with open('./data/simple_graph.yaml', 'r') as f:
        dict = yaml.load(f)
    topo.load_from_dict(dict)

    data = {'c1': torch.zeros(3), 'c2': torch.ones(3), 'c3': torch.ones(3)*4}
    # data = None
    #
    # if dist.get_rank()==0:
    #     data = {'c1': torch.zeros(3)}
    # elif dist.get_rank()==1:
    #     data = {'c2': torch.ones(3)}
    # elif dist.get_rank()==2:
    #     data = {'c3': torch.ones(3)*4}

    buff = Buffer(data)

    group = Group(topo, qos=None)
    distributed = Distributed(buff, group)

    distributed.gather()
    distributed.scatter()

    print("2: ",distributed.buff.data)

    distributed.buff.register()
    distributed.async_gather_scatter()
    print("1: ", distributed.buff.data)
