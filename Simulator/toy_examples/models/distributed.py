import copy
import torch
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


class DataBase:
    def __init__(self, data=None):
        self._data = data
        self._mem = copy.deepcopy(self._data) ## maintains a history of data

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

    def register(self, *args, **kwargs):
        raise NotImplementedError

    def gather_fn(self, *args, **kwargs):
        raise NotImplementedError

    def scatter_fn(self, *args, **kwargs):
        raise NotImplementedError


class Buffer(DataBase):
    def __init__(self, data):
        super(Buffer, self).__init__(data)

    def register(self):
        self.count = 1

    def update_avg(self, avg_list, new_list):
        return [torch.div(avg * self.count + new, self.count + 1) for avg, new in zip(avg_list, new_list)]

    def gather_fn(self, id):
        self.data = (self.update_avg(self.mem[id], self.data[id]), id)
        self.count += 1
        self.mem = copy.deepcopy(self.data)

    def scatter_fn(self):
        return None


class Distributed:
    def __init__(self, buff, group):
        self.buff = buff
        self.group = group

        self.nodes = self.group.nodes
        self.nodes_on_device = self.group.nodes_on_device
        self.client_on_device = self.group.client_on_device
        self.server_on_device = self.group.server_on_device
        self.rank = self.group.topo.rank

    def isend(self, data_dict, socket, **kwargs):
        (self_rank, self_node, dst_rank, dst_node) = socket
        worker = []
        if not self_rank == dst_rank:
            worker += [dist.isend(data, dst=dst_rank, **kwargs)
                       for data in data_dict[self_node]]
        else:
            data_dict[dst_node] = copy.deepcopy(data_dict[self_node])
        return worker

    def irecv(self, data_dict, socket, **kwargs):
        (self_rank, self_node, src_rank, src_node) = socket
        worker = []
        if not self_rank == src_rank:
            worker += [dist.irecv(data, src=src_rank, **kwargs)
                       for data in data_dict[self_node]]
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
        for node_self in self.nodes_on_device:
            for node_adj in self.nodes[node_self]['adj'].keys():
                socket = (self.nodes[node_self]['rank'], node_self,
                          self.nodes[node_adj]['rank'], node_adj)
                if self.nodes[node_self]['type'] == sender_type:
                    self.send(data_dict, socket=socket)
                else:
                    self.recv(data_dict, socket=socket)
                    if sender_type == 'client':
                        self.buff.gather_fn(node_self)

    def sync_gather_scatter(self):
        client_worker = []
        for client in self.client_on_device:
            for adj in self.nodes[client]['adj'].keys():
                socket = (self.nodes[client]['rank'], client,
                          self.nodes[adj]['rank'], adj)
                client_worker += self.isend(self.buff.data, socket=socket)

                client_worker += self.irecv(self.buff.data, socket=socket)

        for server in self.server_on_device:
            for adj in self.nodes[server]['adj'].keys():
                socket = (self.nodes[server]['rank'], server,
                          self.nodes[adj]['rank'], adj)
                self.recv(self.buff.data, socket=socket)
                ##TODO: Process received data
                self.buff.gather_fn(server)

        for server in self.server_on_device:
            for adj in self.nodes[server]['adj'].keys():
                socket = (self.nodes[server]['rank'], server,
                          self.nodes[adj]['rank'], adj)
                self.send(self.buff.data, socket=socket)

        for w in client_worker:
            w.wait()


    def gather(self):
        self.communicate(self.buff.data, sender_type='client')

    def scatter(self):
        self.communicate(self.buff.data, sender_type='server')

    def async_gather_scatter(self):
        client_worker = []
        for client in self.client_on_device:
            for adj in self.nodes[client]['adj'].keys():
                socket = (self.nodes[client]['rank'], client,
                          self.nodes[adj]['rank'], adj)
                client_worker += self.isend(self.buff.data, socket=socket)

                client_worker += self.irecv(self.buff.data, socket=socket)

        for server in self.server_on_device:
            for adj in self.nodes[server]['adj'].keys():
                socket = (self.nodes[server]['rank'], server,
                          self.nodes[adj]['rank'], adj)
                self.recv(self.buff.data, socket=socket)
                ##TODO: Process received data
                self.buff.gather_fn(server)
                self.send(self.buff.data, socket=socket)

        for w in client_worker:
            w.wait()

    def gather_scatter(self, async_flag):
        self.buff.register()
        if async_flag:
            self.async_gather_scatter()
        else:
            self.gather()
            self.scatter()
        self.group.update()

class Group:
    def __init__(self, topo, qos):
        self.topo = topo
        self.qos = qos
        self.topo_ = None
        self.update()

    def update(self):
        import random
        removed_nodes = []
        nodes = list(self.topo.nodes)
        ## TODO: generate random removed nodes according qos
        for node in nodes:
            if random.uniform(0, 1) > 0.5:
                removed_nodes.append(node)

        self.topo_ = copy.deepcopy(self.topo)
        # self.topo_.remove_nodes_from(removed_nodes)

    @property
    def nodes(self):
        return self.topo_.nodes

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

    buff.register()
    distributed.async_gather_scatter()
    print("1: ", buff.data)

    buff.register()
    distributed.sync_gather_scatter()
    print("2: ", buff.data)



