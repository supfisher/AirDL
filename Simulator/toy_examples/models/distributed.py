import copy
import torch.distributed as dist
import torch


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
    def __init__(self, qos=None):
        self.qos = qos
        model = self.qos.topo.model
        self.models = {node: copy.deepcopy(model) for _, node in enumerate(self.qos.topo.nodes_on_device)}
        self._data = {node: list(param.data for param in m.parameters())
                          for node, m in self.models.items()}
        self._mem = copy.deepcopy(self.data) ## maintains a history of data

    @property
    def topo(self):
        return self.qos.topo

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        (data, id) = value
        for b, d in zip(self.data[id], data):
            b -= b
            b += d

    @property
    def mem(self):
        return self._mem

    @mem.setter
    def mem(self, _mem):
        self._mem = _mem

    @property
    def rank(self):
        return self.topo.rank

    @property
    def nodes(self):
        return self.topo.nodes

    @property
    def nodes_on_device(self):
        return self.topo.nodes_on_device

    @property
    def clients_on_device(self):
        return self.topo.clients_on_device

    @property
    def servers_on_device(self):
        return self.topo.servers_on_device

    def register(self, *args, **kwargs):
        raise NotImplementedError

    def gather_fn(self, *args, **kwargs):
        raise NotImplementedError

    def scatter_fn(self, *args, **kwargs):
        raise NotImplementedError


class Buffer(DataBase):
    def __init__(self, qos):
        super(Buffer, self).__init__(qos)

    def register(self):
        self.qos()
        self.count = 0

    def update_avg(self, avg_list, new_list):
        return [torch.div(avg * self.count + new, self.count + 1) for avg, new in zip(avg_list, new_list)]

    def gather_fn(self, id):
        self.data = (self.update_avg(self.mem[id], self.data[id]), id)
        self.count += 1
        self.mem = copy.deepcopy(self.data)

    def scatter_fn(self):
        return None


class Distributed:
    def __init__(self, buff=None):
        self.buff = buff
        self.register()

    def register(self):
        self.buff.register()
        self.topo = self.buff.topo
        self.nodes = self.topo.nodes
        self.rank = self.topo.rank

    def isend(self, socket, **kwargs):
        (self_rank, self_node, dst_rank, dst_node) = socket
        worker = []
        if not self_rank == dst_rank:
            worker += [dist.isend(data, dst=dst_rank, **kwargs)
                       for data in self.buff.data[self_node]]
        else:
            self.buff.data = (self.buff.data[self_node], dst_node)
        return worker

    def irecv(self, socket, **kwargs):
        (self_rank, self_node, src_rank, src_node) = socket
        worker = []
        if not self_rank == src_rank:
            worker += [dist.irecv(data, src=src_rank, **kwargs)
                       for data in self.buff.data[self_node]]
        else:
            self.buff.data = (self.buff.data[src_node], self_node)
        return worker

    def send(self, socket, **kwargs):
        worker = self.isend(socket, **kwargs)
        for w in worker:
            w.wait()

    def recv(self, socket, **kwargs):
        worker = self.irecv(socket, **kwargs)
        for w in worker:
            w.wait()

    def clients_work(self, clients):
        client_worker = []
        for client in clients:
            for adj in self.topo.out_links(client):
                socket = (self.nodes[client]['rank'], client,
                          self.nodes[adj]['rank'], adj)
                client_worker += self.isend(socket=socket)
            for adj in self.topo.in_links(client):
                socket = (self.nodes[client]['rank'], client,
                          self.nodes[adj]['rank'], adj)
                client_worker += self.irecv(socket=socket)
        return client_worker

    def servers_work(self, servers, async_flag, gather_fn):
        for server in servers:
            for adj in self.topo.in_links(server):
                socket = (self.nodes[server]['rank'], server,
                          self.nodes[adj]['rank'], adj)
                self.recv(socket=socket)
                gather_fn(server)
                if async_flag and adj in self.topo.out_links(server):
                    self.send(socket=socket)

        ## for those clients donot send data to server successfully,
        ## we broadcast it with the final averaged data
        if async_flag:
            for server in servers:
                for adj in set(self.topo.out_links(server)).difference(set(self.topo.in_links(server))):
                    socket = (self.nodes[server]['rank'], server,
                              self.nodes[adj]['rank'], adj)
                    self.send(socket=socket)

        if not async_flag:
            for server in servers:
                for adj in self.topo.out_links(server):
                    socket = (self.nodes[server]['rank'], server,
                              self.nodes[adj]['rank'], adj)
                    self.send(socket=socket)

    def gather_scatter(self, async_flag):
        client_worker = self.clients_work(self.topo.clients_on_device)
        self.servers_work(self.topo.servers_on_device, async_flag, self.buff.gather_fn)
        for w in client_worker:
            w.wait()
        ## must have it to update temporary values and update qos
        self.register()

