import ns.distributedml as dml
import ns.network
import sys
import random
import time
from base import _Iter, time_shift
import copy
"""
	Build the Client
"""
class PyDistributedMlTcpClient(dml.DistributedMlTcpAgent):
	def __init__(self, task, num_clients=3, *args, **kwargs):
		super(PyDistributedMlTcpClient, self).__init__(*args, **kwargs)
		self.task = task

		self.num_clients = num_clients
		self.id = self.task.global_rank+1

		self.SetId(self.id)


	def TriggerLogic(self):
		# print("PYTHON:: Client: {}, self.m_TV.get('count'): {}, at epoch: {}, curret time: {}, wall-clock: {}".format(self.id, self.m_TV.get("count"), self.task.global_step, dml.PyTimer.now("s"), self.task.wall_clock))
		if self.task.global_step == 60:
			sys.exit(0)

		return True
		

	@time_shift
	def Processing(self):
		# print("PYTHON:: calling processing: self.id: ", self.id)
		t_now = time.time()
		self.task.train()

		
	def Sleeping(self):
		return self.task.sleeping_time


class ClientHelperSon(dml.DistributedMlTcpAgentHelper, _Iter):
	def __init__(self, address, tasks, num_packets=5, num_clients=3, packet_size=1024, energy_models=None):
		
		self.address = address
		self.num_packets = num_packets
		self.num_clients = num_clients
		self.packet_size = packet_size
		self.energy_models = energy_models
		self.tasks = tasks
		self.apps = ns.network.ApplicationContainer()

		self._iter_index = 0

		super(ClientHelperSon, self).__init__()
		_Iter.__init__(self, self.apps)

	def Install(self, nodes):
		if self.energy_models is None:
			self.energy_models = [None]*len(self.tasks)
		
		for i, (task, model) in enumerate(zip(self.tasks, self.energy_models)):
			app = self.InstallPriv(nodes.Get(i), task, model)
			self.apps.Add(app)
		
		return self.apps


	def InstallPriv(self, node, task, energy_model=None):
		socket = self.CreateSocket (node)

		app = PyDistributedMlTcpClient(task, num_clients=self.num_clients)
		app.SetAttributes(address=self.address, socket=socket, num_packets=self.num_packets, packet_size=self.packet_size)
		app.SetRole("client")

		if energy_model is not None:
			app.SetEnergy(energy_model)

		data_addr, sizes = task.addrs()
		app.SetModel(data_addr, sizes)

		node.AddApplication (app)
		return app


"""
	Build the Server
"""

class PyDistributedMlTcpServer(dml.DistributedMlTcpAgent):
	def __init__(self, task, num_clients=3, *args, **kwargs):
		super(PyDistributedMlTcpServer, self).__init__(*args, **kwargs)
		self.task = task
		self.num_clients = num_clients
		self.id = 0
		self.SetId(self.id)

	def TriggerLogic(self):
		# print("PYTHON:: Server: {}, self.m_TV.get('count'): {}, at epoch: {}, curret time: {}, wall-clock: {}".format(self.id, self.m_TV.get("count"), self.task.global_step, dml.PyTimer.now("s"), self.task.wall_clock))
		
		if self.task.global_step == 60:
			sys.exit(0)

		if (self.m_TV.get("count") == self.num_clients):
			# print("PYTHON:: Server: {} TriggerLogic at epoch: {}, curret time: {}, wall-clock: {}".format(self.id, self.task.global_step, dml.PyTimer.now("s"), self.task.wall_clock))
			return True
		else:
			return False

	@time_shift
	def Processing(self):	
		# print("PYTHON:: Before Server Processing...\t curret time: {}".format(dml.PyTimer.now("s")))
		self.task.evaluate()

	def Sleeping(self):
		return self.task.sleeping_time


class ServerHelperSon(dml.DistributedMlTcpAgentHelper):
	def __init__(self, address, task, num_packets=5, num_clients=3, packet_size=1024):
		self.address = address
		self.num_packets = num_packets
		self.num_clients = num_clients
		self.packet_size = packet_size

		self.apps = ns.network.ApplicationContainer()
		self.task = task

		super(ServerHelperSon, self).__init__("server")


	def InstallPriv(self, node):      
		serverSocket = self.CreateSocket (node)
		app = PyDistributedMlTcpServer(self.task, num_clients=self.num_clients, id=0)
		app.SetAttributes(address=self.address, socket=serverSocket, num_packets=self.num_packets, num_clients=self.num_clients, packet_size=self.packet_size)
		app.SetRole("server")

		app.EnableBroadcast()

		data_addr, sizes = self.task.addrs()
		app.SetModel(data_addr, sizes)

		# app.SetTrigger(True)
		node.AddApplication (app)
		return app

