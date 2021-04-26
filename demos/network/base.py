from itertools import chain
from abc import ABCMeta, abstractmethod
import os
import yaml
import ns.distributedml as dml
import time

def time_shift(func):
	"""
		This time shift function wrapper is used to modifiy the computing time which ns3 cannot simulate.
		We use the wall-clock time times 30 to denote the computing time for each task.
	"""
	def wrapper(*args, **kwargs):
		t_now = time.time()
		func(*args, **kwargs)
		return 10*(time.time()-t_now)

	return wrapper




def parse_yaml(yaml_path):
	"""
		Given a yaml path, read it and parse it, and outputs the backbonenet and cells
	"""
	path = ""
	if os.path.exists(yaml_path):
		path = yaml_path
	else:
		current_path = os.path.abspath(os.path.dirname(__file__))
		path = current_path + '/../configurations/' + yaml_path
	
	with open(path, 'r') as f:
		configs = yaml.load(f.read())

	all_nodes = list(configs.keys())
	BackBoneNet = {}
	Cells = {}

	for n in all_nodes:
		firstlayer = configs[n]
		if 'role' in firstlayer.keys():
			if firstlayer['role'] == "server":
				BackBoneNet['server']=n
				BackBoneNet['adj']=firstlayer['adj']
				for ap in BackBoneNet['adj'].keys():
					Cells[ap] = configs[ap]

	## consider about the leaf nodes
	for n in list((set(all_nodes).difference(set(BackBoneNet['adj'].keys()))).difference(set([BackBoneNet['server']]))):
		if "adj" in configs[n].keys():
			for adj in configs[n]['adj']:
				if adj in Cells.keys():
					if n not in Cells[adj]['adj']:
						Cells[adj]['adj'].append(configs[n])

	return BackBoneNet, Cells


class __Base(object):
	__metaclass__ = ABCMeta

	def __init__(self, config_flag=True, *args, **kwargs):
		self.config_flag = config_flag
		if self.config_flag:
			self.config()

	@abstractmethod
	def config(self):
		print("called __Base config...")
		pass


class _Base(__Base):
	def __init__(self, config_flag=True, install_flag=True, *args, **kwargs):
		super(_Base, self).__init__(config_flag, *args, **kwargs)
		self.install_flag = install_flag
		if self.install_flag:
			self.install()
		

	def config(self):
		pass

	def install(self):
		pass


class _Iter(object):
	def __init__(self, container, *args, **kwargs):
		self.container = container
		self._iter_idx = 0

	def GetN(self):
		return self.container.GetN()

	def Get(self, idx):
		return self.container.Get(idx)

	def __len__(self):
		return self.GetN()

	def __next__(self):
		if self._iter_idx < self.__len__():
			node = self.container.Get(self._iter_idx)
			self._iter_idx += 1
			return node
		else:
			raise StopIteration

	def __iter__(self):
		return self

	def __getitem__(self, idx):
		try:
			idx = idx % self.GetN()
			return self.container.Get(idx)
		except Exception as e:
			print("WARN:: idx out of bound...") 



class Tracer:
	def __init__(self, prefix="./trace"):
		self.prefix = prefix
		if not os.path.exists(prefix):
			os.makedirs(prefix)

		self.tracer = dml.TraceHelper()

	def trace_wifi_energy_consumation(self, energySource, path=None):
		if path:
			self.tracer.trace_wifi_energy_consumation(energySource, os.path.join(self.prefix, path))
		else:
			self.tracer.trace_wifi_energy_consumation(energySource)

	def trace_ml_energy_consumation(self, energySource, path=None):
		if path:
			self.tracer.trace_ml_energy_consumation(energySource, os.path.join(self.prefix, path))
		else:
			self.tracer.trace_ml_energy_consumation(energySource)

	def trace_mobility(self, node, path=None):
		if path:
			self.tracer.trace_mobility(node, os.path.join(self.prefix, path))
		else:
			self.tracer.trace_mobility(node)

	def trace_cwnd(self, socket, path=None):
		if path:
			self.tracer.trace_cwnd(socket, os.path.join(self.prefix, path))
		else:
			self.tracer.trace_cwnd(socket)

	def trace_drop(self, netdevice, path=None):
		if path:
			self.tracer.trace_drop(netdevice, os.path.join(self.prefix, path))
		else:
			self.tracer.trace_drop(netdevice)

class Mpi:
	rank = 0
	world_size = 1
	m = dml.MpiHelper()
	@staticmethod
	def enable(argv):
		Mpi.m.Enable(argv[0:1])
		Mpi.rank = Mpi.m.GetSystemId ()
		Mpi.world_size = Mpi.m.GetSize ()

	@staticmethod
	def disable():
		Mpi.m.Disable()

	
	





