import ns.network
import ns.internet
import ns.energy
import sys
from base import __Base, _Base, _Iter




class Agent(_Base, _Iter):
	def __init__(self, numNodes=1, systemId=0, name="agent", \
					initialEnergy=50000000, updateIntervalS=1, config_flag=True, install_flag=True, *args, **kwargs):
		"""
			Agent is either a single node or a group of nodes have the same attributes
		"""
		self.name = name
		self.numNodes = max(numNodes, 1)
		self.systemId = systemId
		self.initialEnergy = initialEnergy
		self.updateIntervalS = updateIntervalS

		if type(self.name) is list or type(self.name) is set:
			# print("self.name: ", self.name)
			self.nodes = ns.network.NodeContainer()
			for n in self.name:
				agent = Agent(systemId=systemId, name=n, config_flag=False, install_flag=False)
				self.nodes.Add(agent.nodes)
		else:
			self.nodes = ns.network.NodeContainer(self.numNodes, systemId)
		
		super(Agent, self).__init__(config_flag, install_flag)
		_Iter.__init__(self, self.nodes)

		self.node = self.Get(0)

		self._iter_index = 0

	def config(self):
		self.stackHelper = ns.internet.InternetStackHelper()
		
		self.basicSourceHelper = ns.energy.BasicEnergySourceHelper()
		self.basicSourceHelper.Set ("BasicEnergySourceInitialEnergyJ", ns.core.DoubleValue (self.initialEnergy))
		self.basicSourceHelper.Set ("PeriodicEnergyUpdateInterval", ns.core.TimeValue (ns.core.Seconds (self.updateIntervalS)))
		
	def install(self):
		self.stackHelper.Install(self.nodes)
		self._energySources = self.basicSourceHelper.Install(self.nodes)

	@property
	def energySources(self):
		return self._energySources
	

	@property
	def wifiEnergyModels(self):
		return self._wifiEnergyModels

	@property
	def mlEnergyModels(self):
		return self._mlEnergyModels

	@wifiEnergyModels.setter
	def wifiEnergyModels(self, _wifiEnergyModels):
		self._wifiEnergyModels = _wifiEnergyModels

	@mlEnergyModels.setter
	def mlEnergyModels(self, _mlEnergyModels):
		self._mlEnergyModels = _mlEnergyModels


