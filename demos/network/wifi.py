import sys
import ns.wifi
import ns.internet
import ns.mobility
import ns.point_to_point
import ns.propagation
import ns.distributedml as dml


from node import Agent 
from base import __Base, _Base, _Iter
from base import parse_yaml



class MobilityRandomWalk2d(_Base):
	def __init__(self, ap, sta, attributes=dict(), *args, **kwargs):
		self.ap = ap
		self.sta = sta
		self.attributes = attributes
		super(MobilityRandomWalk2d, self).__init__(*args, **kwargs)


	def config(self):
		self.mobility = ns.mobility.MobilityHelper()
		self.mobility.SetPositionAllocator ("ns3::GridPositionAllocator", "MinX", ns.core.DoubleValue(0.0),
										"MinY", ns.core.DoubleValue (0.0), "DeltaX", ns.core.DoubleValue(5.0), "DeltaY", ns.core.DoubleValue(10.0),
		                                 "GridWidth", ns.core.UintegerValue(3), "LayoutType", ns.core.StringValue("RowFirst"))


	def install(self):
		self.mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel", "Bounds", ns.mobility.RectangleValue(ns.mobility.Rectangle (-10, 10, -10, 10)))
		self.mobility.Install(self.sta.nodes)
		self.mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
		self.mobility.Install(self.ap.nodes)


class EnergyModel(_Base):
	def __init__(self, ap, sta, wifi):
		self.ap = ap
		self.sta = sta
		self.wifi = wifi
		super(EnergyModel, self).__init__()

	def config(self):
		self.radioEnergyHelper = ns.wifi.WifiRadioEnergyModelHelper()
		# self.set("TxCurrentA", ns.core.DoubleValue(0.005))
		# self.set("RxCurrentA", ns.core.DoubleValue(0.0045))
		# self.set("IdleCurrentA", ns.core.DoubleValue(0.00001))
		self.set("TxCurrentA", ns.core.DoubleValue(0.24))
		self.set("RxCurrentA", ns.core.DoubleValue(0.24))
		self.set("IdleCurrentA", ns.core.DoubleValue(0.0001))
		self.set("CcaBusyCurrentA", ns.core.DoubleValue(0.03)) #This is used to denote the computation energy cost

		self.mlHelper = dml.MlDeviceEnergyModelHelper()
		self.mlHelper.Set("BusyCurrent", ns.core.DoubleValue(0.033))
		self.mlHelper.Set("IdleCurrent", ns.core.DoubleValue(0))

	def install(self):
		self.ap.wifiEnergyModels = self.radioEnergyHelper.Install (self.wifi.apDevices, self.ap.energySources)
		self.sta.wifiEnergyModels = self.radioEnergyHelper.Install (self.wifi.staDevices, self.sta.energySources)

		self.sta.mlEnergyModels = self.mlHelper.Install(self.sta.energySources)

	def set(self, name, value):
		self.radioEnergyHelper.Set(name, value)


class WifiYansChannel(_Base):
	def __init__(self, ap, sta, errorRate=0, *args, **kwargs):
		self.ap = ap
		self.sta = sta
		self.errorRate = errorRate
		super(WifiYansChannel, self).__init__(*args, **kwargs)


	def config(self):
		
		# self.channel = ns.wifi.YansWifiChannel()
		# delayModel = ns.propagation.ConstantSpeedPropagationDelayModel()
		# self.channel.SetPropagationDelayModel (delayModel)
		# rssLossModel = ns.propagation.FixedRssLossModel()
		# self.channel.SetPropagationLossModel (rssLossModel)
		# self.Phy = ns.wifi.YansWifiPhyHelper()
		# self.Phy.SetChannel (self.channel)

		# wifiManager = "Aarf" #Set wifi rate manager (Aarf, Aarfcd, Amrr, Arf, Cara, Ideal, Minstrel, MinstrelHt, Onoe, Rraa)
		# rtsThreshold = 999999
		# self.wifi.SetRemoteStationManager ("ns3::" + wifiManager + "WifiManager", "RtsCtsThreshold", ns.core.UintegerValue (rtsThreshold));

		self.channel = ns.wifi.YansWifiChannelHelper.Default()
		self.phy = ns.wifi.YansWifiPhyHelper.Default()
		self.phy.SetChannel(self.channel.Create())

		self.wifi = ns.wifi.WifiHelper()
		self.wifi.SetRemoteStationManager("ns3::AarfWifiManager")

		self.mac = ns.wifi.WifiMacHelper()
		self.ssid = ns.wifi.Ssid ("ns-3-ssid")

		
	def install(self):
		self.mac.SetType ("ns3::StaWifiMac", "Ssid", ns.wifi.SsidValue(self.ssid), "ActiveProbing", ns.core.BooleanValue(False))
		self.staDevices = self.wifi.Install(self.phy, self.mac, self.sta.nodes)
		
		self.mac.SetType ("ns3::ApWifiMac", "Ssid", ns.wifi.SsidValue(self.ssid))
		self.apDevices = self.wifi.Install(self.phy, self.mac, self.ap.nodes)

		if self.errorRate>0:
			em = ns.network.RateErrorModel ()
			em.SetAttribute ("ErrorRate", ns.core.DoubleValue (self.errorRate))
			for i in range(len(self.sta)):
				self.staDevices.Get(i).GetPhy().SetAttribute ("PostReceptionErrorModel", ns.core.PointerValue (em))
			self.apDevices.Get(0).GetPhy().SetAttribute ("PostReceptionErrorModel", ns.core.PointerValue (em))




class WifiCell(__Base, _Iter):
	def __init__(self, nWifi=1, systemId=0, ap=None, sta=None, errorRate=0, addrBase=None, *args, **kwargs):
		# self.tracer = dml.TraceHelper()
		"""
		Agent class is a node wrapper class, it automatically calls the Interner stack configuation function
		"""
		self._rank = systemId
		if ap is not None:
			self.ap = ap
		else:
			self.ap = Agent(1, systemId)
		if sta is not None:
			self.sta = sta
		else:
			self.sta = Agent(nWifi, systemId)
		if addrBase is not None:
			self.addrBase = addrBase
		else:
			print("Please input a valid AddrBase!!")
			sys.exit(1)


		self.wifi = WifiYansChannel(ap=self.ap, sta=self.sta, errorRate=errorRate)
		self.mobility = MobilityRandomWalk2d(ap=self.ap, sta=self.sta)
		self.energy = EnergyModel(ap=self.ap, sta=self.sta, wifi=self.wifi)

		super(WifiCell, self).__init__(*args, **kwargs)
		_Iter.__init__(self, self.sta)
		
	def config(self):
		self.assign_address()


	def assign_address(self):
		address = ns.internet.Ipv4AddressHelper()
		address.SetBase(ns.network.Ipv4Address(self.addrBase), ns.network.Ipv4Mask("255.255.255.0"))
		address.Assign(self.wifi.staDevices)
		address.Assign(self.wifi.apDevices)

	@property
	def apWifiEnergyModels(self):
		return [self.ap.wifiEnergyModels.Get(i) for i in range(self.ap.GetN())]

	@property
	def staWifiEnergyModels(self):
		return [self.sta.wifiEnergyModels.Get(i) for i in range(self.sta.GetN())]

	@property
	def staMlEnergyModels(self):
		return [self.sta.mlEnergyModels.Get(i) for i in range(self.sta.GetN())]

	@property
	def apEnergySources(self):
		return self.ap.energySources

	@property
	def staEnergySources(self):
		return self.sta.energySources

	@property
	def apDevices(self):
		return self.wifi.apDevices

	@property
	def staDevices(self):
		return self.wifi.staDevices

	def GetEnergyLeft(self, ):
		return self.sta.energySources.GetRemainingEnergy()

	@property
	def rank(self):
		return self._rank




class P2PChannel(_Base):
	def __init__(self, agent1, agent2, errorRate=0, dataRate="500Mbps", delay="2ms", addrBase="10.1.2.0", *args, **kwargs):
		self.agent1 = agent1
		self.agent2 = agent2
		self.errorRate = errorRate
		self.dataRate = dataRate
		self.delay = delay
		self.addrBase = addrBase
		super(P2PChannel, self).__init__(*args, **kwargs)


	def config(self):
		self.pointToPoint = ns.point_to_point.PointToPointHelper()
		self.pointToPoint.SetDeviceAttribute("DataRate", ns.core.StringValue(self.dataRate))
		self.pointToPoint.SetChannelAttribute("Delay", ns.core.StringValue(self.delay))


	def install(self):
		self.p2pDevices = self.pointToPoint.Install(self.agent1.node, self.agent2.node)
		self.assign_address()
		if self.errorRate>0:
			em = ns.network.RateErrorModel ()
			em.SetAttribute ("ErrorRate", ns.core.DoubleValue (self.errorRate))
			self.p2pDevices.Get(0).SetAttribute ("ReceiveErrorModel", ns.core.PointerValue (em))


	def assign_address(self):
		address = ns.internet.Ipv4AddressHelper()
		address.SetBase(ns.network.Ipv4Address(self.addrBase), ns.network.Ipv4Mask("255.255.255.0"))
		self.p2pInterfaces = address.Assign(self.p2pDevices)
		


def build_n_wifi_cells(numCells, numStaPerCell, channel={'P2PChannel': {'dataRate': '500Mbps', 'delay': '20ms'}}):
	server = "S0"
	BackBoneNet, Cells = {}, {}
	BackBoneNet['server'] = server
	BackBoneNet['adj'] = {}

	for i in range(1, numCells+1):
		adj = "ap"+str(i)
		BackBoneNet['adj'][adj] = {}
		BackBoneNet['adj'][adj]['channel'] = channel
		Cells[adj] = {}
		Cells[adj]['addrBase'] = '10.1.%d.0'%(i+1)
		Cells[adj]['mobility'] = 'ConstantPositionMobilityModel'
		Cells[adj]['adj'] = {}
		for j in range(1, numStaPerCell+1):
			Cells[adj]['adj']['Sta-%d'%j+adj] = {'channel': 'YansWifiChannel', 'mobility': 'RandomWalk2dMobilityModel'}
		
	return BackBoneNet, Cells



class Network:
	def __init__(self, systemWifi, systemServer, path=None, BackBoneNet=None, Cells=None, errorRate=0):
		self.systemWifi = systemWifi
		self.systemServer = systemServer
		if path is not None:
			self.BackBoneNet, self.Cells = parse_yaml(path)
		else:
			self.BackBoneNet, self.Cells = BackBoneNet, Cells

		self.serverAgent = Agent(systemId=systemServer, name=self.BackBoneNet['server'])
		self.errorRate = errorRate

	def config(self):

		assert len(self.systemWifi) == len(self.Cells.keys()), "The systemWifi should be a list or set and its size should be the same of cells numebr..."
		
		self.wifiCells = []
		
		for i, (ap_name, adjs) in enumerate(self.Cells.items()):
			ap = Agent(systemId=self.systemWifi[i], name=ap_name)
			sta = Agent(systemId=self.systemWifi[i], name=list(adjs['adj'].keys()))
			
			cell = WifiCell(systemId=self.systemWifi[i], ap=ap, sta=sta, errorRate=self.errorRate, addrBase="10.{}.{}.0".format(2*i//256+1, 2*(i+1)%256))
			P2P = P2PChannel(self.serverAgent, ap, errorRate=self.errorRate, addrBase="10.{}.{}.0".format(2*i//256+1, (2*i+1)%256))
			self.wifiCells.append(cell)

		serverAddr = P2P.p2pInterfaces.GetAddress (0)

		ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

		self.sinkPort = 8080
		self.sinkAddress = ns.network.InetSocketAddress (serverAddr, self.sinkPort)

		return self.serverAgent, self.wifiCells, self.sinkPort, self.sinkAddress


















		